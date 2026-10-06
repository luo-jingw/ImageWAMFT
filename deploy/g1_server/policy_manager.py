"""Owner of the resident policies, the active policy, the inference settings, the serving controls (output mode,
action safety check) and the request log.

GPU work (loading, unloading, inference, image preview) is serialized by one lock, so a request that arrives during a
model load waits for it. Bookkeeping reads (status, settings, recent requests) use a separate short lock and never
wait for GPU work.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from g1_server.action_safety import SafetyVerdict, hold_chunk
from g1_server.g1_policy import G1Policy, PolicyOutput
from g1_server.inference_settings import InferenceSettings
from g1_server.model_registry import spec_by_name
from g1_server.protocol import ObservationRequest, PreviewRequest, ServerMetadata
from g1_server.serving_controls import OutputMode, ServingControls

DATASET_FPS = 30
LOG_LENGTH = 50


class ServingStopped(RuntimeError):
    """An observation arrived in the stop mode: nothing is inferred or sent."""


@dataclass(frozen=True)
class ServedChunk:
    actions: np.ndarray  # float32 [H, 16] sent to the robot
    source: str  # "model" or "hold"
    mode: OutputMode
    output: PolicyOutput  # the inference, also when a hold chunk was sent
    verdict: SafetyVerdict


@dataclass(frozen=True)
class LoadReport:
    name: str
    step: int
    seconds: float
    gpu_memory_allocated_gib: float


@dataclass(frozen=True)
class RequestRecord:
    index: int
    wall_time: float  # time.time() at completion
    client: str
    model: str
    step: int
    settings: InferenceSettings
    state: np.ndarray  # raw [16]
    views: tuple[np.ndarray, ...]  # as received
    output: PolicyOutput | None
    error: str | None
    preview_image: np.ndarray | None = None  # model input of a camera preview (no inference), uint8 [288, 256, 3]
    mode: OutputMode | None = None  # output mode the reply was decided under
    actions_source: str | None = None  # "model", "hold" or None (no reply: error or stop)
    verdict: SafetyVerdict | None = None


@dataclass(frozen=True)
class ResidentPolicy:
    name: str
    label: str
    step: int
    load_seconds: float


@dataclass(frozen=True)
class ManagerStatus:
    active: str | None
    resident: tuple[ResidentPolicy, ...]
    settings: InferenceSettings
    busy: str | None  # GPU work in progress, e.g. "loading flux2_9b step 15000"
    requests_served: int
    requests_failed: int
    previews: int
    controls: ServingControls
    holds_sent: int  # replies that carried a hold chunk instead of the model's actions
    safety_flags: int  # inferred chunks that failed the safety check
    stopped: int  # observations rejected in the stop mode
    gpu_memory_allocated_gib: float
    gpu_memory_reserved_gib: float


class PolicyManager:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._gpu_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._policies: dict[str, G1Policy] = {}
        self._active: str | None = None
        self._settings = InferenceSettings()
        self._log: deque[RequestRecord] = deque(maxlen=LOG_LENGTH)
        self._busy: str | None = None
        self._served = 0
        self._failed = 0
        self._previews = 0
        self._sequence = 0  # index of the next request record
        self._controls = ServingControls()
        self._holds = 0
        self._safety_flags = 0
        self._stopped = 0

    def _set_busy(self, text: str | None) -> None:
        with self._state_lock:
            self._busy = text

    def load(self, name: str, step: int) -> LoadReport:
        """Loads `name` at `step` (replacing a resident copy at another step); activates it if none is active."""
        spec = spec_by_name(name)
        with self._gpu_lock:
            self._set_busy(f"loading {name} step {step}")
            try:
                with self._state_lock:
                    previous = self._policies.pop(name, None)
                if previous is not None:
                    previous.unload()
                policy = G1Policy.load(spec, step, self.device)
                with self._state_lock:
                    self._policies[name] = policy
                    if self._active is None or self._active == name:
                        self._active = name
            finally:
                self._set_busy(None)
        return LoadReport(name=name, step=step, seconds=policy.load_seconds,
                          gpu_memory_allocated_gib=torch.cuda.memory_allocated(self.device) / 2**30)

    def activate(self, name: str) -> None:
        with self._state_lock:
            if name not in self._policies:
                raise ValueError(f"{name} is not loaded")
            self._active = name

    def unload(self, name: str) -> None:
        with self._gpu_lock:
            with self._state_lock:
                policy = self._policies.pop(name, None)
                if self._active == name:
                    self._active = next(iter(self._policies), None)
            if policy is not None:
                policy.unload()

    def settings(self) -> InferenceSettings:
        with self._state_lock:
            return self._settings

    def update_settings(self, settings: InferenceSettings) -> None:
        settings = settings.validated()
        with self._state_lock:
            self._settings = settings

    def controls(self) -> ServingControls:
        with self._state_lock:
            return self._controls

    def update_controls(self, controls: ServingControls) -> None:
        controls = controls.validated()
        with self._state_lock:
            self._controls = controls

    def active_policy(self) -> G1Policy:
        with self._state_lock:
            if self._active is None:
                raise RuntimeError("no model is loaded; load one on the web page")
            return self._policies[self._active]

    def _reject_stopped(self, request: ObservationRequest, client: str, policy: G1Policy) -> None:
        with self._state_lock:
            self._stopped += 1
            self._sequence += 1
            self._log.append(RequestRecord(
                index=self._sequence, wall_time=time.time(), client=client, model=policy.spec.name, step=policy.step,
                settings=self._settings, state=request.state, views=request.views, output=None,
                error="stop mode: not inferred, no actions sent", mode=OutputMode.STOP))
        raise ServingStopped("the policy server is in STOP mode: no inference and no actions; "
                             "switch the output mode on the web page")

    def serve(self, request: ObservationRequest, client: str) -> ServedChunk:
        """Infers and decides what the robot receives. The controls are read again after inference, so a switch to
        inference-only or stop during an inference already applies to its reply."""
        if self.controls().mode is OutputMode.STOP:
            self._reject_stopped(request, client, self.active_policy())
        with self._gpu_lock:
            policy = self.active_policy()
            settings = self.settings()
            output, error, served = None, None, None
            try:
                output = policy.infer(request, settings)
                controls = self.controls()
                if controls.mode is OutputMode.STOP:
                    self._reject_stopped(request, client, policy)
                verdict = policy.safety_limits.check(output.actions, request.state, controls)
                if controls.mode is OutputMode.EXECUTE and verdict.ok:
                    actions, source = output.actions, "model"
                else:
                    actions, source = hold_chunk(request.state, output.actions.shape[0]), "hold"
                served = ServedChunk(actions=actions, source=source, mode=controls.mode, output=output,
                                     verdict=verdict)
                return served
            except ServingStopped:
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                if served is not None or error is not None:
                    with self._state_lock:
                        self._served += served is not None
                        self._failed += served is None
                        if served is not None:
                            self._holds += served.source == "hold"
                            self._safety_flags += not served.verdict.ok
                        self._sequence += 1
                        self._log.append(RequestRecord(
                            index=self._sequence, wall_time=time.time(), client=client,
                            model=policy.spec.name, step=policy.step, settings=settings, state=request.state,
                            views=request.views, output=output, error=error,
                            mode=None if served is None else served.mode,
                            actions_source=None if served is None else served.source,
                            verdict=None if served is None else served.verdict))

    def preview(self, request: PreviewRequest, client: str) -> None:
        """Records camera views for the web page with their model-input image; runs no inference."""
        with self._gpu_lock:
            policy = self.active_policy()
            image = policy.preprocessor.to_uint8_image(policy.preprocessor.image(request.views))
        with self._state_lock:
            self._previews += 1
            self._sequence += 1
            self._log.append(RequestRecord(
                index=self._sequence, wall_time=time.time(), client=client, model=policy.spec.name, step=policy.step,
                settings=self._settings, state=np.zeros(0, dtype=np.float32), views=request.views, output=None,
                error=None, preview_image=image))

    def predict_image(self, request: ObservationRequest) -> np.ndarray:
        with self._gpu_lock:
            self._set_busy("predicting the image 16 steps ahead")
            try:
                return self.active_policy().predict_image(request, self.settings())
            finally:
                self._set_busy(None)

    def metadata(self) -> ServerMetadata:
        policy = self.active_policy()
        return ServerMetadata(model=policy.spec.label, checkpoint_step=policy.step,
                              action_horizon=self.settings().action_horizon, fps=DATASET_FPS,
                              output_mode=self.controls().mode.value)

    def status(self) -> ManagerStatus:
        with self._state_lock:
            resident = tuple(ResidentPolicy(name, policy.spec.label, policy.step, policy.load_seconds)
                             for name, policy in self._policies.items())
            return ManagerStatus(active=self._active, resident=resident, settings=self._settings, busy=self._busy,
                                 requests_served=self._served, requests_failed=self._failed, previews=self._previews,
                                 controls=self._controls, holds_sent=self._holds, safety_flags=self._safety_flags,
                                 stopped=self._stopped,
                                 gpu_memory_allocated_gib=torch.cuda.memory_allocated(self.device) / 2**30,
                                 gpu_memory_reserved_gib=torch.cuda.memory_reserved(self.device) / 2**30)

    def recent(self, n: int) -> list[RequestRecord]:
        with self._state_lock:
            return list(self._log)[-n:]

    def tasks(self) -> list[str]:
        """Training task strings of the active model ([] when none is loaded)."""
        with self._state_lock:
            return [] if self._active is None else list(self._policies[self._active].tasks)
