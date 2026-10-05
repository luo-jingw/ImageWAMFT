"""Owner of the resident policies, the active policy, the inference settings and the request log.

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

from g1_server.g1_policy import G1Policy, PolicyOutput
from g1_server.inference_settings import InferenceSettings
from g1_server.model_registry import spec_by_name
from g1_server.protocol import ObservationRequest, ServerMetadata

DATASET_FPS = 30
LOG_LENGTH = 50


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

    def active_policy(self) -> G1Policy:
        with self._state_lock:
            if self._active is None:
                raise RuntimeError("no model is loaded; load one on the web page")
            return self._policies[self._active]

    def infer(self, request: ObservationRequest, client: str) -> PolicyOutput:
        with self._gpu_lock:
            policy = self.active_policy()
            settings = self.settings()
            output, error = None, None
            try:
                output = policy.infer(request, settings)
                return output
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                with self._state_lock:
                    self._served += output is not None
                    self._failed += output is None
                    self._log.append(RequestRecord(
                        index=self._served + self._failed, wall_time=time.time(), client=client,
                        model=policy.spec.name, step=policy.step, settings=settings, state=request.state,
                        views=request.views, output=output, error=error))

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
                              action_horizon=self.settings().action_horizon, fps=DATASET_FPS)

    def status(self) -> ManagerStatus:
        with self._state_lock:
            resident = tuple(ResidentPolicy(name, policy.spec.label, policy.step, policy.load_seconds)
                             for name, policy in self._policies.items())
            return ManagerStatus(active=self._active, resident=resident, settings=self._settings, busy=self._busy,
                                 requests_served=self._served, requests_failed=self._failed,
                                 gpu_memory_allocated_gib=torch.cuda.memory_allocated(self.device) / 2**30,
                                 gpu_memory_reserved_gib=torch.cuda.memory_reserved(self.device) / 2**30)

    def recent(self, n: int) -> list[RequestRecord]:
        with self._state_lock:
            return list(self._log)[-n:]

    def tasks(self) -> list[str]:
        """Training task strings of the active model ([] when none is loaded)."""
        with self._state_lock:
            return [] if self._active is None else list(self._policies[self._active].tasks)
