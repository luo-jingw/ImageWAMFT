"""Messages of the G1 policy protocol, the protocol of the FastWAM client in `tmp/g1-client/fastwam/`.

On connect the server sends `ServerMetadata`. Each request is a msgpack map:
  {"image": [cam_left_high, cam_left_wrist, cam_right_wrist] RGB uint8 HWC, "state": float32[16], "prompt": str}
or {"reset": True, "prompt": str}, or a camera preview {"preview": True, "image": [three views]} that is only shown on
the web page (no inference). The response to an observation is `ActionResponse`; a reset and a preview are answered
with an empty map. In the inference-only output mode, and when the action safety check fails, `actions` is a hold
chunk (the request's own state repeated); in the stop mode an observation is answered with an error text frame.
Wire maps are converted to these dataclasses at the server boundary.
"""

from dataclasses import dataclass

import numpy as np

CAMERA_KEYS = ("cam_left_high", "cam_left_wrist", "cam_right_wrist")
STATE_DIM = 16
ACTION_DIM = 16
ACTION_LAYOUT = ("14 arm joint targets in rad (left pitch, roll, yaw, elbow, wrist roll, pitch, yaw, then right), "
                 "left gripper, right gripper; absolute, de-normalized")


@dataclass(frozen=True)
class ObservationRequest:
    views: tuple[np.ndarray, ...]  # RGB uint8 [H, W, 3] in CAMERA_KEYS order
    state: np.ndarray  # float32 [16]: 14 arm joints (rad), left gripper, right gripper
    prompt: str


@dataclass(frozen=True)
class ResetRequest:
    prompt: str


@dataclass(frozen=True)
class PreviewRequest:
    views: tuple[np.ndarray, ...]  # RGB uint8 [H, W, 3] in CAMERA_KEYS order


def parse_views(message: dict) -> tuple[np.ndarray, ...]:
    images = message.get("image")
    if not isinstance(images, (list, tuple)) or len(images) != len(CAMERA_KEYS):
        raise ValueError(f"`image` must be a list of {len(CAMERA_KEYS)} views in order {CAMERA_KEYS}")
    views = []
    for key, view in zip(CAMERA_KEYS, images):
        if not isinstance(view, np.ndarray) or view.dtype != np.uint8 or view.ndim != 3 or view.shape[2] != 3:
            shape = getattr(view, "shape", None)
            raise ValueError(f"view {key} must be uint8 [H, W, 3] RGB, got {type(view).__name__} {shape}")
        views.append(view)
    return tuple(views)


def parse_request(message: object) -> ObservationRequest | ResetRequest | PreviewRequest:
    if not isinstance(message, dict):
        raise ValueError(f"request must be a map, got {type(message).__name__}")
    prompt = message.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError(f"`prompt` must be a string, got {type(prompt).__name__}")
    if message.get("reset", False):
        return ResetRequest(prompt=prompt)
    if message.get("preview", False):
        return PreviewRequest(views=parse_views(message))
    views = parse_views(message)
    state = np.asarray(message.get("state"), dtype=np.float32).reshape(-1)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"`state` must be {STATE_DIM} finite values, got shape {state.shape}")
    return ObservationRequest(views=views, state=state, prompt=prompt)


@dataclass(frozen=True)
class ServerTiming:
    preprocess_ms: float
    text_ms: float
    model_ms: float
    total_ms: float

    def to_message(self) -> dict[str, float]:
        return {"preprocess_ms": self.preprocess_ms, "text_ms": self.text_ms, "model_ms": self.model_ms,
                "total_ms": self.total_ms}


@dataclass(frozen=True)
class ActionResponse:
    actions: np.ndarray  # float32 [H, 16], ACTION_LAYOUT: the model's chunk, or a hold chunk (the request's state)
    timing: ServerTiming
    model: str
    checkpoint_step: int
    num_inference_steps: int
    seed: int | None
    output_mode: str  # "execute" or "inference_only"
    actions_source: str  # "model" or "hold"
    safety_ok: bool
    safety_reasons: tuple[str, ...]

    def to_message(self) -> dict[str, object]:
        return {"actions": np.ascontiguousarray(self.actions, dtype=np.float32),
                "server_timing": self.timing.to_message(), "model": self.model,
                "checkpoint_step": self.checkpoint_step, "num_inference_steps": self.num_inference_steps,
                "seed": -1 if self.seed is None else self.seed, "output_mode": self.output_mode,
                "actions_source": self.actions_source, "safety_ok": self.safety_ok,
                "safety_reasons": list(self.safety_reasons)}


@dataclass(frozen=True)
class ServerMetadata:
    model: str
    checkpoint_step: int
    action_horizon: int
    fps: int
    output_mode: str

    def to_message(self) -> dict[str, object]:
        return {"policy": "ImageWAM", "model": self.model, "checkpoint_step": self.checkpoint_step,
                "camera_keys": list(CAMERA_KEYS), "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
                "action_horizon": self.action_horizon, "action_layout": ACTION_LAYOUT, "fps": self.fps,
                "output_mode": self.output_mode}
