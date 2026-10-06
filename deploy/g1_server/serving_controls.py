"""Serving controls: what the server sends to the robot for each request (plan.md D20, D21)."""

from dataclasses import dataclass
from enum import Enum


class OutputMode(str, Enum):
    EXECUTE = "execute"  # infer and send the model's chunk if it passes the safety check
    INFERENCE_ONLY = "inference_only"  # infer, show on the page, send a hold chunk (the request's own state)
    STOP = "stop"  # no inference; reply with an error, so the client stops and returns arm authority


@dataclass(frozen=True)
class ServingControls:
    mode: OutputMode = OutputMode.INFERENCE_ONLY  # the startup default never moves the robot
    safety_check: bool = True
    range_margin: float = 0.10  # widen the training action range by this fraction of it on each side
    max_first_step_jump: float = 0.30  # rad; largest arm-joint difference between the first action and the state

    def validated(self) -> "ServingControls":
        if not 0.0 <= self.range_margin <= 1.0:
            raise ValueError(f"range_margin must be in [0, 1], got {self.range_margin}")
        if not 0.0 < self.max_first_step_jump <= 3.2:
            raise ValueError(f"max_first_step_jump must be in (0, 3.2] rad, got {self.max_first_step_jump}")
        return self
