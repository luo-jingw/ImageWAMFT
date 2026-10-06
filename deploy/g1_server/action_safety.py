"""Safety check of an action chunk before it is sent to the robot, and the hold chunk sent instead (plan.md D21).

Checked against the run's training data (`dataset_stats.json` → `action.default.global_min/max`):
  - range: every value within [min - m * (max - min), max + m * (max - min)], m = range_margin;
  - first-step jump: the largest arm-joint (0-13) difference between the first action and the request's state is at
    most max_first_step_jump (rad). Grippers are excluded: they open and close in single steps in the data.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from g1_server.serving_controls import ServingControls

ARM_JOINTS = slice(0, 14)


@dataclass(frozen=True)
class SafetyVerdict:
    ok: bool
    reasons: tuple[str, ...]  # empty when ok
    max_first_step_jump: float  # rad, arm joints
    out_of_range_values: int


@dataclass(frozen=True)
class ActionSafetyLimits:
    low: np.ndarray  # [16] training action minimum
    high: np.ndarray  # [16] training action maximum

    @classmethod
    def from_dataset_stats(cls, path: Path) -> "ActionSafetyLimits":
        stats = json.loads(path.read_text())["action"]["default"]
        return cls(low=np.asarray(stats["global_min"], dtype=np.float32),
                   high=np.asarray(stats["global_max"], dtype=np.float32))

    def check(self, actions: np.ndarray, state: np.ndarray, controls: ServingControls) -> SafetyVerdict:
        jump = float(np.abs(actions[0, ARM_JOINTS] - state[ARM_JOINTS]).max())
        if not controls.safety_check:
            return SafetyVerdict(ok=True, reasons=(), max_first_step_jump=jump, out_of_range_values=0)
        span = self.high - self.low
        low, high = self.low - controls.range_margin * span, self.high + controls.range_margin * span
        outside = (actions < low) | (actions > high) | ~np.isfinite(actions)
        reasons = []
        if outside.any():
            dims = sorted({int(d) for d in np.nonzero(outside)[1]})
            reasons.append(f"{int(outside.sum())} values outside the training range (dims {dims})")
        if jump > controls.max_first_step_jump:
            reasons.append(f"first-step arm jump {jump:.3f} rad > {controls.max_first_step_jump:.3f} rad")
        return SafetyVerdict(ok=not reasons, reasons=tuple(reasons), max_first_step_jump=jump,
                             out_of_range_values=int(outside.sum()))


def hold_chunk(state: np.ndarray, horizon: int) -> np.ndarray:
    """The request's own state repeated: the robot holds its current pose."""
    return np.tile(np.asarray(state, dtype=np.float32), (horizon, 1))
