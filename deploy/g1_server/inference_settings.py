"""Inference settings applied to every request until changed on the web page."""

from dataclasses import dataclass

TRAINED_ACTION_HORIZON = 16


@dataclass(frozen=True)
class InferenceSettings:
    num_inference_steps: int = 10  # action denoising steps; 10 as in training validation
    sigma_shift: float | None = None  # None: the run's scheduler `infer_shift`
    seed: int | None = 42  # initial action noise seed; None draws a new seed per request
    action_horizon: int = TRAINED_ACTION_HORIZON  # actions per chunk; trained with 16
    prompt_override: str | None = None  # task string used instead of the client's prompt

    def validated(self) -> "InferenceSettings":
        if not 1 <= self.num_inference_steps <= 100:
            raise ValueError(f"num_inference_steps must be in [1, 100], got {self.num_inference_steps}")
        if self.sigma_shift is not None and self.sigma_shift <= 0:
            raise ValueError(f"sigma_shift must be positive, got {self.sigma_shift}")
        if not 1 <= self.action_horizon <= 64:  # the action expert's max_action_horizon
            raise ValueError(f"action_horizon must be in [1, 64], got {self.action_horizon}")
        if self.prompt_override is not None and not self.prompt_override.strip():
            raise ValueError("prompt_override must be None or a non-empty task string")
        return self
