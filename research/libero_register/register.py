"""Training-free test-time registers for the video expert of ImageWAM-FLUX.2 (after Jiang et al., 2025, "Vision
Transformers Don't Need Trained Registers").

In this model the attention sinks are the prompt-template tokens. The text MLP of the last double block
(`REGISTER_LAYER`) writes a massive update into four residual channels of those tokens; in the single blocks they then
act as near-constant keys with small values. A test-time register moves that update to added tokens:
  - `registers` zero tokens are appended to the text context, right after the proprio token (contiguous valid prefix,
    text RoPE positions continue);
  - shift "neurons" (as in the paper): in the post-activation output of the register-layer text MLP, each of the top-k
    register neurons (`find_neurons.py`) is set to its largest-magnitude value over the valid text tokens on the
    register tokens, and to 0 on every other token;
  - shift "direction" (adapted, because the update is spread over hundreds of neurons): the component of the MLP output
    along the massive direction d is removed from every text token, and the largest-magnitude component over the valid
    text tokens is put on the register tokens.
`ablate` removes the neurons or the direction everywhere without adding tokens; `shift="none"` adds tokens only.
`enabled = False` turns an installed register off (paired full-model inference).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REGISTER_LAYER = 4  # last double block of FLUX.2-klein-4B
NEURONS_FILE = HERE / "runs" / "register_neurons.json"
DIRECTION_FILE = HERE / "runs" / "register_direction.npy"


@dataclass(frozen=True)
class RegisterConfig:
    name: str
    registers: int = 0  # zero-initialized tokens appended after the proprio token
    shift: str = "none"  # "none" | "neurons" | "direction"
    neurons: int = 0  # top-ranked register neurons used by shift="neurons"
    ablate: bool = False  # remove the neurons / direction everywhere instead of moving them to registers


def neuron_counts() -> dict[str, int]:
    """Neurons needed for 50%, 90% and 99% of the massive update on the template tokens."""
    share = json.loads(NEURONS_FILE.read_text())["neurons_for_share"]
    return {"50": share["0.5"], "90": share["0.9"], "99": share["0.99"]}


def configs() -> dict[str, RegisterConfig]:
    grid = [RegisterConfig("tokens_r1", registers=1), RegisterConfig("tokens_r16", registers=16),
            RegisterConfig("ablate_dir", shift="direction", ablate=True)]
    grid += [RegisterConfig(f"ttr_dir_r{r}", registers=r, shift="direction") for r in (1, 4, 16)]
    for share, k in neuron_counts().items():
        grid.append(RegisterConfig(f"ablate_n{share}", shift="neurons", neurons=k, ablate=True))
        grid += [RegisterConfig(f"ttr_n{share}_r{r}", registers=r, shift="neurons", neurons=k) for r in (1, 4, 16)]
    return {c.name: c for c in grid}


class Registers:
    """Installed test-time registers; `.remove()` restores the model. `positions` is the [start, stop) range of the
    register tokens in the text stream of the latest call."""

    def __init__(self, model, config: RegisterConfig):
        if config.shift not in ("none", "neurons", "direction"):
            raise ValueError(f"unknown shift {config.shift!r}")
        if config.shift != "none" and not config.ablate and not config.registers:
            raise ValueError("moving the update requires registers")
        self.model, self.config, self.enabled = model, config, True
        self.positions: tuple[int, int] = (0, 0)
        self._handles = []
        block = model.mot.mixtures["video"].double_blocks[REGISTER_LAYER]
        if config.shift == "neurons":
            ranked = json.loads(NEURONS_FILE.read_text())["ranked_neurons"]
            self.neurons = torch.tensor(ranked[: config.neurons], dtype=torch.long)
            self._handles.append(block.txt_mlp[1].register_forward_hook(self._shift_neurons))
        elif config.shift == "direction":
            self.direction = torch.from_numpy(np.load(DIRECTION_FILE)).float()
            self._handles.append(block.txt_mlp[2].register_forward_hook(self._shift_direction))
        if config.registers:
            original = model._append_proprio_to_context_if_enabled

            def append(*args, **kwargs):
                context, mask = original(*args, **kwargs)
                return self._append(context, mask) if self.enabled else (context, mask)

            model._append_proprio_to_context_if_enabled = append

    def _append(self, context: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if context.shape[0] != 1:
            raise ValueError("test-time registers support batch size 1")
        start = int(mask[0].sum())  # valid prefix: text tokens + proprio
        if not bool(mask[0, :start].all()):
            raise ValueError("expected a contiguous valid text prefix")
        r = self.config.registers
        context = torch.cat([context[:, :start], context.new_zeros(1, r, context.shape[-1]), context[:, start:]], dim=1)
        mask = torch.cat([mask[:, :start], mask.new_ones(1, r), mask[:, start:]], dim=1)
        self.positions = (start, start + r)
        return context, mask

    def _shift_neurons(self, module, inputs, act: torch.Tensor):
        if not self.enabled:
            return None
        idx = self.neurons.to(act.device)
        values = act[:, :, idx]  # [1, Ltxt, k]
        act = act.clone()
        act[:, :, idx] = 0
        if not self.config.ablate:
            start, stop = self.positions
            valid = values[:, :start]
            peak = valid.gather(1, valid.abs().argmax(dim=1, keepdim=True))  # [1, 1, k], signed
            act[:, start:stop, idx] = peak.expand(-1, stop - start, -1)
        return act

    def _shift_direction(self, module, inputs, out: torch.Tensor):
        if not self.enabled:
            return None
        d = self.direction.to(device=out.device)
        proj = out.float() @ d  # [1, Ltxt]
        shifted = out.float() - proj[..., None] * d
        if not self.config.ablate:
            start, stop = self.positions
            valid = proj[:, :start]
            peak = valid.gather(1, valid.abs().argmax(dim=1, keepdim=True))  # [1, 1]
            shifted[:, start:stop] += peak[..., None] * d
        return shifted.to(out.dtype)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        if self.config.registers:
            del self.model._append_proprio_to_context_if_enabled  # drop the instance override


def install(model, config: RegisterConfig) -> Registers:
    return Registers(model, config)
