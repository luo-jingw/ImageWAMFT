"""Simulated (quantize-dequantize) weight/activation quantization of the ImageWAM-FLUX.2 transformer linears.

Quantized layers: every linear inside the transformer blocks of both experts (video expert: double-block img/txt qkv,
proj and MLP, single-block linear1 and linear2; action expert: the same block linears). Embedders, modulation
linears, final layers, attention score/value products, the VAE and the text encoder stay in bfloat16.

For y = x W^T with input transform T (T = diag(1/s) for SmoothQuant, then R for rotation):
    y = (x T) (W T^{-T})^T,  quantize(x T) per token,  quantize(W T^{-T}) per output channel.
  - rotation (QuaRot-style incoherence processing): R is block-diagonal with 1024 × 1024 randomized Hadamard blocks,
    R = diag(H D / 32), D a fixed random sign vector, so T^{-T} = R. Full QuaRot also rotates the residual stream;
    FLUX.2 blocks normalize with LayerNorm and per-channel adaLN modulation, which is not rotation invariant, so only
    the per-linear (online) rotation is used;
  - SmoothQuant: s_j = max|X_j|^alpha / max|W_j|^(1 - alpha) from calibration activations, T^{-T} = diag(s).
Weights: symmetric per output channel, clipping ratio searched per channel for minimum squared error. Activations:
symmetric per token (dynamic), clipping ratio `a_clip`. b bits use the integers -(2^(b-1) - 1) … 2^(b-1) - 1.
`QuantSwitch.enabled = False` runs the original bfloat16 linears (for paired full-precision inference).
"""

import math
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET = re.compile(  # the video expert's blocks are also reachable as mixtures.video.{double,single}_blocks
    r"^mixtures\.(video\.transformer|video|action)\."
    r"(double_blocks\.\d+\.(img_attn\.qkv|img_attn\.proj|img_mlp\.[02]|txt_attn\.qkv|txt_attn\.proj|txt_mlp\.[02])"
    r"|single_blocks\.\d+\.linear[12])$")
BLOCK = 1024
CLIP_RATIOS = torch.linspace(1.0, 0.3, 15)


@dataclass(frozen=True)
class QuantConfig:
    name: str
    w_bits: int  # 16: weights not quantized
    a_bits: int  # 16: activations not quantized
    rotate: bool
    smooth_alpha: float | None = None  # SmoothQuant migration strength; None: off
    a_clip: float = 1.0


CONFIGS = {c.name: c for c in (
    QuantConfig("fp_rot", 16, 16, rotate=True),  # rotation only: must equal bfloat16 up to rounding
    QuantConfig("rtn_w8a8", 8, 8, rotate=False),
    QuantConfig("rot_w8a8", 8, 8, rotate=True),
    QuantConfig("rot_w4a16", 4, 16, rotate=True),
    QuantConfig("rot_w16a4", 16, 4, rotate=True),
    QuantConfig("rtn_w4a4", 4, 4, rotate=False),
    QuantConfig("sq_w4a4", 4, 4, rotate=False, smooth_alpha=0.5),
    QuantConfig("rot_w4a4", 4, 4, rotate=True),
    QuantConfig("sqrot_w4a4", 4, 4, rotate=True, smooth_alpha=0.5),
    QuantConfig("rot_w3a3", 3, 3, rotate=True),
    QuantConfig("rot_w2a16", 2, 16, rotate=True),
    QuantConfig("rot_w16a2", 16, 2, rotate=True),
    QuantConfig("rtn_w2a2", 2, 2, rotate=False),
    QuantConfig("rot_w2a2", 2, 2, rotate=True),
    QuantConfig("sqrot_w2a2", 2, 2, rotate=True, smooth_alpha=0.5),
)}


@dataclass
class QuantSwitch:
    enabled: bool = True


def hadamard_block(seed: int = 0) -> torch.Tensor:
    """Randomized Hadamard H D / sqrt(BLOCK), orthogonal, float32 [BLOCK, BLOCK]."""
    h = torch.ones(1, 1)
    while h.shape[0] < BLOCK:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    signs = torch.randint(0, 2, (BLOCK,), generator=torch.Generator().manual_seed(seed)).float() * 2 - 1
    return h * signs[None, :] / math.sqrt(BLOCK)


def rotate_last(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """x [..., n * BLOCK] @ diag(rotation, …, rotation)."""
    shape = x.shape
    return (x.reshape(*shape[:-1], shape[-1] // BLOCK, BLOCK) @ rotation).reshape(shape)


def quantize_activation(x: torch.Tensor, bits: int, clip: float) -> torch.Tensor:
    if bits >= 16:
        return x
    qmax = 2 ** (bits - 1) - 1
    scale = (x.abs().amax(dim=-1, keepdim=True) * clip / qmax).clamp_min(1e-8)
    return (x / scale).round().clamp(-qmax, qmax) * scale


def quantize_weight(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per output channel (row), clipping ratio chosen per row for minimum squared error."""
    if bits >= 16:
        return w
    qmax = 2 ** (bits - 1) - 1
    absmax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    best, best_err = None, None
    for ratio in CLIP_RATIOS.tolist():
        scale = absmax * ratio / qmax
        q = (w / scale).round().clamp(-qmax, qmax) * scale
        err = (q - w).pow(2).sum(dim=1, keepdim=True)
        if best is None:
            best, best_err = q, err
        else:
            better = err < best_err
            best = torch.where(better, q, best)
            best_err = torch.where(better, err, best_err)
    return best


class FakeQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, config: QuantConfig, switch: QuantSwitch, rotation: torch.Tensor | None,
                 smooth: torch.Tensor | None) -> None:
        super().__init__()
        self.linear = linear
        self.config = config
        self.switch = switch
        w = linear.weight.detach().float()
        if smooth is not None:
            w = w * smooth[None, :]
        if rotation is not None:
            w = rotate_last(w, rotation)
        self.register_buffer("qweight", quantize_weight(w, config.w_bits).to(linear.weight.dtype), persistent=False)
        self.register_buffer("rotation", rotation, persistent=False)
        self.register_buffer("smooth", smooth, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.switch.enabled:
            return self.linear(x)
        h = x.float()
        if self.smooth is not None:
            h = h / self.smooth
        if self.rotation is not None:
            h = rotate_last(h, self.rotation)
        h = quantize_activation(h, self.config.a_bits, self.config.a_clip)
        return F.linear(h.to(x.dtype), self.qweight, self.linear.bias)


@dataclass
class QuantizedModel:
    """The wrapped linears of one model; `remove` restores the original modules."""

    switch: QuantSwitch
    replaced: dict[str, tuple[nn.Module, str, nn.Linear]] = field(default_factory=dict)  # name -> (parent, attr, orig)

    def remove(self) -> None:
        for parent, attr, original in self.replaced.values():
            setattr(parent, attr, original)
        self.replaced.clear()


def target_linears(mot: nn.Module) -> dict[str, nn.Linear]:
    modules = dict(mot.named_modules())
    return {name: module for name, module in modules.items() if TARGET.match(name) and isinstance(module, nn.Linear)}


def quantize_model(mot: nn.Module, config: QuantConfig, calibration_absmax: dict[str, torch.Tensor] | None = None,
                   switch: QuantSwitch | None = None) -> QuantizedModel:
    """Wraps every target linear of the MoT in place. SmoothQuant needs `calibration_absmax` (input channel absmax per
    linear name, from `collect_input_absmax`)."""
    switch = switch or QuantSwitch()
    handle = QuantizedModel(switch=switch)
    modules = dict(mot.named_modules())
    targets = target_linears(mot)
    rotation = None
    for name, linear in targets.items():
        if config.rotate and rotation is None:
            rotation = hadamard_block().to(linear.weight.device)
        smooth = None
        if config.smooth_alpha is not None:
            x_max = calibration_absmax[name].to(linear.weight.device).float().clamp_min(1e-5)
            w_max = linear.weight.detach().float().abs().amax(dim=0).clamp_min(1e-5)
            smooth = (x_max.pow(config.smooth_alpha) / w_max.pow(1 - config.smooth_alpha)).clamp_min(1e-5)
        parent_name, attr = name.rsplit(".", 1)
        parent = modules[parent_name]
        setattr(parent, attr, FakeQuantLinear(linear, config, switch, rotation, smooth))
        handle.replaced[name] = (parent, attr, linear)
    return handle


class InputAbsmaxCollector:
    """Forward hooks recording the per-input-channel absmax of every target linear (SmoothQuant calibration)."""

    def __init__(self, mot: nn.Module) -> None:
        self.absmax: dict[str, torch.Tensor] = {}
        self.hooks = [linear.register_forward_hook(self._hook(name)) for name, linear in target_linears(mot).items()]

    def _hook(self, name: str):
        def record(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            x = inputs[0].detach().float().abs().reshape(-1, inputs[0].shape[-1]).amax(dim=0)
            self.absmax[name] = torch.maximum(self.absmax[name], x) if name in self.absmax else x
        return record

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
