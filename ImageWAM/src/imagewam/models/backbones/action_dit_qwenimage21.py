"""Action expert for the Qwen-Image-2.1 ImageWAM stack (MoT block protocol "qwenimage21").

Single-stream blocks with the Qwen-Image-2.1 layout (`QwenImage21Block`): residual width `hidden_dim` (1024), attention
width num_heads * attn_head_dim (32 * 128 = 4096, shared with the video expert for the joint attention), SwiGLU MLP,
per-head RMS QK-norm, tanh-gated residuals and one shared modulation for all blocks. Every action token is modulated
with the action timestep. Rotary positions are supplied by the video expert (`QwenImage21Positions.action`).

The blocks, time embedding and modulation are initialized offline from the video transformer by
`scripts/qwenimage21/preprocess_action_dit_qwenimage21.py`; `action_encoder` and `head` start fresh.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from imagewam.utils.logging_config import get_logger

from .qwenimage21_video_expert import (
    QwenImage21AdaNormOut,
    QwenImage21Block,
    QwenImage21Modulation,
    QwenImage21TimeEmbed,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class ActionDiTQwenImage21Config:
    action_dim: int
    hidden_dim: int = 1024
    num_heads: int = 32
    attn_head_dim: int = 128
    num_layers: int = 32
    mlp_ratio: int = 3
    eps: float = 1e-6
    max_action_horizon: int = 64
    use_gradient_checkpointing: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ActionDiTQwenImage21Config":
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - names)
        if unknown:
            raise ValueError(f"Unknown ActionDiTQwenImage21 config keys: {unknown}")
        return cls(**raw)


@dataclass
class ActionPreState:
    tokens: torch.Tensor  # [B, T, hidden]
    modulation: QwenImage21Modulation  # each [B, 1, hidden]
    temb: torch.Tensor  # [B, hidden]


class QwenImage21ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, eps: float) -> None:
        super().__init__()
        self.norm_out = QwenImage21AdaNormOut(hidden_dim, eps=eps)
        self.proj_out = nn.Linear(hidden_dim, action_dim, bias=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.norm_out(x, temb))


class ActionDiTQwenImage21(nn.Module):
    block_protocol = "qwenimage21"

    def __init__(self, config: ActionDiTQwenImage21Config) -> None:
        super().__init__()
        self.config = config
        self.action_dim = int(config.action_dim)
        self.hidden_dim = int(config.hidden_dim)
        self.num_heads = int(config.num_heads)
        self.num_kv_heads = self.num_heads
        self.attn_head_dim = int(config.attn_head_dim)
        self.max_action_horizon = int(config.max_action_horizon)
        self.use_gradient_checkpointing = bool(config.use_gradient_checkpointing)
        dim = self.hidden_dim
        self.action_encoder = nn.Linear(self.action_dim, dim)
        self.time_text_embed = QwenImage21TimeEmbed(dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=False))
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21Block(dim, self.num_heads, self.attn_head_dim, int(config.mlp_ratio), float(config.eps))
                for _ in range(int(config.num_layers))
            ]
        )
        self.head = QwenImage21ActionHead(dim, self.action_dim, float(config.eps))

    @property
    def blocks(self) -> nn.ModuleList:
        return self.transformer_blocks

    @classmethod
    def from_pretrained(
        cls,
        config: ActionDiTQwenImage21Config,
        pretrained_path: str | Path | None,
        device: torch.device | str,
        torch_dtype: torch.dtype,
    ) -> "ActionDiTQwenImage21":
        model = cls(config)
        if pretrained_path:
            payload = torch.load(str(pretrained_path), map_location="cpu")
            state_dict = payload.get("state_dict", payload)
            result = model.load_state_dict(state_dict, strict=False)
            fresh = sorted({key.split(".")[0] for key in result.missing_keys})
            if result.unexpected_keys:
                raise ValueError(f"Unexpected ActionDiTQwenImage21 keys: {result.unexpected_keys[:20]}")
            if any(prefix not in {"action_encoder", "head"} for prefix in fresh):
                raise ValueError(f"ActionDiTQwenImage21 initialization misses copied weights: {result.missing_keys[:20]}")
            logger.info("Loaded ActionDiTQwenImage21 init from %s (fresh modules: %s)", pretrained_path, fresh)
        else:
            logger.info("Initializing ActionDiTQwenImage21 without pretrained weights.")
        return model.to(device=device, dtype=torch_dtype)

    def pre_dit(self, actions: torch.Tensor, timestep: torch.Tensor) -> ActionPreState:
        """actions [B, T, action_dim], timestep [B] (or [1]) in [0, 1]."""
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"`actions` must be [B, T, {self.action_dim}], got {tuple(actions.shape)}")
        batch, horizon, _ = actions.shape
        if horizon > self.max_action_horizon:
            raise ValueError(f"Action length {horizon} exceeds max_action_horizon={self.max_action_horizon}")
        timestep = timestep.reshape(-1)
        if timestep.shape[0] == 1 and batch > 1:
            timestep = timestep.expand(batch)
        if timestep.shape[0] != batch:
            raise ValueError(f"`timestep` must have {batch} entries, got {timestep.shape[0]}")
        dtype = self.action_encoder.weight.dtype
        tokens = self.action_encoder(actions.to(dtype))
        temb = self.time_text_embed(timestep, dtype)
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.modulation(temb).unsqueeze(1).chunk(4, dim=-1)
        return ActionPreState(
            tokens=tokens,
            modulation=QwenImage21Modulation(scale_msa=scale_msa, gate_msa=gate_msa, scale_mlp=scale_mlp, gate_mlp=gate_mlp),
            temb=temb,
        )

    def post_dit(self, tokens: torch.Tensor, pre: ActionPreState) -> torch.Tensor:
        return self.head(tokens, pre.temb)
