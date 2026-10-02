"""Qwen-Image-2.1 transformer as an ImageWAM video expert (MoT block protocol "qwenimage21").

Re-implements `QwenImage21Transformer2DModel` (diffusers @ 578c9b2c6636ab2424a0e56186268b83623656b2,
`models/transformers/transformer_qwenimage21.py`) with parameter names identical to the released
checkpoint, split into the per-block `prepare_qkv` / `apply_post` steps that the MoT joint attention needs.

Conventions (plan.md D2, D4, D5):
  - token order [text (incl. appended proprio) | reference image | target image]; the attention mask is
    built by the caller;
  - context tokens (text, reference) are modulated with timestep 0, target tokens with the sampled timestep
    (Qwen-Image-2.1 `causal_condition`);
  - rotary positions follow the Qwen-Image-2.1 counter: text tokens advance one position on all three axes,
    each image block sits at the current position on the frame axis with a zero-centred height/width grid and
    advances the counter by max(h, w); action tokens continue the counter after the target block.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

ROPE_THETA = 10000
TIMESTEP_DIM = 256


@dataclass(frozen=True)
class QwenImage21Config:
    num_layers: int
    num_attention_heads: int
    attention_head_dim: int
    in_channels: int
    out_channels: int
    context_in_dim: int
    mlp_ratio: int
    axes_dims_rope: tuple[int, int, int]
    eps: float

    @property
    def hidden_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @classmethod
    def from_json(cls, path: Path) -> "QwenImage21Config":
        raw = json.loads(Path(path).read_text())
        if int(raw.get("patch_size", 1)) != 1:
            raise ValueError(f"Only patch_size=1 is supported, got {raw.get('patch_size')}")
        return cls(
            num_layers=int(raw["num_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            attention_head_dim=int(raw["attention_head_dim"]),
            in_channels=int(raw["in_channels"]),
            out_channels=int(raw.get("out_channels") or raw["in_channels"]),
            context_in_dim=int(raw["context_in_dim"]),
            mlp_ratio=int(raw["mlp_ratio"]),
            axes_dims_rope=tuple(int(x) for x in raw["axes_dims_rope"]),
            eps=float(raw.get("eps", 1e-6)),
        )


@dataclass
class QwenImage21Positions:
    """Complex rotary tables [L, head_dim / 2] for each token segment."""

    text: torch.Tensor
    ref: torch.Tensor
    target: torch.Tensor
    action: torch.Tensor

    def video(self, include_target: bool) -> torch.Tensor:
        parts = [self.text, self.ref] + ([self.target] if include_target else [])
        return torch.cat(parts, dim=0)


@dataclass
class QwenImage21Modulation:
    """Per-token AdaLN parameters, each broadcastable to [B, L, hidden]."""

    scale_msa: torch.Tensor
    gate_msa: torch.Tensor
    scale_mlp: torch.Tensor
    gate_mlp: torch.Tensor


@dataclass
class QwenImage21QKV:
    q: torch.Tensor  # [B, L, heads * head_dim], after QK-norm and rotary
    k: torch.Tensor
    v: torch.Tensor
    residual_x: torch.Tensor
    post_state: dict[str, torch.Tensor]  # keys gate_msa / scale_mlp / gate_mlp (MoT generic post protocol)


@dataclass
class QwenImage21PreState:
    tokens: torch.Tensor  # [B, L, hidden]
    rotary: torch.Tensor  # [L, head_dim / 2] complex
    modulation: QwenImage21Modulation
    temb_target: torch.Tensor  # [B, hidden]
    txt_len: int
    ref_len: int
    target_len: int
    text_mask: torch.Tensor  # [B, txt_len] bool


class QwenImage21Rope:
    """Rotary tables computed directly from integer positions (same values as the diffusers lookup table)."""

    def __init__(self, axes_dim: tuple[int, int, int], theta: int = ROPE_THETA) -> None:
        self.axes_dim = axes_dim
        # Explicit CPU device: the expert may be constructed under a `torch.device("meta")` context.
        self.inv_freqs = [
            1.0 / torch.pow(theta, torch.arange(0, d, 2, device="cpu").to(torch.float32).div(d)) for d in axes_dim
        ]

    def _axis(self, positions: torch.Tensor, axis: int) -> torch.Tensor:
        freqs = torch.outer(positions.to(torch.float32), self.inv_freqs[axis].to(positions.device))
        return torch.polar(torch.ones_like(freqs), freqs)

    def table(self, frame: torch.Tensor, height: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._axis(frame, 0), self._axis(height, 1), self._axis(width, 2)], dim=-1)

    def sequence(self, start: int, length: int, device: torch.device) -> tuple[torch.Tensor, int]:
        index = torch.arange(start, start + length, device=device)
        return self.table(index, index, index), start + length

    def image_block(self, start: int, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, int]:
        h_index = torch.arange(-(height - height // 2), height // 2, device=device)
        w_index = torch.arange(-(width - width // 2), width // 2, device=device)
        frame = torch.full((height * width,), start, device=device)
        hh = h_index.repeat_interleave(width)
        ww = w_index.repeat(height)
        return self.table(frame, hh, ww), start + max(height, width)


def apply_rotary(x: torch.Tensor, rotary: torch.Tensor) -> torch.Tensor:
    """x [B, L, H, D] real, rotary [L, D/2] complex -> rotated x (diffusers `use_real=False` path)."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_complex * rotary.unsqueeze(1)).flatten(3).type_as(x)


class QwenImage21RMSNorm(nn.Module):
    """diffusers `RMSNorm` (affine, fp32 statistics, result in the weight dtype when low precision)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight


class QwenImage21ZeroCenterRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x * rrms * (self.weight.float() + 1)).to(input_dtype)


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden_dim: int, eps: float) -> None:
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_dim, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.act(self.in_layer(self.text_norm(hidden_states))))


class QwenImage21TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class QwenImage21TimeEmbed(nn.Module):
    """`time_text_embed`: cos-first sinusoid of 1000 * t (t in [0, 1]) followed by an MLP."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = QwenImage21TimestepEmbedding(TIMESTEP_DIM, hidden_dim)
        half = TIMESTEP_DIM // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # diffusers casts t to the model dtype before scaling; keep that for parity.
        t = 1000.0 * timestep.to(dtype).float()
        args = t[:, None] * self.freqs[None].to(device=t.device, dtype=torch.float32)
        proj = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.timestep_embedder(proj.to(dtype))


class QwenImage21SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.out = nn.Linear(mlp_dim, hidden_dim, bias=False)
        self.gate_layer = nn.Linear(hidden_dim, mlp_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.gate_layer(hidden_states)) * self.proj(hidden_states))


class QwenImage21Attention(nn.Module):
    """Projection layout of diffusers `QwenImage21Attention` (to_q/to_k/to_v/to_out.0, per-head RMS QK-norm)."""

    def __init__(self, dim: int, heads: int, head_dim: int, eps: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim, bias=False), nn.Dropout(0.0)])
        self.norm_q = QwenImage21RMSNorm(head_dim, eps=eps)
        self.norm_k = QwenImage21RMSNorm(head_dim, eps=eps)


class QwenImage21Block(nn.Module):
    """Single-stream block with shared (non-learned per block) modulation, split for MoT joint attention."""

    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: int, eps: float) -> None:
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenImage21Attention(dim, heads, head_dim, eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenImage21SwiGLU(dim, dim * mlp_ratio)

    def prepare_qkv(self, x: torch.Tensor, rotary: torch.Tensor, modulation: QwenImage21Modulation) -> QwenImage21QKV:
        h = self.img_norm1(x) * (1 + modulation.scale_msa)
        q = self.attn.to_q(h).unflatten(-1, (self.attn.heads, -1))
        k = self.attn.to_k(h).unflatten(-1, (self.attn.heads, -1))
        v = self.attn.to_v(h).unflatten(-1, (self.attn.heads, -1))
        q = self.attn.norm_q(q).to(v.dtype)
        k = self.attn.norm_k(k).to(v.dtype)
        q = apply_rotary(q, rotary)
        k = apply_rotary(k, rotary)
        return QwenImage21QKV(
            q=q.flatten(2),
            k=k.flatten(2),
            v=v.flatten(2),
            residual_x=x,
            post_state={
                "gate_msa": modulation.gate_msa,
                "scale_mlp": modulation.scale_mlp,
                "gate_mlp": modulation.gate_mlp,
            },
        )

    def apply_post(self, residual_x: torch.Tensor, attn_out: torch.Tensor, post_state: dict[str, torch.Tensor]) -> torch.Tensor:
        attn = self.attn.to_out[1](self.attn.to_out[0](attn_out))
        x = residual_x + post_state["gate_msa"].tanh() * attn
        h = self.img_norm2(x) * (1 + post_state["scale_mlp"])
        return x + post_state["gate_mlp"].tanh() * self.img_mlp(h)


class QwenImage21AdaNormOut(nn.Module):
    """`norm_out`: scale-only AdaLN."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False, bias=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale = self.linear(self.silu(temb).to(x.dtype))
        return self.norm(x) * (1 + scale.unsqueeze(1))


class QwenImage21VideoExpert(nn.Module):
    block_protocol = "qwenimage21"

    def __init__(self, config: QwenImage21Config) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_attention_heads
        self.attn_head_dim = config.attention_head_dim
        self.hidden_dim = config.hidden_dim
        self.use_gradient_checkpointing = False
        dim = config.hidden_dim
        self.time_text_embed = QwenImage21TimeEmbed(dim)
        self.txt_in = QwenImage21TextProjection(config.context_in_dim, dim, eps=config.eps)
        self.img_in = nn.Linear(config.in_channels, dim, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=False))
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21Block(dim, config.num_attention_heads, config.attention_head_dim, config.mlp_ratio, config.eps)
                for _ in range(config.num_layers)
            ]
        )
        self.norm_out = QwenImage21AdaNormOut(dim, eps=config.eps)
        self.proj_out = nn.Linear(dim, config.out_channels, bias=False)
        self.rope = QwenImage21Rope(config.axes_dims_rope)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.transformer_blocks

    @classmethod
    def from_pretrained(cls, transformer_dir: Path, device: torch.device, torch_dtype: torch.dtype) -> "QwenImage21VideoExpert":
        transformer_dir = Path(transformer_dir)
        config = QwenImage21Config.from_json(transformer_dir / "config.json")
        with torch.device("meta"):
            model = cls(config)
        state: dict[str, torch.Tensor] = {}
        for shard in sorted(transformer_dir.glob("*.safetensors")):
            state.update(load_file(str(shard), device="cpu"))
        result = model.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"Qwen-Image-2.1 load mismatch: {result}")
        # The non-persistent sinusoid buffer was created on the meta device; materialize it.
        half = TIMESTEP_DIM // 2
        model.time_text_embed.register_buffer(
            "freqs",
            torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half),
            persistent=False,
        )
        return model.to(device=device, dtype=torch_dtype).eval()

    # ---------------------------------------------------------------- positions
    def build_positions(
        self,
        text_len: int,
        ref_hw: tuple[int, int],
        target_hw: tuple[int, int],
        action_len: int,
        device: torch.device,
    ) -> QwenImage21Positions:
        text, cursor = self.rope.sequence(0, text_len, device)
        ref, cursor = self.rope.image_block(cursor, ref_hw[0], ref_hw[1], device)
        target, cursor = self.rope.image_block(cursor, target_hw[0], target_hw[1], device)
        action, _ = self.rope.sequence(cursor, action_len, device)
        return QwenImage21Positions(text=text, ref=ref, target=target, action=action)

    # ---------------------------------------------------------------- latents
    @staticmethod
    def pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, H*W, C] (row-major, the pipeline `_pack_latents`)."""
        batch, channels, height, width = latents.shape
        return latents.reshape(batch, channels, height * width).transpose(1, 2)

    @staticmethod
    def unpack_latents(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = tokens.shape
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)

    # ---------------------------------------------------------------- forward pieces
    def _modulation_rows(self, timestep: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (temb [B+1, D], modulation [B+1, 4D]); the last row is t = 0."""
        t = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
        temb = self.time_text_embed(t, dtype)
        return temb, self.modulation(temb)

    def pre_dit(
        self,
        target_latents: torch.Tensor | None,
        video_timestep: torch.Tensor,
        text_states: torch.Tensor,
        text_mask: torch.Tensor,
        ref_latents: torch.Tensor,
        positions: QwenImage21Positions,
    ) -> QwenImage21PreState:
        dtype = self.img_in.weight.dtype
        batch = int(ref_latents.shape[0])
        txt = self.txt_in(text_states.to(dtype))
        image_tokens = ref_latents if target_latents is None else torch.cat([ref_latents, target_latents], dim=1)
        img = self.img_in(image_tokens.to(dtype))
        tokens = torch.cat([txt, img], dim=1)
        txt_len = int(txt.shape[1])
        ref_len = int(ref_latents.shape[1])
        target_len = 0 if target_latents is None else int(target_latents.shape[1])
        prefix_len = txt_len + ref_len

        temb, modulation = self._modulation_rows(video_timestep.reshape(batch).to(dtype), dtype)
        (mod_scale_msa, mod_gate_msa, mod_scale_mlp, mod_gate_mlp) = modulation.chunk(4, dim=-1)

        def per_token(row_values: torch.Tensor) -> torch.Tensor:
            zero_row = row_values[-1:].unsqueeze(1).expand(batch, prefix_len, -1)
            if target_len == 0:
                return zero_row
            target_rows = row_values[:-1].unsqueeze(1).expand(batch, target_len, -1)
            return torch.cat([zero_row, target_rows], dim=1)

        token_modulation = QwenImage21Modulation(
            scale_msa=per_token(mod_scale_msa),
            gate_msa=per_token(mod_gate_msa),
            scale_mlp=per_token(mod_scale_mlp),
            gate_mlp=per_token(mod_gate_mlp),
        )
        return QwenImage21PreState(
            tokens=tokens,
            rotary=positions.video(include_target=target_len > 0),
            modulation=token_modulation,
            temb_target=temb[:-1],
            txt_len=txt_len,
            ref_len=ref_len,
            target_len=target_len,
            text_mask=text_mask.to(device=tokens.device, dtype=torch.bool),
        )

    def post_dit(self, tokens: torch.Tensor, pre: QwenImage21PreState) -> torch.Tensor:
        start = pre.txt_len + pre.ref_len
        target = tokens[:, start : start + pre.target_len]
        return self.proj_out(self.norm_out(target, pre.temb_target))
