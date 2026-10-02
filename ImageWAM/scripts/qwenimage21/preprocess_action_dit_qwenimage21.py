#!/usr/bin/env python3
"""Initialize the Qwen-Image-2.1 action DiT from the Qwen-Image-2.1 video transformer (run in `.venv`).

Same policy as `scripts/flux2/preprocess_action_dit_flux2.py`: every action-expert parameter whose name exists in the
video transformer (`time_text_embed.*`, `modulation.*`, `transformer_blocks.<i>.*`) is copied from it; mismatched axes
are linearly interpolated, and weights whose input axis shrinks are multiplied by sqrt(src_in / dst_in) (alpha
scaling). `action_encoder.*` and `head.*` keep their fresh initialization.

Writes {"state_dict", "meta"} to --output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from imagewam.models.backbones.action_dit_qwenimage21 import ActionDiTQwenImage21, ActionDiTQwenImage21Config
from imagewam.models.backbones.qwenimage21_video_expert import QwenImage21Config

FRESH_PREFIXES = ("action_encoder.", "head.")


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if src.ndim != len(target_shape):
        raise ValueError(f"Rank mismatch: src={tuple(src.shape)} target={target_shape}")
    out = src.to(torch.float32)
    for dim, new_size in enumerate(target_shape):
        if out.shape[dim] == new_size:
            continue
        moved = out.movedim(dim, -1).contiguous()
        out = _interpolate_last_dim(moved, new_size).movedim(-1, dim).contiguous()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-dir", type=Path, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--max-action-horizon", type=int, default=64)
    parser.add_argument("--no-alpha-scaling", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    apply_alpha_scaling = not args.no_alpha_scaling

    video_cfg = QwenImage21Config.from_json(args.transformer_dir / "config.json")
    action_cfg = ActionDiTQwenImage21Config(
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        num_heads=video_cfg.num_attention_heads,
        attn_head_dim=video_cfg.attention_head_dim,
        num_layers=video_cfg.num_layers,
        mlp_ratio=video_cfg.mlp_ratio,
        eps=video_cfg.eps,
        max_action_horizon=args.max_action_horizon,
    )
    torch.manual_seed(0)
    action_expert = ActionDiTQwenImage21(action_cfg)
    dst_state = action_expert.state_dict()

    src_state: dict[str, torch.Tensor] = {}
    for shard in sorted(args.transformer_dir.glob("*.safetensors")):
        src_state.update(load_file(str(shard), device="cpu"))

    out_state: dict[str, torch.Tensor] = {}
    copied, resized, fresh = [], [], []
    for key, dst in dst_state.items():
        if key.startswith(FRESH_PREFIXES):
            fresh.append(key)
            continue
        if key not in src_state:
            raise KeyError(f"Action-expert parameter {key} has no source in the video transformer")
        src = src_state[key]
        if tuple(src.shape) == tuple(dst.shape):
            value = src.to(torch.float32)
            copied.append(key)
        else:
            value = _resize_tensor_to_shape(src, tuple(dst.shape))
            if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != dst.shape[-1]:
                value = value * (float(src.shape[-1]) / float(dst.shape[-1])) ** 0.5
            resized.append(f"{key}: {tuple(src.shape)} -> {tuple(dst.shape)}")
        out_state[key] = value.to(torch.bfloat16).contiguous()

    num_params = sum(p.numel() for p in dst_state.values())
    payload = {
        "state_dict": out_state,
        "meta": {
            "source": str(args.transformer_dir),
            "config": action_cfg.__dict__,
            "copied": len(copied),
            "resized": len(resized),
            "fresh": fresh,
            "num_parameters": num_params,
            "alpha_scaling": apply_alpha_scaling,
            "policy": "copy Qwen-Image-2.1 time embedding, shared modulation and blocks; interpolate mismatched axes "
            "with alpha scaling; fresh action_encoder and head",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"parameters: {num_params:,}")
    print(f"copied ({len(copied)}): e.g. {copied[:3]}")
    print(f"resized ({len(resized)}), distinct shapes:")
    for line in sorted({entry.split(': ', 1)[0].split('.', 2)[-1] if entry.startswith('transformer_blocks') else entry.split(': ')[0] for entry in resized}):
        example = next(entry for entry in resized if entry.split(": ")[0].endswith(line))
        print(f"  {line}  [{example.split(': ')[1]}]")
    print(f"fresh ({len(fresh)}): {fresh}")
    print(f"[ok] saved {args.output}")


if __name__ == "__main__":
    main()
