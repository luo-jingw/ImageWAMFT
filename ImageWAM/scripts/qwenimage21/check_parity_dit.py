"""Parity of `QwenImage21VideoExpert` against the diffusers reference fixtures (`dump_reference_fixtures.py`).

Runs the fixture forward ([text | reference | target], native Qwen-Image-2.1 mask: token-causal text,
bidirectional image blocks) block by block and prints max absolute / relative differences for the
intermediate embeddings, every block output and the target prediction, in fp32 and bf16.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from imagewam.models.backbones.qwenimage21_video_expert import QwenImage21PreState, QwenImage21VideoExpert


def diff(name: str, ours: torch.Tensor, ref: torch.Tensor) -> float:
    ours = ours.float().cpu()
    ref = ref.float().cpu()
    abs_diff = (ours - ref).abs().max().item()
    rel = abs_diff / max(ref.abs().max().item(), 1e-12)
    print(f"  {name:28s} shape={tuple(ref.shape)} max_abs={abs_diff:.4e} rel={rel:.3e}")
    return rel


def native_mask(text_len: int, block_lens: list[int], device: torch.device) -> torch.Tensor:
    total = text_len + sum(block_lens)
    block_id = torch.full((total,), -1, dtype=torch.long, device=device)
    start = text_len
    for index, length in enumerate(block_lens):
        block_id[start : start + length] = index
        start += length
    q = torch.arange(total, device=device)[:, None]
    k = torch.arange(total, device=device)[None, :]
    same_block = (block_id[:, None] == block_id[None, :]) & (block_id[:, None] >= 0)
    return ((q >= k) | same_block)[None, None]


def run_blocks(
    expert: QwenImage21VideoExpert, pre: QwenImage21PreState, mask: torch.Tensor, ref_blocks: list[torch.Tensor] | None
) -> torch.Tensor:
    x = pre.tokens
    heads, head_dim = expert.num_heads, expert.attn_head_dim
    for index, block in enumerate(expert.blocks):
        built = block.prepare_qkv(x, pre.rotary, pre.modulation)
        b, length, _ = built.q.shape
        q, k, v = (t.view(b, length, heads, head_dim).transpose(1, 2) for t in (built.q, built.k, built.v))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(b, length, heads * head_dim)
        x = block.apply_post(built.residual_x, attn, built.post_state)
        if ref_blocks is not None and (index < 2 or index % 8 == 7 or index == len(expert.blocks) - 1):
            diff(f"block {index:2d}", x, ref_blocks[index])
    return x


@torch.no_grad()
def forward(expert: QwenImage21VideoExpert, fixture: dict, device: torch.device, check_blocks: bool) -> torch.Tensor:
    inputs = fixture["inputs"]
    dtype = expert.img_in.weight.dtype
    height, width = inputs["grid"]
    text = inputs["text"][None].to(device=device, dtype=dtype)
    ref = expert.pack_latents(inputs["ref_latents"]).to(device=device, dtype=dtype)
    target = expert.pack_latents(inputs["noisy_target_latents"]).to(device=device, dtype=dtype)
    text_len = int(inputs["text_len"])
    positions = expert.build_positions(text_len, (height, width), (height, width), 0, device)
    pre = expert.pre_dit(
        target_latents=target,
        video_timestep=torch.tensor([inputs["timestep"]], device=device),
        text_states=text,
        text_mask=torch.ones(1, text_len, dtype=torch.bool, device=device),
        ref_latents=ref,
        positions=positions,
    )
    if check_blocks:
        inter = fixture["intermediate"]
        diff("txt_in", pre.tokens[:, :text_len], inter["txt_in"][:, :text_len])  # fixture also holds the 72 slot rows
        diff("img_in", pre.tokens[:, text_len:], inter["img_in"])
        diff("temb (target row)", pre.temb_target, inter["temb"][:1])
        print(f"  modulation rows reference: {tuple(inter['modulation'].shape)} (row 0: t, row 1: t=0)")
    mask = native_mask(text_len, [height * width, height * width], device)
    x = run_blocks(expert, pre, mask, fixture["blocks"] if check_blocks else None)
    return expert.post_dit(x, pre)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True, help="dit.pt from dump_reference_fixtures.py")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    fixture = torch.load(args.fixtures, map_location="cpu", weights_only=False)
    text_len = int(fixture["inputs"]["text_len"])
    target_slice = slice(text_len + fixture["inputs"]["grid"][0] * fixture["inputs"]["grid"][1], None)

    expert = QwenImage21VideoExpert.from_pretrained(args.transformer_dir, device=device, torch_dtype=torch.float32)
    print(f"parameters: {sum(p.numel() for p in expert.parameters()):,}")
    print("fp32 vs diffusers fp32:")
    out32 = forward(expert, fixture, device, check_blocks=True)
    diff("target prediction (fp32)", out32, fixture["output_fp32"][:, target_slice])

    expert = expert.to(torch.bfloat16)
    print("bf16:")
    out16 = forward(expert, fixture, device, check_blocks=False)
    diff("ours bf16 vs diffusers bf16", out16, fixture["output_bf16"][:, target_slice])
    diff("ours bf16 vs diffusers fp32", out16, fixture["output_fp32"][:, target_slice])
    diff("diffusers bf16 vs fp32", fixture["output_bf16"][:, target_slice], fixture["output_fp32"][:, target_slice])


if __name__ == "__main__":
    main()
