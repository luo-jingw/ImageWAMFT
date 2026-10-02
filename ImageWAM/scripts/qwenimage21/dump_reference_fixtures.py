"""Dump Qwen-Image-2.1 reference fixtures with diffusers (run in `.venv-qwenimage`, diffusers @ 578c9b2).

Inputs (from `export_fixture_inputs.py`): current.png, target.png, prompts.json in `--inputs-dir`.
Outputs, in `--out-dir`:
  text_embeds.pt   text-to-image Qwen3-VL features (last decoder layer before the final norm, system tokens
                   dropped) for every prompt, plus their token counts
  vae.pt           RGBA inputs, raw posterior modes, normalized latents and reconstructions of both frames (fp32)
  dit.pt           one transformer forward in the ImageWAM layout [text | reference image | target image]:
                   inputs, per-block outputs, intermediate embeddings and the final output (fp32), and the
                   final output of the same forward in bf16
  edit_sample.png  zero-shot edit of the current frame with prompt 0 at 288x256 (sanity check)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline, QwenImage21Transformer2DModel

TIMESTEP = 0.6
SEED = 0


def load_rgba(path: Path) -> torch.Tensor:
    """RGB image -> [1, 4, 1, H, W] in [-1, 1] with an opaque alpha channel."""
    rgb = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1.0
    alpha = torch.ones_like(rgb[:1])
    return torch.cat([rgb, alpha], dim=0)[None, :, None]


def latent_stats(vae: AutoencoderKLQwenImage21) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    return mean, std


def dump_text(pipe: QwenImage21Pipeline, prompts: list[str], device: torch.device) -> dict:
    embeds = []
    for prompt in prompts:
        prompt_embeds, prompt_mask, image_pad_mask = pipe._get_qwen_prompt_embeds([prompt], None, device)
        assert not bool(image_pad_mask.any()), "text-to-image template must contain no image slots"
        valid = int(prompt_mask.sum()) if prompt_mask is not None else prompt_embeds.shape[1]
        embeds.append(prompt_embeds[0, :valid].float().cpu())
    return {"prompts": prompts, "embeds": embeds, "lengths": [int(e.shape[0]) for e in embeds]}


@torch.no_grad()
def dump_vae(model_dir: Path, inputs_dir: Path, device: torch.device) -> dict:
    vae = AutoencoderKLQwenImage21.from_pretrained(model_dir / "vae", torch_dtype=torch.float32).to(device).eval()
    mean, std = latent_stats(vae)
    out: dict = {}
    for name in ("current", "target"):
        rgba = load_rgba(inputs_dir / f"{name}.png").to(device)
        mode = vae.encode(rgba).latent_dist.mode()
        normalized = (mode - mean.to(device)) / std.to(device)
        recon = vae.decode(normalized * std.to(device) + mean.to(device), return_dict=False)[0]
        out[name] = {"rgba": rgba.cpu(), "mode": mode.cpu(), "normalized": normalized[:, :, 0].cpu(), "recon": recon.cpu()}
    del vae
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def dump_dit(model_dir: Path, text: torch.Tensor, ref: torch.Tensor, target: torch.Tensor, device: torch.device) -> dict:
    """Forward in the ImageWAM order: text tokens, then the reference block, then the target block."""
    transformer = QwenImage21Transformer2DModel.from_pretrained(model_dir / "transformer", torch_dtype=torch.float32)
    transformer = transformer.to(device).eval()
    _, channels, height, width = ref.shape
    pack = lambda x: x.reshape(1, channels, height * width).transpose(1, 2)  # noqa: E731  (pipeline `_pack_latents`)
    generator = torch.Generator().manual_seed(SEED)
    noise = torch.randn(target.shape, generator=generator)
    noisy_target = (1.0 - TIMESTEP) * target + TIMESTEP * noise
    hidden_states = torch.cat([pack(ref), pack(noisy_target)], dim=1).to(device)
    text_len = text.shape[0]
    slots = height * width // 4
    encoder_hidden_states = torch.cat([text, text.new_zeros(slots, text.shape[1])], dim=0)[None].to(device)
    img_mask = torch.tensor([False] * text_len + [True] * slots + [True] * slots, device=device)[None]
    img_shapes = [[(1, height, width), (1, height, width)]]
    timestep = torch.tensor([TIMESTEP], device=device)

    captured: dict[str, torch.Tensor] = {}
    hooks = [
        transformer.txt_in.register_forward_hook(lambda m, a, o: captured.__setitem__("txt_in", o.float().cpu())),
        transformer.img_in.register_forward_hook(lambda m, a, o: captured.__setitem__("img_in", o.float().cpu())),
        transformer.time_text_embed.register_forward_hook(lambda m, a, o: captured.__setitem__("temb", o.float().cpu())),
        transformer.modulation.register_forward_hook(lambda m, a, o: captured.__setitem__("modulation", o.float().cpu())),
    ]
    blocks: list[torch.Tensor] = []
    hooks += [b.register_forward_hook(lambda m, a, o: blocks.append(o.float().cpu())) for b in transformer.transformer_blocks]
    kwargs = dict(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_shapes=img_shapes,
        img_mask=img_mask,
        return_dict=False,
    )
    output_fp32 = transformer(**kwargs)[0].float().cpu()
    for h in hooks:
        h.remove()
    transformer = transformer.to(torch.bfloat16)
    kwargs["hidden_states"] = hidden_states.to(torch.bfloat16)
    kwargs["encoder_hidden_states"] = encoder_hidden_states.to(torch.bfloat16)
    output_bf16 = transformer(**kwargs)[0].float().cpu()
    del transformer
    torch.cuda.empty_cache()
    return {
        "inputs": {
            "text": text.cpu(),
            "ref_latents": ref.cpu(),
            "target_latents": target.cpu(),
            "noise": noise,
            "noisy_target_latents": noisy_target,
            "timestep": TIMESTEP,
            "text_len": text_len,
            "grid": (height, width),
        },
        "layout": "imagewam_order_native_mask: [text(causal) | ref block | target block]",
        "intermediate": captured,
        "blocks": blocks,
        "output_fp32": output_fp32,
        "output_bf16": output_bf16,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads((args.inputs_dir / "prompts.json").read_text())["prompts"]

    pipe = QwenImage21Pipeline.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16).to(device)
    text = dump_text(pipe, prompts, device)
    torch.save(text, args.out_dir / "text_embeds.pt")
    print("text token counts:", text["lengths"])
    current = Image.open(args.inputs_dir / "current.png").convert("RGB")
    edit = pipe(
        prompt=prompts[0],
        image=current,
        height=current.size[1],
        width=current.size[0],
        output_resolution=272,
        num_inference_steps=40,
        generator=torch.Generator(device).manual_seed(SEED),
    ).images[0]
    edit.save(args.out_dir / "edit_sample.png")
    del pipe
    torch.cuda.empty_cache()

    vae = dump_vae(args.model_dir, args.inputs_dir, device)
    torch.save(vae, args.out_dir / "vae.pt")
    print("vae normalized latent shape:", tuple(vae["current"]["normalized"].shape))

    dit = dump_dit(args.model_dir, text["embeds"][0], vae["current"]["normalized"], vae["target"]["normalized"], device)
    torch.save(dit, args.out_dir / "dit.pt")
    diff = (dit["output_fp32"] - dit["output_bf16"]).abs().max().item()
    print(f"dit: {len(dit['blocks'])} blocks, output {tuple(dit['output_fp32'].shape)}, bf16-vs-fp32 max abs diff {diff:.4g}")


if __name__ == "__main__":
    main()
