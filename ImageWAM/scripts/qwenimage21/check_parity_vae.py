"""Compare the vendored Qwen-Image-2.1 VAE against the diffusers fp32 reference fixtures (run in the main `.venv`).

The fixture file (`vae.pt` from `dump_reference_fixtures.py`) holds, for the frames "current" and "target":
  rgba        [1, 4, 1, H, W] fp32 in [-1, 1], opaque alpha
  mode        [1, 64, 1, h, w] raw posterior mode
  normalized  [1, 64, h, w] latents normalized with latents_mean / latents_std
  recon       [1, 4, 1, H, W] diffusers decode of the normalized latents

Prints, per frame, the max abs and max relative diff of `encode` vs `normalized` and of `decode(normalized)` vs
`recon`, the parameter counts, and the round-trip PSNR of decode(encode(x)) vs x with images mapped to [0, 1].
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from imagewam.models.backbones.qwenimage21_vae import QwenImage21VAE

FRAMES = ("current", "target")


@dataclass
class DiffStats:
    max_abs: float
    max_rel: float


@dataclass
class FrameFixture:
    rgb: torch.Tensor
    normalized: torch.Tensor
    recon_rgb: torch.Tensor


def diff_stats(output: torch.Tensor, reference: torch.Tensor) -> DiffStats:
    """Max abs diff, and max abs diff relative to the reference's max magnitude."""
    max_abs = (output.float() - reference.float()).abs().max().item()
    return DiffStats(max_abs=max_abs, max_rel=max_abs / reference.float().abs().max().item())


def psnr_db(output: torch.Tensor, reference: torch.Tensor) -> float:
    """PSNR in dB of two images in [-1, 1] after mapping both to [0, 1]."""
    mse = (((output.float() + 1.0) / 2.0 - (reference.float() + 1.0) / 2.0) ** 2).mean().item()
    return 10.0 * math.log10(1.0 / mse)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def load_frame(fixtures: dict[str, dict[str, torch.Tensor]], name: str, device: torch.device) -> FrameFixture:
    entry = fixtures[name]
    return FrameFixture(
        rgb=entry["rgba"][:, :3, 0].to(device),
        normalized=entry["normalized"].to(device),
        recon_rgb=entry["recon"][:, :3, 0].to(device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    vae = QwenImage21VAE.from_pretrained(args.vae_dir, device=device, torch_dtype=torch.float32)
    print(f"loaded {len(vae.state_dict())} tensors with strict key matching, dtype {vae.dtype}, device {device}")
    total = count_parameters(vae)
    encoder = count_parameters(vae.encoder)
    decoder = count_parameters(vae.decoder)
    print(f"parameters: total {total} ({total / 1e6:.2f}M), encoder {encoder} ({encoder / 1e6:.2f}M), "
          f"decoder {decoder} ({decoder / 1e6:.2f}M)")

    fixtures: dict[str, dict[str, torch.Tensor]] = torch.load(args.fixtures, map_location="cpu")
    with torch.inference_mode():
        for name in FRAMES:
            frame = load_frame(fixtures, name, device)
            latents = vae.encode(frame.rgb)
            recon = vae.decode(frame.normalized)
            roundtrip = vae.decode(latents)
            latent_diff = diff_stats(latents, frame.normalized)
            recon_diff = diff_stats(recon, frame.recon_rgb)
            print(f"[{name}] input {tuple(frame.rgb.shape)} -> latents {tuple(latents.shape)} {latents.dtype} "
                  f"-> recon {tuple(recon.shape)} {recon.dtype}")
            print(f"[{name}] latents vs diffusers: max abs {latent_diff.max_abs:.3e}, max rel {latent_diff.max_rel:.3e}")
            print(f"[{name}] recon   vs diffusers: max abs {recon_diff.max_abs:.3e}, max rel {recon_diff.max_rel:.3e}")
            print(f"[{name}] round-trip PSNR decode(encode(x)) vs x: {psnr_db(roundtrip, frame.rgb):.2f} dB "
                  f"(diffusers recon vs x: {psnr_db(frame.recon_rgb, frame.rgb):.2f} dB)")


if __name__ == "__main__":
    main()
