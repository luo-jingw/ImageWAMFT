"""Compare the scale of the video flow-matching target in the FLUX.2 and Qwen-Image-2.1 latent spaces (run in `.venv`).

Encodes validation frame pairs (t, t+16) with each frozen VAE exactly as the training stacks do and reports, per latent
element: E[z^2] of the target latent, E[(z_target - z_ref)^2] (how large the 16-frame change is in that space), their
ratio, and E[(noise - z_target)^2], the per-element size of the velocity target v = noise - x0 that `loss_video` is
measured against.

  python scripts/analyze_latent_scales.py --flux-run-dir <4B run> --qwen-run-dir <Qwen run> --pairs 64
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from imagewam.runtime import build_datasets
from imagewam.utils import misc


def encoder_of(run_dir: Path):
    cfg = OmegaConf.load(run_dir / "config.yaml")
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    if model.stack == "flux2":
        return lambda image: model._encode_flux2_image_tokens(image, time_value=0.0)[0]
    stack = model._qwenimage21_stack()
    return lambda image: stack.encode_image(image)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flux-run-dir", type=Path, required=True)
    parser.add_argument("--qwen-run-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    misc.register_work_dir(str(args.work_dir))
    cfg = OmegaConf.load(args.flux_run_dir / "config.yaml")
    cfg.data.train.pretrained_norm_stats = str(args.flux_run_dir / "dataset_stats.json")
    _, val = build_datasets(cfg.data)
    indices = np.linspace(0, len(val) - 1, args.pairs).round().astype(int).tolist()
    frames = [val[i]["video"] for i in indices]  # each [3, 2, H, W]
    current = torch.stack([f[:, 0] for f in frames])
    target = torch.stack([f[:, -1] for f in frames])

    for name, run_dir in (("FLUX.2 AE", args.flux_run_dir), ("Qwen-Image-2.1 VAE", args.qwen_run_dir)):
        encode = encoder_of(run_dir)
        with torch.no_grad():
            z_ref = torch.cat([encode(current[i:i + 8]) for i in range(0, len(indices), 8)]).float()
            z_tgt = torch.cat([encode(target[i:i + 8]) for i in range(0, len(indices), 8)]).float()
        noise = torch.randn_like(z_tgt)
        energy = z_tgt.pow(2).mean().item()
        change = (z_tgt - z_ref).pow(2).mean().item()
        velocity = (noise - z_tgt).pow(2).mean().item()
        print(f"{name:20s} tokens x dims {tuple(z_tgt.shape[1:])}  E[z^2]={energy:.4f}  "
              f"E[(z_t+16 - z_t)^2]={change:.4f}  change/energy={change / energy:.4f}  E[(noise - z)^2]={velocity:.4f}",
              flush=True)
        del encode
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
