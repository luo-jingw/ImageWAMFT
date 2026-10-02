"""Is the predicted future frame shifted in time? (run in `.venv`, one GPU, from the ImageWAM directory)

A. Data: on every validation episode, cross-correlate per-frame image change of each camera region (head, left
   wrist, right wrist of the 288x256 tiling) with the joint speed of the matching arm. The lag with the highest
   correlation estimates how far each camera lags the recorded joint state (positive = video later).
B. Model: for validation samples spread over the episodes, predict frame t+16 from frame t (as in validation) and
   compare it with the ground-truth frames t+16+k, k in [-K, K], per region. The k with the lowest error estimates the
   temporal offset of the prediction (negative = the predicted scene is behind the ground truth).

  python scripts/analyze_prediction_time_offset.py --run-dir <run> --checkpoint <weights .pt> --out <json>
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from imagewam.runtime import build_datasets
from imagewam.utils import misc

REGIONS = {"full": (slice(0, 288), slice(0, 256)), "head": (slice(0, 192), slice(0, 256)),
           "left_wrist": (slice(192, 288), slice(0, 128)), "right_wrist": (slice(192, 288), slice(128, 256))}
ARM = {"head": slice(0, 14), "left_wrist": slice(0, 7), "right_wrist": slice(7, 14)}


def to01(frame: torch.Tensor) -> np.ndarray:
    return ((frame.float().clamp(-1, 1) + 1) / 2).numpy()


def episode_ranges(val) -> list[tuple[int, int]]:
    """Contiguous index ranges sharing one instruction (one validation episode per dataset root)."""
    n = len(val)
    instr = lambda i: val[i]["instruction"]  # noqa: E731
    ranges, start = [], 0
    while start < n:
        lo, hi, first = start, n - 1, instr(start)
        while lo < hi:  # last index with the same instruction
            mid = (lo + hi + 1) // 2
            if instr(mid) == first:
                lo = mid
            else:
                hi = mid - 1
        ranges.append((start, lo))
        start = lo + 1
    return ranges


def data_lags(val, ranges, max_lag: int) -> dict:
    out = {}
    for start, end in ranges:
        frames, states = [], []
        for j in range(start, end + 1):
            s = val[j]
            frames.append(to01(s["video"][:, 0]))
            states.append(s["proprio"][0].numpy())
        frames, states = np.stack(frames), np.stack(states)
        speed = np.abs(np.diff(states, axis=0))
        name = val[start]["instruction"].split(":", 1)[-1].strip()[:40]
        result = {}
        for region, arm in ARM.items():
            rs, cs = REGIONS[region]
            energy = np.abs(np.diff(frames[:, :, rs, cs], axis=0)).mean(axis=(1, 2, 3))
            joint = speed[:, arm].sum(axis=1)
            corr = {}
            for lag in range(-max_lag, max_lag + 1):  # corr(energy(t + lag), joint(t))
                a = energy[max(lag, 0): len(energy) + min(lag, 0)]
                b = joint[max(-lag, 0): len(joint) - max(lag, 0)]
                corr[lag] = float(np.corrcoef(a, b)[0, 1])
            best = max(corr, key=corr.get)
            result[region] = {"best_lag": best, "corr_at_best": round(corr[best], 3), "corr_at_0": round(corr[0], 3)}
        out[f"{name} [{start},{end}]"] = result
    return out


def predict(model, sample, steps: int, seed: int) -> np.ndarray:
    kwargs = dict(
        input_image=sample["video"][:, 0].unsqueeze(0),
        proprio=sample["proprio"][0],
        context=sample["text_hidden_states"],
        context_mask=sample["text_attention_mask"],
        num_inference_steps=steps,
        seed=seed,
    )
    with torch.no_grad():
        if model.stack == "flux2":
            image = model.infer_video_flux2(prompt=None, **kwargs)["image"]
        else:
            image = model._qwenimage21_stack().infer_image(**kwargs)["image"]
    return to01(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-episode", type=int, default=10)
    parser.add_argument("--max-offset", type=int, default=8)
    parser.add_argument("--data-lags", action="store_true", help="also run part A")
    args = parser.parse_args()
    torch.manual_seed(0)
    random.seed(0)

    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    misc.register_work_dir(str(args.out.parent / f"{args.out.stem}_work"))
    cfg.data.train.pretrained_norm_stats = str(args.run_dir / "dataset_stats.json")
    _, val = build_datasets(cfg.data)
    ranges = episode_ranges(val)
    result: dict = {"run_dir": str(args.run_dir), "checkpoint": str(args.checkpoint), "episodes": ranges}
    if args.data_lags:
        result["data_lags"] = data_lags(val, ranges, max_lag=6)
        print(json.dumps(result["data_lags"], indent=1), flush=True)

    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(args.checkpoint))
    model.eval()

    K, horizon = args.max_offset, 16
    cells = [(10, 0), (10, 1), (30, 0)]  # (sampling steps, seed)
    rows = []
    for start, end in ranges:
        lo, hi = start + K, end - horizon - K
        for i in np.linspace(lo, hi, args.samples_per_episode).round().astype(int).tolist():
            sample = val[i]
            gt = {k: to01(val[i + horizon + k]["video"][:, 0]) for k in range(-K, K + 1)}
            assert np.array_equal(gt[0], to01(sample["video"][:, -1])), f"index {i}: frames are not consecutive"
            current = to01(sample["video"][:, 0])
            for steps, seed in cells + [("copy_current", None)]:
                pred = current if steps == "copy_current" else predict(model, sample, steps, seed)
                mse = {region: {k: float(((pred[:, rs, cs] - g[:, rs, cs]) ** 2).mean()) for k, g in gt.items()}
                       for region, (rs, cs) in REGIONS.items()}
                rows.append({"index": i, "episode": start, "steps": steps, "seed": seed, "mse": mse})
        print(f"episode starting at {start}: done", flush=True)
    result["rows"] = rows

    summary = {}
    for steps, seed in cells + [("copy_current", None)]:
        cell = [r for r in rows if r["steps"] == steps and r["seed"] == seed]
        key = f"steps={steps},seed={seed}"
        summary[key] = {}
        for region in REGIONS:
            curve = {k: float(np.mean([r["mse"][region][k] for r in cell])) for k in range(-K, K + 1)}
            best = [min(r["mse"][region], key=r["mse"][region].get) for r in cell]
            summary[key][region] = {
                "argmin_mean_curve": min(curve, key=curve.get),
                "per_sample_best_k_mean": round(float(np.mean(best)), 2),
                "per_sample_best_k_median": float(np.median(best)),
                "per_sample_best_k_abs_mean": round(float(np.mean(np.abs(best))), 2),
                "psnr_k0": round(float(np.mean([10 * np.log10(1 / r["mse"][region][0]) for r in cell])), 2),
                "psnr_best_k": round(float(np.mean([10 * np.log10(1 / min(r["mse"][region].values())) for r in cell])), 2),
            }
    seed_pairs = {}
    for r0 in (r for r in rows if r["steps"] == 10 and r["seed"] == 0):
        r1 = next(r for r in rows if r["steps"] == 10 and r["seed"] == 1 and r["index"] == r0["index"])
        b0 = min(r0["mse"]["full"], key=r0["mse"]["full"].get)
        b1 = min(r1["mse"]["full"], key=r1["mse"]["full"].get)
        seed_pairs[r0["index"]] = (b0, b1)
    summary["seed0_vs_seed1_best_k_abs_diff_mean"] = round(float(np.mean([abs(a - b) for a, b in seed_pairs.values()])), 2)
    result["summary"] = summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
