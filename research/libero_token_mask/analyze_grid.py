"""Summarizes a token-mask grid run (eval_token_mask.py output); works on partial runs.

Writes <run>/analysis/summary.json and figures, and prints:
  1. success rate per mask fraction, overall and per suite, with binomial standard errors;
  2. paired action deviation (masked vs. unmasked inference, normalized action space) per fraction, and per-episode
     deviation of failed vs. successful episodes;
  3. attention profile of the unmasked model per layer (shares of valid text, proprio, agent-view image, wrist image,
     action tokens) and concentration over the 392 image tokens;
  4. whether the deviation of a call is explained by the unmasked attention share that fell on its masked tokens.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

IMAGE_TOKENS, GRID_H, GRID_W, CAMERA_W = 392, 14, 28, 14
TEXT_SLOTS = 513  # 512 text positions + packed proprio token


def load(run: Path) -> list[dict]:
    episodes = []
    for meta_path in sorted(run.glob("*/task*/trial*_frac*.json")):
        meta = json.loads(meta_path.read_text())
        arrays = np.load(meta_path.with_suffix(".npz"))
        meta["arrays"] = {key: arrays[key] for key in arrays.files}
        episodes.append(meta)
    return episodes


def unmasked_attention(episode: dict) -> np.ndarray:
    arrays = episode["arrays"]
    key = "attn_masked" if episode["masked_tokens"] == 0 else "attn_unmasked"
    return arrays[key].astype(np.float32)  # [calls, layers, prefix]


def attention_profile(episodes: list[dict]) -> dict:
    shares, image_maps, entropies, top20 = [], [], [], []
    for episode in episodes:
        attention = unmasked_attention(episode)
        self_key = "attn_self_masked" if episode["masked_tokens"] == 0 else "attn_self_unmasked"
        self_mass = episode["arrays"][self_key]
        text_part = attention[:, :, :TEXT_SLOTS]
        image = attention[:, :, TEXT_SLOTS:]
        # Proprio sits right after the valid text; padded text columns receive exactly zero attention.
        nonzero = np.nonzero(text_part.sum(axis=(0, 1)) > 0)[0]
        proprio_index = int(nonzero.max())
        proprio = text_part[:, :, proprio_index]
        text = text_part.sum(axis=2) - proprio
        image_grid = image.reshape(*image.shape[:2], GRID_H, GRID_W)
        agent = image_grid[..., :CAMERA_W].sum(axis=(2, 3))
        wrist = image_grid[..., CAMERA_W:].sum(axis=(2, 3))
        shares.append(np.stack([text, proprio, agent, wrist, self_mass], axis=-1))  # [calls, layers, 5]
        p = image / np.clip(image.sum(axis=2, keepdims=True), 1e-12, None)
        entropies.append(-(p * np.log2(np.clip(p, 1e-12, None))).sum(axis=2))
        top20.append(np.sort(p, axis=2)[:, :, -20:].sum(axis=2))
        image_maps.append(image_grid.mean(axis=0))
    shares = np.concatenate(shares)
    return {
        "share_names": ["valid_text", "proprio", "agentview_image", "wrist_image", "action_tokens"],
        "share_per_layer": shares.mean(axis=0).round(4).tolist(),
        "image_entropy_bits_per_layer": np.concatenate(entropies).mean(axis=0).round(3).tolist(),
        "image_entropy_uniform_bits": float(np.log2(IMAGE_TOKENS)),
        "image_top20_mass_per_layer": np.concatenate(top20).mean(axis=0).round(4).tolist(),
        "mean_image_map": np.mean(image_maps, axis=0),  # [layers, 14, 28]
    }


def deviation_analysis(episodes: list[dict]) -> dict:
    out = {}
    for fraction in sorted({e["mask_fraction"] for e in episodes if e["masked_tokens"] > 0}):
        calls_dev, masked_share, ep_dev, ep_success = [], [], [], []
        for episode in (e for e in episodes if e["mask_fraction"] == fraction):
            arrays = episode["arrays"]
            dev = np.abs(arrays["action_masked"] - arrays["action_unmasked"]).mean(axis=(1, 2))  # [calls]
            mask = np.unpackbits(arrays["token_mask"], axis=1)[:, :IMAGE_TOKENS].astype(bool)
            image = arrays["attn_unmasked"].astype(np.float32)[:, :, TEXT_SLOTS:].mean(axis=1)  # layer mean
            share = (image * mask).sum(axis=1) / np.clip(image.sum(axis=1), 1e-12, None)
            calls_dev.append(dev)
            masked_share.append(share)
            ep_dev.append(float(dev.mean()))
            ep_success.append(bool(episode["success"]))
        dev, share = np.concatenate(calls_dev), np.concatenate(masked_share)
        rho, pval = spearmanr(share, dev) if len(dev) > 2 else (float("nan"), float("nan"))
        ep_dev, ep_success = np.array(ep_dev), np.array(ep_success)
        out[f"{fraction:.3f}"] = {
            "calls": int(len(dev)),
            "deviation_mean": float(dev.mean()),
            "deviation_median": float(np.median(dev)),
            "deviation_p90": float(np.percentile(dev, 90)),
            "masked_attention_share_mean": float(share.mean()),
            "spearman_masked_share_vs_deviation": [float(rho), float(pval)],
            "episode_deviation_success": float(ep_dev[ep_success].mean()) if ep_success.any() else None,
            "episode_deviation_failure": float(ep_dev[~ep_success].mean()) if (~ep_success).any() else None,
        }
    return out


def success_table(episodes: list[dict]) -> dict:
    table = {}
    for fraction in sorted({e["mask_fraction"] for e in episodes}):
        rows = [e for e in episodes if e["mask_fraction"] == fraction]
        entry = {}
        for suite in ["all"] + sorted({e["suite"] for e in rows}):
            subset = rows if suite == "all" else [e for e in rows if e["suite"] == suite]
            n = len(subset)
            p = sum(e["success"] for e in subset) / n
            entry[suite] = {"n": n, "success_rate": round(100 * p, 1), "se": round(100 * np.sqrt(p * (1 - p) / n), 1)}
        table[f"{fraction:.3f}"] = entry
    return table


def save_heatmaps(profile: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps = profile["mean_image_map"]
    layers = [0, 4, 9, 14, 19, 24]
    fig, axes = plt.subplots(len(layers), 1, figsize=(6, 1.8 * len(layers)))
    for ax, layer in zip(axes, layers):
        ax.imshow(maps[layer], cmap="magma")
        ax.axvline(CAMERA_W - 0.5, color="white", linewidth=1)
        ax.set_title(f"layer {layer}: action-to-image attention (left agent view, right wrist)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "image_attention_layers.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    episodes = load(args.run)
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)
    summary = {"episodes": len(episodes), "success": success_table(episodes), "deviation": deviation_analysis(episodes)}
    profile = attention_profile(episodes)
    save_heatmaps(profile, out)
    np.save(out / "mean_image_map.npy", profile.pop("mean_image_map"))
    summary["attention"] = profile
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("episodes", "success", "deviation")}, indent=1))
    names = profile["share_names"]
    print("attention share per layer (" + ", ".join(names) + "):")
    for layer, row in enumerate(profile["share_per_layer"]):
        print(f"  layer {layer:2d}: " + "  ".join(f"{v:.3f}" for v in row)
              + f" | image entropy {profile['image_entropy_bits_per_layer'][layer]:.2f} bits"
              + f" | top-20 image tokens {profile['image_top20_mass_per_layer'][layer]:.3f}")


if __name__ == "__main__":
    main()
