"""All figures of the LIBERO image-token reduction study (run in `.venv-libero`); works on partial runs.

Deletes `figures/` and rebuilds it from the run directories:
  01_success_by_method.png    success rate per strategy and reduced-token count
  02_success_table.png        success rate per condition and suite, paired McNemar test against no reduction
  03_task_grid.png            successes per task and condition
  04_action_deviation.png     paired action deviation (reduced vs. unreduced inference on the same observation)
  05_methods_on_a_frame.png   what each strategy does to one input frame
  06_attention_by_layer.png   where the unreduced action expert attends, per layer
  07_attention_on_frames.png  action-to-image attention overlaid on frames
  08_kv_reuse.png             K/V reuse: the frame the canvas encodes, deviation per call, where top-9 refreshes
  09_kv_dose.png              open-loop dose test, reused vs. removed tokens (kv_dose.py)
  summary.json                every number the figures show
  index.html                  local page with all figures

Episode keys are (strategy, reduced tokens, suite, task, trial). The first grid (`--baseline`, files
`trial*_frac*.json`) holds the no-reduction and random-removal episodes; `--runs` hold the eval_token_mask.py
strategy grids (`<strategy>/<suite>/task*/trial*_rm*.json`).

  python make_figures.py
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

HERE = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": "#b9b8b2",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#ecebe7",
    "grid.linewidth": 0.8,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7b5ea7", "#2f8f9d", "#b5651d"]
SUITES = [("libero_spatial", "Spatial"), ("libero_object", "Object"), ("libero_goal", "Goal"), ("libero_10", "Long")]
IMAGE_TOKENS, GRID_H, GRID_W, CAMERA_W = 392, 14, 28, 14
TEXT_SLOTS = 513  # 512 text positions + the packed proprio token
WAIT = 30  # dummy steps before the first inference; rollout videos start after them

METHODS = {
    "none": ("完整输入", INK),
    "random": ("随机删除", SERIES[0]),
    "grid": ("规整网格删除", SERIES[1]),
    "attn_topk": ("注意力 top-k 剪枝", SERIES[5]),
    "evit": ("EViT 融合", SERIES[6]),
    "tome": ("ToMe 合并", SERIES[2]),
    "tofu": ("ToFu MLERP 合并", SERIES[7]),
    "kv_random": ("随机复用旧 KV", SERIES[3]),
    "kv_topk": ("复用低注意力旧 KV", SERIES[4]),
    "kv_randk": ("复用旧 KV、随机刷新", INK2),
}
CONDITIONS = [("none", 0), ("random", 131), ("random", 196), ("random", 261), ("grid", 131), ("grid", 196),
              ("grid", 261), ("attn_topk", 131), ("attn_topk", 196), ("attn_topk", 261), ("evit", 131), ("evit", 196),
              ("evit", 261), ("tome", 131), ("tome", 196), ("tome", 261), ("tofu", 196), ("tofu", 261),
              ("kv_random", 131), ("kv_random", 196), ("kv_random", 261), ("kv_topk", 131), ("kv_topk", 196),
              ("kv_topk", 261), ("kv_topk", 383), ("kv_randk", 383)]
SHARE = {131: "1/3", 196: "1/2", 261: "2/3"}
SHORT = {"none": "完整", "random": "随机删", "grid": "网格删", "attn_topk": "注意力删", "evit": "EViT", "tome": "ToMe",
         "tofu": "ToFu", "kv_random": "KV随机", "kv_topk": "KV注意力"}
REFRESH_9 = {"kv_topk": ("只刷新注意力 top-9", "KV top9"), "kv_randk": ("随机刷新 9 个", "KV 随机9")}  # 383 reused
LINE_STRATEGIES = ("random", "grid", "attn_topk", "evit", "tome", "tofu", "kv_random", "kv_topk")
X = {0: 0.0, 131: 1.0, 196: 2.0, 261: 3.0, 383: 4.4}
EXAMPLE = ("libero_goal", 3, 0)  # suite, task, trial of the per-frame illustrations


def label(strategy: str, n: int) -> str:
    if n == 383:
        return REFRESH_9[strategy][0]
    return METHODS[strategy][0] + (f" {SHARE[n]}" if n in SHARE else "")


def short(strategy: str, n: int) -> str:
    if n == 383:
        return REFRESH_9[strategy][1]
    return SHORT[strategy] + (f" {SHARE[n]}" if n in SHARE else "")


# ---- data ------------------------------------------------------------------------------------------------------

def load(baseline: Path, runs: list[Path]) -> dict:
    episodes = {}
    paths = [(p, True) for p in glob.glob(str(baseline / "*/task*/trial*_frac*.json"))]
    for run in runs:
        paths += [(p, False) for p in glob.glob(str(run / "*/*/task*/trial*_rm*.json"))]
    for path, first_grid in paths:
        meta = json.loads(Path(path).read_text())
        n = int(meta["masked_tokens"])
        if first_grid:
            meta["strategy"] = "none" if n == 0 else "random"
        meta["npz"] = path[:-5] + ".npz"
        meta["video_glob"] = str(Path(path).parent / f"*episode={Path(path).stem}--*.mp4")
        episodes[(meta["strategy"], n, meta["suite"], meta["task_id"], meta["trial"])] = meta
    return episodes


def rows_of(episodes: dict, strategy: str, n: int) -> list[dict]:
    return [m for k, m in episodes.items() if k[:2] == (strategy, n)]


def token_mask(z) -> np.ndarray:
    return np.unpackbits(z["token_mask"], axis=1)[:, :IMAGE_TOKENS].astype(bool)


def token_age(mask: np.ndarray) -> np.ndarray:
    """Calls since each token was last computed, per call [calls, 392]; 0 for tokens computed at that call."""
    age = np.zeros(mask.shape, dtype=np.int32)
    current = np.zeros(mask.shape[1], dtype=np.int32)
    for i, row in enumerate(mask):
        current = np.where(row, current + 1, 0)
        age[i] = current
    return age


def summarize(episodes: dict) -> dict:
    """Success with paired McNemar tests, paired action deviation and K/V entry age per condition."""
    base = {k[2:]: m["success"] for k, m in episodes.items() if k[0] == "none"}
    summary = {}
    for strategy, n in CONDITIONS:
        rows = rows_of(episodes, strategy, n)
        if not rows:
            continue
        entry = {"label": label(strategy, n), "strategy": strategy, "tokens": n, "success": {}}
        for suite in ["all"] + [s for s, _ in SUITES]:
            subset = rows if suite == "all" else [m for m in rows if m["suite"] == suite]
            if not subset:
                continue
            wins = sum(m["success"] for m in subset)
            p = wins / len(subset)
            pairs = [(base[(m["suite"], m["task_id"], m["trial"])], m["success"]) for m in subset]
            lost = sum(b and not s for b, s in pairs)
            gained = sum(s and not b for b, s in pairs)
            pvalue = binomtest(min(lost, gained), lost + gained, 0.5).pvalue if lost + gained else 1.0
            entry["success"][suite] = {"n": len(subset), "wins": wins, "rate": round(100 * p, 1),
                                       "se": round(100 * np.sqrt(p * (1 - p) / len(subset)), 1),
                                       "lost": lost, "gained": gained, "mcnemar_p": float(pvalue)}
        if n > 0:
            calls, by_index, ages = [], [], []
            for m in rows:
                z = np.load(m["npz"])
                dev = np.abs(z["action_masked"] - z["action_unmasked"]).mean(axis=(1, 2))
                # K/V reuse computes every token at the first call of an episode: no reduction there
                reduced = token_mask(z).any(axis=1) if strategy.startswith("kv") else np.ones(len(dev), dtype=bool)
                by_index.append(dev)
                calls.append(dev[reduced])
                if strategy.startswith("kv"):
                    mask = token_mask(z)
                    ages.append(token_age(mask)[mask])
            dev = np.concatenate(calls)
            padded = np.full((len(by_index), max(len(d) for d in by_index)), np.nan)
            for i, d in enumerate(by_index):
                padded[i, : len(d)] = d
            counts = np.isfinite(padded).sum(axis=0)
            entry["deviation"] = {
                "calls": int(len(dev)), "mean": float(dev.mean()), "median": float(np.median(dev)),
                "p10": float(np.percentile(dev, 10)), "p25": float(np.percentile(dev, 25)),
                "p75": float(np.percentile(dev, 75)), "p90": float(np.percentile(dev, 90)),
                "by_call_index": [float(v) for v in np.nanmean(padded[:, counts >= 20], axis=0)],
            }
            if ages:
                ages = np.concatenate(ages)
                entry["kv_age_calls"] = {"mean": float(ages.mean()), "median": float(np.median(ages)),
                                         "p90": float(np.percentile(ages, 90)), "max": int(ages.max())}
        summary[f"{strategy}:{n}"] = entry
    return summary


def attention_profile(episodes: dict) -> dict:
    """Unreduced action-to-prefix attention of the first grid (no-reduction episodes and the paired unreduced
    inferences of the random-removal episodes): shares per layer, image concentration, mean image map."""
    shares, top20, maps = [], [], []
    for key, meta in episodes.items():
        if key[0] not in ("none", "random") or key[1] not in (0, 131, 261):
            continue
        z = np.load(meta["npz"])
        attention = z["attn_masked" if key[1] == 0 else "attn_unmasked"].astype(np.float32)  # [calls, layers, 905]
        self_mass = z["attn_self_masked" if key[1] == 0 else "attn_self_unmasked"]
        text_part, image = attention[:, :, :TEXT_SLOTS], attention[:, :, TEXT_SLOTS:]
        # The proprio token sits right after the valid text; padded text columns receive exactly zero attention.
        proprio_index = int(np.nonzero(text_part.sum(axis=(0, 1)) > 0)[0].max())
        proprio = text_part[:, :, proprio_index]
        image_grid = image.reshape(*image.shape[:2], GRID_H, GRID_W)
        shares.append(np.stack([text_part.sum(axis=2) - proprio, proprio, image_grid[..., :CAMERA_W].sum(axis=(2, 3)),
                                image_grid[..., CAMERA_W:].sum(axis=(2, 3)), self_mass], axis=-1))
        p = image / np.clip(image.sum(axis=2, keepdims=True), 1e-12, None)
        top20.append(np.sort(p, axis=2)[:, :, -20:].sum(axis=2))
        maps.append(image_grid.mean(axis=0))
    return {"share_names": ["valid_text", "proprio", "agentview_image", "wrist_image", "action_tokens"],
            "share_per_layer": np.concatenate(shares).mean(axis=0).round(4).tolist(),
            "image_top20_mass_per_layer": np.concatenate(top20).mean(axis=0).round(4).tolist(),
            "mean_image_map": np.mean(maps, axis=0)}


# ---- frames ----------------------------------------------------------------------------------------------------

def frame_at(meta: dict, t: int) -> np.ndarray:
    """Model-input layout of control step t: agent view | wrist view, 224 × 448."""
    frame = imageio.get_reader(glob.glob(meta["video_glob"])[0]).get_data(int(t) - WAIT)  # [256, 512, 3]
    halves = [Image.fromarray(frame[:, :256]).resize((224, 224), Image.BILINEAR),
              Image.fromarray(frame[:, 256:]).resize((224, 224), Image.BILINEAR)]
    return np.concatenate([np.asarray(h) for h in halves], axis=1)


def patch(index: int) -> tuple[slice, slice]:
    r, c = divmod(int(index), GRID_W)
    return slice(r * 16, (r + 1) * 16), slice(c * 16, (c + 1) * 16)


def dimmed(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shown = frame.astype(np.float32).copy()
    for index in np.nonzero(mask)[0]:
        rows, cols = patch(index)
        shown[rows, cols] = shown[rows, cols] * 0.15 + 70
    return shown.clip(0, 255).astype(np.uint8)


def merged_view(frame: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Each token patch filled with the mean color of its merge group, with group borders."""
    shown = np.zeros_like(frame, dtype=np.float32)
    for g in np.unique(group):
        members = np.nonzero(group == g)[0]
        color = np.mean([frame[patch(i)].reshape(-1, 3).mean(axis=0) for i in members], axis=0)
        for i in members:
            shown[patch(i)] = color
    labels = group.reshape(GRID_H, GRID_W)
    for r in range(GRID_H):
        for c in range(GRID_W):
            if c + 1 < GRID_W and labels[r, c] != labels[r, c + 1]:
                shown[r * 16:(r + 1) * 16, (c + 1) * 16 - 1] = 252
            if r + 1 < GRID_H and labels[r, c] != labels[r + 1, c]:
                shown[(r + 1) * 16 - 1, c * 16:(c + 1) * 16] = 252
    return shown.clip(0, 255).astype(np.uint8)


def overlay(ax, frame: np.ndarray, heat: np.ndarray, title: str) -> None:
    heat = heat / max(heat.max(), 1e-12)
    up = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize((448, 224), Image.BILINEAR)) / 255.0
    ax.imshow(frame * 0.55 / 255.0)
    ax.imshow(up, cmap="inferno", alpha=np.clip(up * 1.1, 0, 0.85), vmin=0, vmax=1)
    ax.axvline(223.5, color="#fcfcfb", linewidth=1.5)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.axis("off")


def heatmap_axes(ax, rows: int, cols: int) -> None:
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, cols), minor=True)
    ax.set_yticks(np.arange(-0.5, rows), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=1.5)
    ax.tick_params(which="minor", length=0)


# ---- figures ---------------------------------------------------------------------------------------------------

def fig_success(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    base = summary["none:0"]["success"]["all"]["rate"]
    labels = []
    # one line per strategy over 1/3, 1/2, 2/3; the 9-token K/V refresh conditions are separate points
    series = [(s, [n for c, n in CONDITIONS if c == s and 0 < n < 383]) for s in LINE_STRATEGIES]
    series += [("kv_topk", [383]), ("kv_randk", [383])]
    for strategy, counts in series:
        points = [(n, summary[f"{strategy}:{n}"]["success"]["all"]) for n in counts if f"{strategy}:{n}" in summary]
        if not points:
            continue
        if points[0][0] == 131:  # anchor at no reduction only from the first level
            points.insert(0, (0, summary["none:0"]["success"]["all"]))
        name, color = METHODS[strategy]
        xs = [X[n] + ({"kv_topk": -0.15, "kv_randk": 0.15}[strategy] if n == 383 else 0.0) for n, _ in points]
        ys = [r["rate"] for _, r in points]
        ax.errorbar(xs, ys, yerr=[r["se"] for _, r in points], color=color, linewidth=2.2, marker="o", markersize=7,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, capsize=3, elinewidth=1,
                    linestyle="-" if len(points) > 1 else "none")
        n = points[-1][0]
        labels.append([xs[-1] + 0.1 if n < 383 else 4.75, ys[-1],
                       f"{label(strategy, n) if n == 383 else name} {ys[-1]:.1f}%", color])
    ax.annotate(f"完整输入 {base:.0f}%", (0, base), textcoords="offset points", xytext=(0, 11), ha="center",
                fontsize=10, color=INK, fontweight="bold")
    for column in sorted({round(x, 1) for x, *_ in labels}):
        group = sorted([item for item in labels if round(item[0], 1) == column], key=lambda item: item[1])
        for i in range(1, len(group)):  # spread labels that would overlap
            group[i][1] = max(group[i][1], group[i - 1][1] + 4.2)
        for x, y, text, color in group:
            ax.text(x, y, text, color=color, va="center", fontsize=9.5)
    ax.axvspan(3.7, 6.3, color="#f4f3ef", zorder=0)
    ax.text(5.0, 0.985, "复用旧 KV，只重新计算 9 个 token", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=9, color=MUTED)
    ax.set_xticks([X[n] for n in (0, 131, 196, 261, 383)],
                  ["0\n(完整 392 个)", "131\n(1/3)", "196\n(1/2)", "261\n(2/3)", "383\n(只算 9 个)"])
    ax.set_xlabel("被删除 / 合并掉 / 复用旧 KV 的图像 token 数")
    ax.set_xlim(-0.3, 6.3)
    lowest = min(item[1] for item in labels)
    ax.set_ylim(lowest - 8 if lowest >= 8 else -5, 106)
    ax.set_ylabel("成功率 (%)")
    ax.set_title("各种减少图像 token 的方法在 LIBERO 上的成功率（每点 200 回合，误差线 = 标准误）", loc="left",
                 fontsize=11, color=INK)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "01_success_by_method.png", dpi=170)
    plt.close(fig)


def fig_table(summary: dict, out: Path) -> None:
    keys = [f"{s}:{n}" for s, n in CONDITIONS if f"{s}:{n}" in summary]
    columns = [("all", "全部")] + list(SUITES)
    rates = np.array([[summary[k]["success"].get(c, {}).get("rate", np.nan) for c, _ in columns] for k in keys])
    fig, ax = plt.subplots(figsize=(8.8, 0.46 * len(keys) + 1.4))
    ax.imshow(100 - rates, cmap="Oranges", vmin=0, vmax=100, aspect="auto")
    for (r, c), value in np.ndenumerate(rates):
        cell = summary[keys[r]]["success"].get(columns[c][0])
        if cell is None:
            continue
        mark = "**" if cell["mcnemar_p"] < 0.01 else "*" if cell["mcnemar_p"] < 0.05 else ""
        detail = f"  −{cell['lost']}/+{cell['gained']}" if keys[r] != "none:0" else ""
        ax.text(c, r, (f"{value:.1f}" if c == 0 else f"{value:.0f}") + mark + detail, ha="center", va="center",
                fontsize=9, color="#fcfcfb" if value < 45 else INK)
    ax.set_xticks(range(len(columns)), [name for _, name in columns])
    ax.set_yticks(range(len(keys)), [summary[k]["label"] + ("" if k in ("none:0", "kv_topk:383", "kv_randk:383")
                                                          else f"（{summary[k]['tokens']} 个）") for k in keys])
    heatmap_axes(ax, len(keys), len(columns))
    ax.set_title("成功率 (%)；−丢/+得 = 与完整输入同一任务、同一初始状态相比，由成功变失败 / 由失败变成功的回合数\n"
                 "* p<0.05，** p<0.01（配对 McNemar 检验）", loc="left", fontsize=9.5, color=INK2, pad=24)
    fig.tight_layout()
    fig.savefig(out / "02_success_table.png", dpi=170)
    plt.close(fig)


def fig_tasks(episodes: dict, summary: dict, out: Path) -> None:
    keys = [(s, n) for s, n in CONDITIONS if f"{s}:{n}" in summary]
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    for ax, (suite, name) in zip(axes.ravel(), SUITES):
        grid = np.full((10, len(keys)), np.nan)
        texts = []
        for task in range(10):
            for col, (s, n) in enumerate(keys):
                trials = [m for trial in range(5) if (m := episodes.get((s, n, suite, task, trial))) is not None]
                if trials:
                    grid[task, col] = sum(m["success"] for m in trials) / len(trials)
            text = episodes[("none", 0, suite, task, 0)]["task"]
            texts.append(text if len(text) <= 40 else text[:38] + "…")
        ax.imshow(1 - grid, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
        for (r, c), value in np.ndenumerate(grid):
            if np.isfinite(value):
                ax.text(c, r, f"{value * 5:.0f}", ha="center", va="center", fontsize=8,
                        color="#fcfcfb" if value < 0.45 else INK)
        ax.set_xticks(range(len(keys)), [short(s, n) for s, n in keys], rotation=60, ha="left", fontsize=8)
        ax.set_yticks(range(10), [f"{i}  {t}" for i, t in enumerate(texts)], fontsize=7.5)
        ax.set_title(name, loc="left", fontsize=11, color=INK, pad=4)
        heatmap_axes(ax, 10, len(keys))
    fig.suptitle("每个任务的成功次数（满分 5；颜色越深失败越多）", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "03_task_grid.png", dpi=150)
    plt.close(fig)


def fig_deviation(summary: dict, out: Path) -> None:
    keys = [f"{s}:{n}" for s, n in CONDITIONS if n > 0 and "deviation" in summary.get(f"{s}:{n}", {})]
    fig, ax = plt.subplots(figsize=(9.6, 0.42 * len(keys) + 1.6))
    for i, key in enumerate(keys):
        d = summary[key]["deviation"]
        color = METHODS[summary[key]["strategy"]][1]
        y = len(keys) - 1 - i
        ax.plot([d["p10"], d["p90"]], [y, y], color=color, linewidth=1.2)
        ax.add_patch(plt.Rectangle((d["p25"], y - 0.28), d["p75"] - d["p25"], 0.56, facecolor=color, alpha=0.3,
                                   edgecolor=color))
        ax.plot([d["median"]] * 2, [y - 0.28, y + 0.28], color=color, linewidth=2.5)
        rate = summary[key]["success"]["all"]["rate"]
        ax.text(d["p90"] + 0.006, y, f"中位数 {d['median']:.3f} ｜ 成功率 {rate:.1f}%", va="center", fontsize=8.5,
                color=INK2)
    ax.set_yticks(range(len(keys)), [summary[k]["label"] for k in reversed(keys)])
    ax.set_xlim(0, max(summary[k]["deviation"]["p90"] for k in keys) * 1.45)
    ax.set_xlabel("动作偏差：同一观测下，减少 token 与完整输入的动作差（归一化单位，16×7 平均）")
    ax.set_title("每次推理的动作偏差（框 = 25–75%，线 = 10–90%，粗竖线 = 中位数）", loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "04_action_deviation.png", dpi=170)
    plt.close(fig)


def task_wins(episodes: dict, strategy: str, n: int) -> int:
    return sum(m["success"] for trial in range(5)
               if (m := episodes.get((strategy, n, EXAMPLE[0], EXAMPLE[1], trial))) is not None)


def fig_methods(episodes: dict, out: Path) -> None:
    base = episodes[("none", 0) + EXAMPLE]
    frame = frame_at(base, np.load(base["npz"])["t"][0])  # the first call sees the same state in every condition
    panels = [("原始输入（14×28 = 392 个 token）", frame)]
    for strategy, n, title in (("random", 196, "随机删除 1/2（每次推理重新随机）"), ("grid", 196, "棋盘格删除 1/2"),
                               ("grid", 131, "网格删除 1/3：(行+列)%3==0"),
                               ("grid", 261, "网格删除 2/3：只留 (行+列)%3==0"),
                               ("attn_topk", 261, "注意力 top-k 剪枝 2/3（暗块 = 删除）"),
                               ("evit", 261, "EViT 2/3（暗块 = 融合成 1 个 token）")):
        meta = episodes.get((strategy, n) + EXAMPLE)
        if meta is not None:
            mask = token_mask(np.load(meta["npz"]))[0]
            panels.append((f"{title}；该任务成功 {task_wins(episodes, strategy, n)}/5", dimmed(frame, mask)))
    for strategy, name in (("tome", "ToMe"), ("tofu", "ToFu MLERP")):
        for n in (196, 261):
            meta = episodes.get((strategy, n) + EXAMPLE)
            if meta is not None:
                group = np.load(meta["npz"])["merge_group"][0]
                panels.append((f"{name} 合并成 {IMAGE_TOKENS - n} 个（同组填平均色）；"
                               f"成功 {task_wins(episodes, strategy, n)}/5", merged_view(frame, group)))
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, 3.25 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, image) in zip(axes.ravel(), panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    fig.suptitle(f"各方法对同一帧输入做了什么（左：第三人称相机，右：腕部相机）：{base['task']}",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    fig.savefig(out / "05_methods_on_a_frame.png", dpi=150)
    plt.close(fig)


def fig_attention_layers(profile: dict, out: Path) -> None:
    shares = np.array(profile["share_per_layer"])  # [layers, 5]
    names = ["文本（有效 token）", "proprio", "第三人称相机", "腕部相机", "动作 token 自身"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    layers = np.arange(shares.shape[0])
    bottom = np.zeros(len(layers))
    for i, (name, color) in enumerate(zip(names, SERIES)):
        ax1.bar(layers, shares[:, i], bottom=bottom, color=color, width=0.82, edgecolor="#fcfcfb", linewidth=1,
                label=name)
        bottom += shares[:, i]
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("注意力占比")
    ax1.legend(ncol=5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    ax1.grid(axis="x", visible=False)
    ax2.plot(layers, profile["image_top20_mass_per_layer"], color=SERIES[0], linewidth=2, marker="o", markersize=5)
    ax2.axhline(20 / IMAGE_TOKENS, color=MUTED, linestyle="--", linewidth=1)
    ax2.text(24.4, 20 / IMAGE_TOKENS + 0.02, "均匀分布时 = 5.1%", ha="right", fontsize=8.5, color=MUTED)
    ax2.set_ylim(0, 0.8)
    ax2.set_ylabel("前 20 token 占比")
    ax2.set_xlabel("动作专家层号（0–4 为 double block，5–24 为 single block）")
    ax2.set_xticks(layers)
    ax2.grid(axis="x", visible=False)
    ax1.set_title("动作专家每层注意力的去向（不减少 token 时的推理，第一批 600 回合平均）", loc="left", fontsize=11,
                  color=INK, pad=30)
    fig.tight_layout()
    fig.savefig(out / "06_attention_by_layer.png", dpi=170)
    plt.close(fig)


def fig_attention_frames(episodes: dict, out: Path) -> None:
    examples = [("libero_spatial", 0), ("libero_goal", 3), ("libero_10", 8)]
    layer_groups = [("早期层 0–4", slice(0, 5)), ("中间层 8–15", slice(8, 16)), ("后期层 16–24", slice(16, 25))]
    fig, axes = plt.subplots(len(examples), 3, figsize=(13, 2.45 * len(examples) + 0.5))
    for row, (suite, task) in enumerate(examples):
        meta = episodes[("none", 0, suite, task, 0)]
        z = np.load(meta["npz"])
        call = int(len(z["t"]) * 0.45)
        frame = frame_at(meta, z["t"][call])
        image_att = z["attn_masked"].astype(np.float32)[call, :, TEXT_SLOTS:].reshape(-1, GRID_H, GRID_W)
        header = f"{dict(SUITES)[suite]} 任务 {task}（第 {z['t'][call]} 步）：{meta['task']}"
        for col, (name, layers) in enumerate(layer_groups):
            overlay(axes[row, col], frame, image_att[layers].mean(axis=0), f"{header if col == 0 else ''}\n{name}")
    fig.suptitle("动作专家把注意力放在画面的哪里（左：第三人称相机，右：腕部相机；越亮注意力越高，每格单独归一化）",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.2, w_pad=0.6)
    fig.savefig(out / "07_attention_on_frames.png", dpi=150)
    plt.close(fig)


def canvas_view(meta: dict, z, call: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The current frame, the frame the K/V canvas encodes (each token from the call that last computed it) and the
    age of each token in calls."""
    age = token_age(token_mask(z))[call]
    frames = {}
    shown = np.zeros((224, 448, 3), dtype=np.uint8)
    for index in range(IMAGE_TOKENS):
        source = call - int(age[index])
        if source not in frames:
            frames[source] = frame_at(meta, z["t"][source])
        shown[patch(index)] = frames[source][patch(index)]
    return frames[call] if call in frames else frame_at(meta, z["t"][call]), shown, age


def fig_kv(episodes: dict, summary: dict, out: Path) -> None:
    examples = [(s, n) for s, n in (("kv_topk", 131), ("kv_random", 131)) if (s, n) + EXAMPLE in episodes]
    fig = plt.figure(figsize=(13, 3.0 * len(examples) + 4.4))
    grid = fig.add_gridspec(len(examples) + 1, 2, height_ratios=[1] * len(examples) + [1.25])
    for row, (strategy, n) in enumerate(examples):
        meta = episodes[(strategy, n) + EXAMPLE]
        z = np.load(meta["npz"])
        call = min(len(z["t"]) - 1, 6)
        current, shown, age = canvas_view(meta, z, call)
        ax1, ax2 = fig.add_subplot(grid[row, 0]), fig.add_subplot(grid[row, 1])
        ax1.imshow(current)
        rate = summary[f"{strategy}:{n}"]["success"]["all"]["rate"]
        ax1.set_title(f"{label(strategy, n)}（成功率 {rate:.1f}%）：第 {call + 1} 次推理时的真实画面（第 {z['t'][call]} 步）",
                      fontsize=9.5, loc="left", color=INK)
        ax2.imshow(shown)
        for index in np.nonzero(age > 0)[0]:
            r, c = divmod(int(index), GRID_W)
            ax2.add_patch(plt.Rectangle((c * 16 - 0.5, r * 16 - 0.5), 16, 16, fill=False, edgecolor="#eb6834",
                                        linewidth=0.8))
        ax2.set_title(f"KV 画布实际编码的画面（橙框 = 复用旧 KV 的 token，平均已 {age[age > 0].mean():.1f} 次推理未更新）",
                      fontsize=9.5, loc="left", color=INK)
        for ax in (ax1, ax2):
            ax.axis("off")
    ax3 = fig.add_subplot(grid[-1, 0])
    light = {("random", 131): "#9cc3ef", ("kv_random", 131): "#f5cf66", ("kv_topk", 131): "#f3b5cf"}
    for strategy, n in (("random", 131), ("random", 261), ("kv_random", 131), ("kv_random", 261), ("kv_topk", 131),
                        ("kv_topk", 261), ("kv_topk", 383), ("kv_randk", 383)):
        entry = summary.get(f"{strategy}:{n}", {}).get("deviation")
        if entry is None:
            continue
        values = entry["by_call_index"]
        ax3.plot(np.arange(1, len(values) + 1), values, color=light.get((strategy, n), METHODS[strategy][1]),
                 linewidth=2, linestyle="--" if strategy == "random" else ":" if n == 383 else "-",
                 label=label(strategy, n))
    ax3.set_xlabel("回合内第几次推理（每 12 步一次）")
    ax3.set_ylabel("平均动作偏差")
    ax3.set_title("偏差随回合推进的变化（虚线 = 直接删除，作参照）", loc="left", fontsize=10, color=INK)
    ax3.legend(frameon=False, fontsize=7.5, ncol=2)
    ax4 = fig.add_subplot(grid[-1, 1])
    counts, total = np.zeros(IMAGE_TOKENS), 0
    for meta in rows_of(episodes, "kv_topk", 131):
        mask = token_mask(np.load(meta["npz"]))
        counts += mask[1:].sum(axis=0)  # the first call computes every token
        total += len(mask) - 1
    if total:
        image = ax4.imshow((counts / total).reshape(GRID_H, GRID_W), cmap="magma", vmin=0, vmax=1)
        ax4.axvline(GRID_W / 2 - 0.5, color="#fcfcfb", linewidth=1.5)
        ax4.set_xticks([GRID_W / 4 - 0.5, 3 * GRID_W / 4 - 0.5], ["第三人称相机", "腕部相机"])
        ax4.set_yticks([])
        ax4.tick_params(length=0)
        ax4.grid(False)
        fig.colorbar(image, ax=ax4, fraction=0.025, pad=0.01, label="复用旧 KV 的推理占比")
        ax4.set_title(f"复用低注意力旧 KV 1/3：各位置被复用的频率（{len(rows_of(episodes, 'kv_topk', 131))} 回合汇总）",
                      loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "08_kv_reuse.png", dpi=150)
    plt.close(fig)


def fig_kv_dose(dose: dict, out: Path) -> None:
    counts = sorted(int(k) for k in dose["summary"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for key, name, color in (("stale", "复用旧 KV", SERIES[3]), ("removed", "直接删除", SERIES[0])):
        ax1.plot(counts, [dose["summary"][str(c)][f"{key}_deviation"] for c in counts], color=color, marker="o",
                 linewidth=2, label=name)
        ax2.plot(counts, [100 * dose["summary"][str(c)][f"{key}_gripper_flip"] for c in counts], color=color,
                 marker="o", linewidth=2, label=name)
    ax1.set_ylabel("与完整推理的动作偏差")
    ax2.set_ylabel("夹爪开合指令翻转的步数占比 (%)")
    for ax in (ax1, ax2):
        ax.set_xlabel("被复用 / 被删除的图像 token 数（同一组随机 token）")
        ax.set_xticks(counts)
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(frameon=False)
    n = dose["summary"][str(counts[-1])]["n"]
    fig.suptitle(f"开环剂量测试：在保存的仿真状态上，同样的 token 复用旧 KV 还是直接删除（{n} 个状态，KV 来自上一次推理）",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "09_kv_dose.png", dpi=170)
    plt.close(fig)


# ---- page ------------------------------------------------------------------------------------------------------

def captions(summary: dict) -> dict[str, str]:
    def rate(key: str) -> str:
        return f"{summary[key]['success']['all']['rate']:.1f}%" if key in summary else "–"

    return {
        "01_success_by_method.png": (
            f"去掉 2/3 的图像 token：按注意力保留 {rate('attn_topk:261')}（EViT {rate('evit:261')}），"
            f"规整网格 {rate('grid:261')}，随机 {rate('random:261')}；ToMe {rate('tome:261')}，ToFu {rate('tofu:261')}。"
            f"复用旧 KV 1/3：按注意力刷新 {rate('kv_topk:131')}，随机 {rate('kv_random:131')}。"),
        "02_success_table.png": "每个条件 200 回合（4 个 suite × 10 任务 × 5 初始状态），与完整输入逐回合配对检验。",
        "03_task_grid.png": "逐任务的成功次数（每格满分 5）。",
        "04_action_deviation.png": "同一观测下减少 token 与完整输入的动作差；复用旧 KV 只统计实际复用的推理。",
        "05_methods_on_a_frame.png": "同一帧上各方法删掉、融合或合并了哪些 token；注意力保留的是机械臂、碗、盘子和腕部视野里的物体。",
        "06_attention_by_layer.png": "早期和后期层主要看文本，中间层主要看腕部相机；中间层里 392 个图像 token 中前 20 个占约一半图像注意力。",
        "07_attention_on_frames.png": "注意力叠加在真实画面上：中间层集中在腕部相机里的夹爪和被抓物体。",
        "08_kv_reuse.png": "复用 1/3 旧 KV：按上次注意力选择复用低注意力 token（上）时画布与当前画面基本一致，随机复用（中）会把移动中的机械臂和夹爪换成旧画面；被复用的主要是静止背景（右下）。",
        "09_kv_dose.png": "在相同的仿真状态和相同的 token 上，复用旧 KV 的偏差是直接删除的 3–4 倍；只有几乎全部复用时才反过来。",
    }


INDEX = """<!doctype html><meta charset="utf-8"><title>LIBERO image-token reduction</title>
<style>body{{font-family:"Noto Sans CJK SC",sans-serif;background:#fcfcfb;color:#0b0b0b;max-width:1180px;margin:0 auto;
padding:24px 16px}}h1{{font-size:20px}}figure{{margin:0 0 36px}}img{{max-width:100%;border:1px solid #ecebe7}}
figcaption,li{{color:#52514e;font-size:14px;line-height:1.6}}</style>
<h1>LIBERO：减少 ImageWAM-FLUX.2-4B 的图像 token（OSMesa 渲染，每个条件 200 回合）</h1>
<ul>
<li>随机删除 / 规整网格删除：在注意力中屏蔽被选中的 token（等价于从序列中拿掉）；随机的每次推理重新选，网格为固定的斜格或棋盘格。</li>
<li>注意力 top-k 剪枝：每次推理先用完整输入跑 prefill 和第 1 步去噪，按动作对图像的注意力保留最高的 token，删除其余。</li>
<li>EViT：与 top-k 相同的选择，但被去掉的 token 按注意力加权融合成 1 个 token。</li>
<li>ToMe / ToFu MLERP：在 VAE latent 上按相似度两两匹配合并，注意力加 log(大小)；ToFu 合并后保持组内最大范数。</li>
<li>复用旧 KV：被选中的 token 不重新计算，每层的 K/V 取该 token 上次计算时的值（每回合第一次推理全算）；按注意力刷新 = 只重新计算上一次推理注意力最高的 token。</li>
</ul>
{figures}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, default=HERE / "runs" / "20261004_osmesa_grid")
    parser.add_argument("--runs", type=Path, nargs="+", default=[HERE / "runs" / name for name in (
        "20261004_reduce_grid", "20261005_attn_merge_grid", "20261005_kv_topk_ratio")])
    parser.add_argument("--kv-dose", type=Path, default=HERE / "runs" / "20261004_reduce_grid" / "kv_dose.json")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)
    episodes = load(args.baseline, [run for run in args.runs if run.exists()])
    summary = summarize(episodes)
    profile = attention_profile(episodes)
    dose = json.loads(args.kv_dose.read_text())
    fig_success(summary, args.out)
    fig_table(summary, args.out)
    fig_tasks(episodes, summary, args.out)
    fig_deviation(summary, args.out)
    fig_methods(episodes, args.out)
    fig_attention_layers(profile, args.out)
    fig_attention_frames(episodes, args.out)
    fig_kv(episodes, summary, args.out)
    fig_kv_dose(dose, args.out)
    profile.pop("mean_image_map")
    (args.out / "summary.json").write_text(json.dumps(
        {"conditions": summary, "attention": profile,
         "kv_dose": {**dose["summary"], "max_restore_error": dose["max_restore_error"]}}, indent=1, ensure_ascii=False))
    text = captions(summary)
    figures = "\n".join(f'<figure><img src="{name}" alt="{name}"><figcaption>{caption}</figcaption></figure>'
                        for name, caption in text.items())
    (args.out / "index.html").write_text(INDEX.format(figures=figures), encoding="utf-8")
    for key, entry in summary.items():
        s, d = entry["success"]["all"], entry.get("deviation", {})
        print(f"{key:14s} n={s['n']:3d} SR {s['rate']:5.1f} ±{s['se']:.1f}  −{s['lost']}/+{s['gained']} "
              f"p={s['mcnemar_p']:.2g}  dev median {d.get('median', 0):.3f}  "
              + " ".join(f"{name} {entry['success'][suite]['rate']:.0f}" for suite, name in SUITES
                         if suite in entry["success"]))
    print("written:", sorted(p.name for p in args.out.iterdir()))


if __name__ == "__main__":
    main()
