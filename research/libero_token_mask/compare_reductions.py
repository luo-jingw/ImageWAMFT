"""Compares the image-token reduction strategies on LIBERO (run in `.venv-libero`); works on partial runs.

Reads the no-reduction and random-removal episodes of the first grid (`--baseline`) and the strategy grid of
eval_token_mask.py (`--run`); writes summary.json, PNG figures and a local index.html to `--out`:
  1. success rate per condition, overall and per suite, with binomial standard errors and the exact McNemar test
     against the no-reduction episode of the same task and initial state;
  2. paired action deviation (reduced vs. unreduced inference from the same observation) per condition; K/V-reuse
     conditions count only calls that reused tokens (the first call of an episode computes all);
  3. K/V reuse: age of the reused entries, deviation per call index, where kv_topk refreshes;
  4. what each strategy does to one input frame, and the frame the K/V canvas effectively encodes.

  python compare_reductions.py --baseline runs/20261004_osmesa_grid --run runs/20261004_reduce_grid --out figures/reduce
"""

import argparse
import glob
import json
from pathlib import Path

import imageio
import numpy as np
from PIL import Image
from scipy.stats import binomtest

from make_figures import GRID_H, GRID_W, INK, INK2, MUTED, SERIES, SUITES, WAIT, plt  # also sets rcParams

IMAGE_TOKENS = 392
METHODS = {
    "none": ("完整输入", INK),
    "random": ("随机删除", SERIES[0]),
    "grid": ("规整网格删除", SERIES[1]),
    "tome": ("ToMe 合并", SERIES[2]),
    "kv_random": ("随机复用旧 KV", SERIES[3]),
    "kv_topk": ("只刷新注意力 top-9", SERIES[4]),
    "kv_randk": ("随机刷新 9 个", INK2),
    "attn_topk": ("注意力 top-k 剪枝", "#7b5ea7"),
    "evit": ("EViT 融合", "#2f8f9d"),
    "tofu": ("ToFu MLERP 合并", "#b5651d"),
}
CONDITIONS = [("none", 0), ("random", 131), ("random", 196), ("random", 261), ("grid", 131), ("grid", 196),
              ("grid", 261), ("attn_topk", 131), ("attn_topk", 196), ("attn_topk", 261), ("tome", 131), ("tome", 196),
              ("tome", 261), ("tofu", 196), ("tofu", 261), ("evit", 131), ("evit", 196), ("evit", 261),
              ("kv_random", 131), ("kv_random", 261), ("kv_topk", 383), ("kv_randk", 383)]
SHARE = {131: "1/3", 196: "1/2", 261: "2/3"}
SHORT = {"none": "完整", "random": "随机删", "grid": "网格删", "tome": "ToMe", "kv_random": "旧KV",
         "kv_topk": "KV top9", "kv_randk": "KV 随机9", "attn_topk": "注意力删", "evit": "EViT", "tofu": "ToFu"}
LINE_STRATEGIES = ("random", "grid", "attn_topk", "tome", "tofu", "evit", "kv_random")
X = {0: 0.0, 131: 1.0, 196: 2.0, 261: 3.0, 383: 4.4}


def label(strategy: str, n: int) -> str:
    return METHODS[strategy][0] + (f" {SHARE[n]}" if n in SHARE else "")


def short(strategy: str, n: int) -> str:
    return SHORT[strategy] + (f" {SHARE[n]}" if n in SHARE else "")


def load(baseline: Path, runs: list[Path]) -> dict:
    episodes = {}
    paths = [(p, True) for p in glob.glob(str(baseline / "*/task*/trial*_frac*.json"))]
    for run in runs:
        paths += [(p, False) for p in glob.glob(str(run / "*/*/task*/trial*_rm*.json"))]
    for path, old in paths:
        meta = json.loads(Path(path).read_text())
        n = int(meta["masked_tokens"])
        if old:
            meta["strategy"] = "none" if n == 0 else "random"
        meta["npz"] = path[:-5] + ".npz"
        meta["video_glob"] = str(Path(path).parent / f"*episode={Path(path).stem}--*.mp4")
        episodes[(meta["strategy"], n, meta["suite"], meta["task_id"], meta["trial"])] = meta
    return episodes


def rows_of(episodes: dict, strategy: str, n: int) -> list[dict]:
    return [m for k, m in episodes.items() if k[:2] == (strategy, n)]


def reduced_calls(strategy: str, z) -> np.ndarray:
    """Calls whose inference was reduced; for K/V reuse the first call of an episode computes every token."""
    calls = len(z["t"])
    if strategy.startswith("kv"):
        return np.unpackbits(z["token_mask"], axis=1)[:, :IMAGE_TOKENS].any(axis=1)
    return np.ones(calls, dtype=bool)


def token_age(mask: np.ndarray) -> np.ndarray:
    """Calls since each token was last computed, per call [calls, 392]; 0 for tokens computed at that call."""
    age = np.zeros(mask.shape, dtype=np.int32)
    current = np.zeros(mask.shape[1], dtype=np.int32)
    for i, row in enumerate(mask):
        current = np.where(row, current + 1, 0)
        age[i] = current
    return age


def summarize(episodes: dict) -> dict:
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
            calls, by_index, ep_dev, ep_ok, ages = [], [], [], [], []
            for m in rows:
                z = np.load(m["npz"])
                dev = np.abs(z["action_masked"] - z["action_unmasked"]).mean(axis=(1, 2))
                reduced = reduced_calls(strategy, z)
                by_index.append(dev)
                if reduced.any():
                    calls.append(dev[reduced])
                    ep_dev.append(float(dev[reduced].mean()))
                    ep_ok.append(bool(m["success"]))
                if strategy.startswith("kv"):
                    mask = np.unpackbits(z["token_mask"], axis=1)[:, :IMAGE_TOKENS].astype(bool)
                    age = token_age(mask)
                    if mask.any():
                        ages.append(age[mask])
            dev = np.concatenate(calls)
            ep_dev, ep_ok = np.array(ep_dev), np.array(ep_ok)
            longest = max(len(d) for d in by_index)
            padded = np.full((len(by_index), longest), np.nan)
            for i, d in enumerate(by_index):
                padded[i, : len(d)] = d
            counts = np.isfinite(padded).sum(axis=0)
            entry["deviation"] = {
                "calls": int(len(dev)), "mean": float(dev.mean()), "median": float(np.median(dev)),
                "p10": float(np.percentile(dev, 10)), "p25": float(np.percentile(dev, 25)),
                "p75": float(np.percentile(dev, 75)), "p90": float(np.percentile(dev, 90)),
                "episode_mean_success": float(ep_dev[ep_ok].mean()) if ep_ok.any() else None,
                "episode_mean_failure": float(ep_dev[~ep_ok].mean()) if (~ep_ok).any() else None,
                "by_call_index": [float(v) for v in np.nanmean(padded[:, counts >= 20], axis=0)],
            }
            if ages:
                ages = np.concatenate(ages)
                entry["kv_age_calls"] = {"mean": float(ages.mean()), "median": float(np.median(ages)),
                                         "p90": float(np.percentile(ages, 90)), "max": int(ages.max())}
        summary[f"{strategy}:{n}"] = entry
    return summary


def fig_success(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    base = summary["none:0"]["success"]["all"]["rate"]
    labels = []
    for strategy in LINE_STRATEGIES + ("kv_topk", "kv_randk"):
        points = [(n, summary[f"{s}:{n}"]["success"]["all"]) for s, n in CONDITIONS
                  if s == strategy and n > 0 and f"{s}:{n}" in summary]
        if not points:
            continue
        if strategy in LINE_STRATEGIES and points[0][0] == 131:  # anchor at no reduction only from the first level
            points.insert(0, (0, summary["none:0"]["success"]["all"]))
        name, color = METHODS[strategy]
        xs = [X[n] + {"kv_topk": -0.15, "kv_randk": 0.15}.get(strategy, 0.0) for n, _ in points]
        ys = [r["rate"] for _, r in points]
        ax.errorbar(xs, ys, yerr=[r["se"] for _, r in points], color=color, linewidth=2.2, marker="o", markersize=7,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, capsize=3, elinewidth=1,
                    linestyle="-" if len(points) > 1 else "none")
        labels.append([xs[-1] + 0.1 if xs[-1] < 3.7 else 4.75, ys[-1], f"{name} {ys[-1]:.1f}%", color])
    ax.annotate(f"完整输入 {base:.0f}%", (0, base), textcoords="offset points", xytext=(0, 11), ha="center",
                fontsize=10, color=INK, fontweight="bold")
    for column in sorted({round(x, 1) for x, *_ in labels}):
        group = sorted([item for item in labels if round(item[0], 1) == column], key=lambda item: item[1])
        for i in range(1, len(group)):  # spread labels that would overlap
            group[i][1] = max(group[i][1], group[i - 1][1] + 4.2)
        for x, y, text, color in group:
            ax.text(x, y, text, color=color, va="center", fontsize=9.5)
    ax.axvspan(3.7, 6.3, color="#f4f3ef", zorder=0)
    ax.text(5.0, 0.985, "KV 复用：只重新计算 9 个 token", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=9, color=MUTED)
    ax.set_xticks([X[n] for n in (0, 131, 196, 261, 383)],
                  ["0\n(完整 392 个)", "131\n(1/3)", "196\n(1/2)", "261\n(2/3)", "383\n(只算 9 个)"])
    ax.set_xlabel("被去掉 / 合并掉 / 复用旧 KV 的图像 token 数")
    ax.set_xlim(-0.3, 6.3)
    lowest = min(item[1] for item in labels) if labels else 50
    ax.set_ylim(lowest - 8 if lowest >= 8 else -5, 106)
    ax.set_ylabel("成功率 (%)")
    ax.set_title("各种减少图像 token 的方法在 LIBERO 上的成功率（每点 200 回合，误差线 = 标准误）", loc="left",
                 fontsize=11, color=INK)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "r1_success_by_method.png", dpi=170)
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
        text = f"{value:.0f}{mark}" if c else f"{value:.1f}{mark}"
        detail = f"  −{cell['lost']}/+{cell['gained']}" if keys[r] != "none:0" else ""
        ax.text(c, r, text + detail, ha="center", va="center", fontsize=9,
                color="#fcfcfb" if value < 45 else INK)
    ax.set_xticks(range(len(columns)), [name for _, name in columns])
    ax.set_yticks(range(len(keys)), [summary[k]["label"] + ("" if k.startswith(("none", "kv_topk", "kv_randk"))
                                                          else f"（{summary[k]['tokens']} 个）") for k in keys])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(columns)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(keys)), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("成功率 (%)；−丢/+得 = 与完整输入同一任务、同一初始状态相比，由成功变失败 / 由失败变成功的回合数\n"
                 "* p<0.05，** p<0.01（配对 McNemar 检验）", loc="left", fontsize=9.5, color=INK2, pad=24)
    fig.tight_layout()
    fig.savefig(out / "r2_success_table.png", dpi=170)
    plt.close(fig)


def fig_tasks(episodes: dict, summary: dict, out: Path) -> None:
    keys = [(s, n) for s, n in CONDITIONS if f"{s}:{n}" in summary]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    for ax, (suite, name) in zip(axes.ravel(), SUITES):
        grid = np.full((10, len(keys)), np.nan)
        texts = []
        for task in range(10):
            for col, (s, n) in enumerate(keys):
                trials = [episodes.get((s, n, suite, task, trial)) for trial in range(5)]
                trials = [m for m in trials if m is not None]
                if trials:
                    grid[task, col] = sum(m["success"] for m in trials) / len(trials)
            text = episodes[("none", 0, suite, task, 0)]["task"]
            texts.append(text if len(text) <= 40 else text[:38] + "…")
        ax.imshow(1 - grid, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
        for (r, c), value in np.ndenumerate(grid):
            if np.isfinite(value):
                ax.text(c, r, f"{value * 5:.0f}", ha="center", va="center", fontsize=8,
                        color="#fcfcfb" if value < 0.45 else INK)
        ax.set_xticks(range(len(keys)), [short(s, n) for s, n in keys], rotation=55, ha="left", fontsize=8)
        ax.set_yticks(range(10), [f"{i}  {t}" for i, t in enumerate(texts)], fontsize=7.5)
        ax.set_title(name, loc="left", fontsize=11, color=INK, pad=4)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(keys)), minor=True)
        ax.set_yticks(np.arange(-0.5, 10), minor=True)
        ax.grid(which="minor", color="#fcfcfb", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
    fig.suptitle("每个任务的成功次数（满分 5；颜色越深失败越多）", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "r3_task_grid.png", dpi=150)
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
    fig.savefig(out / "r4_action_deviation.png", dpi=170)
    plt.close(fig)


def frame_at(meta: dict, t: int) -> np.ndarray:
    video = glob.glob(meta["video_glob"])[0]
    frame = imageio.get_reader(video).get_data(int(t) - WAIT)  # [256, 512, 3]: agent view | wrist view
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
    """Each token patch filled with the mean color of its ToMe group, with group borders."""
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


EXAMPLE = ("libero_goal", 3, 0)


def fig_methods(episodes: dict, out: Path) -> None:
    base = episodes[("none", 0) + EXAMPLE]
    z0 = np.load(base["npz"])
    frame = frame_at(base, z0["t"][0])  # the first call sees the same state in every condition
    panels = [("原始输入（14×28 = 392 个 token）", frame)]
    for strategy, n, title in (("random", 196, "随机删除 1/2（每次推理重新随机）"), ("grid", 196, "棋盘格删除 1/2"),
                               ("grid", 131, "网格删除 1/3：(行+列)%3==0"),
                               ("grid", 261, "网格删除 2/3：只留 (行+列)%3==0")):
        meta = episodes.get((strategy, n) + EXAMPLE)
        if meta is not None:
            mask = np.unpackbits(np.load(meta["npz"])["token_mask"][0])[:IMAGE_TOKENS].astype(bool)
            panels.append((f"{title}；该任务成功 {task_wins(episodes, strategy, n)}/5", dimmed(frame, mask)))
    for strategy, n, title in (("attn_topk", 196, "注意力 top-k 剪枝 1/2（暗块 = 删除）"),
                               ("evit", 196, "EViT 1/2（暗块 = 融合成 1 个 token）")):
        meta = episodes.get((strategy, n) + EXAMPLE)
        if meta is not None:
            mask = np.unpackbits(np.load(meta["npz"])["token_mask"][0])[:IMAGE_TOKENS].astype(bool)
            panels.append((f"{title}；成功 {task_wins(episodes, strategy, n)}/5", dimmed(frame, mask)))
    for strategy, name in (("tome", "ToMe"), ("tofu", "ToFu MLERP")):
        for n in (131, 196, 261):
            meta = episodes.get((strategy, n) + EXAMPLE)
            if meta is not None:
                group = np.load(meta["npz"])["merge_group"][0]
                panels.append((f"{name} 合并成 {IMAGE_TOKENS - n} 个（同组填平均色）；成功 {task_wins(episodes, strategy, n)}/5",
                               merged_view(frame, group)))
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
    fig.savefig(out / "r5_methods_on_a_frame.png", dpi=150)
    plt.close(fig)


def task_wins(episodes: dict, strategy: str, n: int) -> str:
    trials = [episodes.get((strategy, n, EXAMPLE[0], EXAMPLE[1], trial)) for trial in range(5)]
    return str(sum(m["success"] for m in trials if m is not None))


def canvas_view(meta: dict, z, call: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The current frame, the frame the K/V canvas encodes (each token from the call that last computed it) and the
    age of each token in calls."""
    mask = np.unpackbits(z["token_mask"], axis=1)[:, :IMAGE_TOKENS].astype(bool)
    age = token_age(mask)[call]
    frames = {}
    shown = np.zeros((224, 448, 3), dtype=np.uint8)
    for index in range(IMAGE_TOKENS):
        source = call - int(age[index])
        if source not in frames:
            frames[source] = frame_at(meta, z["t"][source])
        shown[patch(index)] = frames[source][patch(index)]
    return frames[call] if call in frames else frame_at(meta, z["t"][call]), shown, age


def fig_kv(episodes: dict, summary: dict, out: Path) -> None:
    examples = [("kv_topk", 383), ("kv_random", 261)]
    examples = [(s, n) for s, n in examples if (s, n) + EXAMPLE in episodes]
    fig = plt.figure(figsize=(13, 3.0 * len(examples) + 4.4))
    grid = fig.add_gridspec(len(examples) + 1, 2, height_ratios=[1] * len(examples) + [1.25])
    for row, (strategy, n) in enumerate(examples):
        meta = episodes[(strategy, n) + EXAMPLE]
        z = np.load(meta["npz"])
        call = min(len(z["t"]) - 1, 6)
        current, shown, age = canvas_view(meta, z, call)
        ax1, ax2 = fig.add_subplot(grid[row, 0]), fig.add_subplot(grid[row, 1])
        ax1.imshow(current)
        ax1.set_title(f"{label(strategy, n)}：第 {call + 1} 次推理时的真实画面（第 {z['t'][call]} 步）", fontsize=9.5,
                      loc="left", color=INK)
        ax2.imshow(shown)
        for index in np.nonzero(age == 0)[0]:
            r, c = divmod(int(index), GRID_W)
            ax2.add_patch(plt.Rectangle((c * 16 - 0.5, r * 16 - 0.5), 16, 16, fill=False, edgecolor="#2a78d6",
                                        linewidth=1.2 if strategy == "kv_topk" else 0.4))
        ax2.set_title(f"KV 画布实际编码的画面（蓝框 = 本次重新计算；其余 token 平均已 {age[age > 0].mean():.1f} 次推理未更新）",
                      fontsize=9.5, loc="left", color=INK)
        for ax in (ax1, ax2):
            ax.axis("off")
    ax3 = fig.add_subplot(grid[-1, 0])
    for strategy, n in (("random", 131), ("random", 261), ("kv_random", 131), ("kv_random", 261), ("kv_topk", 383),
                        ("kv_randk", 383)):
        entry = summary.get(f"{strategy}:{n}", {}).get("deviation")
        if entry is None:
            continue
        values = entry["by_call_index"]
        color = {("random", 131): "#9cc3ef", ("kv_random", 131): "#f5cf66"}.get((strategy, n), METHODS[strategy][1])
        ax3.plot(np.arange(1, len(values) + 1), values, color=color, linewidth=2,
                 linestyle="--" if strategy == "random" else "-", label=label(strategy, n))
    ax3.set_xlabel("回合内第几次推理（每 12 步一次）")
    ax3.set_ylabel("平均动作偏差")
    ax3.set_title("偏差随回合推进的变化（虚线 = 直接删除，作参照）", loc="left", fontsize=10, color=INK)
    ax3.legend(frameon=False, fontsize=8, ncol=2)
    ax4 = fig.add_subplot(grid[-1, 1])
    counts, total = np.zeros(IMAGE_TOKENS), 0
    for meta in rows_of(episodes, "kv_topk", 383):
        mask = np.unpackbits(np.load(meta["npz"])["token_mask"], axis=1)[:, :IMAGE_TOKENS].astype(bool)
        counts += (~mask[1:]).sum(axis=0)
        total += len(mask) - 1
    if total:
        image = ax4.imshow((counts / total).reshape(GRID_H, GRID_W), cmap="magma")
        ax4.axvline(GRID_W / 2 - 0.5, color="#fcfcfb", linewidth=1.5)
        ax4.set_xticks([GRID_W / 4 - 0.5, 3 * GRID_W / 4 - 0.5], ["第三人称相机", "腕部相机"])
        ax4.set_yticks([])
        ax4.tick_params(length=0)
        ax4.grid(False)
        fig.colorbar(image, ax=ax4, fraction=0.025, pad=0.01, label="被刷新的推理占比")
        ax4.set_title(f"top-9 刷新的 token 位置（{len(rows_of(episodes, 'kv_topk', 383))} 回合、各场景汇总）",
                      loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "r6_kv_reuse.png", dpi=150)
    plt.close(fig)


INDEX = """<!doctype html><meta charset="utf-8"><title>LIBERO token reduction comparison</title>
<style>body{{font-family:"Noto Sans CJK SC",sans-serif;background:#fcfcfb;color:#0b0b0b;max-width:1180px;margin:0 auto;
padding:24px 16px}}h1{{font-size:20px}}figure{{margin:0 0 36px}}img{{max-width:100%;border:1px solid #ecebe7}}
figcaption,p,li{{color:#52514e;font-size:14px;line-height:1.6}}</style>
<h1>LIBERO：减少图像 token 的几种方法对比（ImageWAM-FLUX.2-4B，OSMesa 渲染，每个条件 200 回合）</h1>
<ul>
<li>随机删除：每次推理重新随机选 token，在注意力中屏蔽（等价于从序列中拿掉）。</li>
<li>规整网格删除：固定的斜格 / 棋盘格图案，每次推理相同。</li>
<li>ToMe 合并：在 VAE latent 上做二分图相似度匹配，把相似 token 按大小加权平均合并，并在注意力上加 log(大小)。</li>
<li>复用旧 KV：被选中的 token 不重新计算，每层的 K/V 直接取该 token 上一次被计算时的值（每回合第一次推理全算）。
top-9 = 只重新计算上一次推理中动作注意力最高的 9 个 token；随机 9 = 随机选 9 个重新计算。</li>
</ul>
{figures}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--captions", type=Path, default=None, help="JSON {figure name: caption}")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    episodes = load(args.baseline, args.run)
    summary = summarize(episodes)
    (args.out / "summary_reduce.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    fig_success(summary, args.out)
    fig_table(summary, args.out)
    fig_tasks(episodes, summary, args.out)
    fig_deviation(summary, args.out)
    fig_methods(episodes, args.out)
    fig_kv(episodes, summary, args.out)
    captions = json.loads(args.captions.read_text()) if args.captions else {}
    names = ["r1_success_by_method.png", "r2_success_table.png", "r3_task_grid.png", "r4_action_deviation.png",
             "r5_methods_on_a_frame.png", "r6_kv_reuse.png"]
    figures = "\n".join(f'<figure><img src="{name}" alt="{name}"><figcaption>{captions.get(name, "")}</figcaption>'
                        "</figure>" for name in names)
    (args.out / "index.html").write_text(INDEX.format(figures=figures), encoding="utf-8")
    for key, entry in summary.items():
        s = entry["success"]["all"]
        d = entry.get("deviation", {})
        print(f"{key:14s} n={s['n']:3d} SR {s['rate']:5.1f} ±{s['se']:.1f}  −{s['lost']}/+{s['gained']} "
              f"p={s['mcnemar_p']:.2g}  dev median {d.get('median', 0):.3f}  "
              + " ".join(f"{name} {entry['success'][suite]['rate']:.0f}" for suite, name in SUITES
                         if suite in entry["success"]))


if __name__ == "__main__":
    main()
