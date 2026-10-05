"""Static figures for the LIBERO token-mask grid (run in `.venv-libero`); writes PNGs and a local index.html.

  python make_figures.py runs/20261004_osmesa_grid figures/
"""

import argparse
import glob
import json
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

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
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SUITES = [("libero_spatial", "Spatial"), ("libero_object", "Object"), ("libero_goal", "Goal"), ("libero_10", "Long")]
FRACTIONS = [(0.0, "0000", "0"), (1 / 3, "0333", "1/3"), (2 / 3, "0667", "2/3")]
TEXT_SLOTS, GRID_H, GRID_W, CAM_W, WAIT = 513, 14, 28, 14, 30


def load_episodes(run: Path) -> dict:
    episodes = {}
    for path in glob.glob(str(run / "*/task*/trial*_frac*.json")):
        meta = json.loads(Path(path).read_text())
        key = (meta["suite"], meta["task_id"], meta["trial"], f"{round(meta['mask_fraction'] * 1000):04d}")
        meta["npz"] = path[:-5] + ".npz"
        episodes[key] = meta
    return episodes


def fig_success(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = [0, 1, 2]
    labels = []
    for (suite, name), color in zip(SUITES, SERIES):
        rows = [summary["success"][f"{f:.3f}"][suite] for f, _, _ in FRACTIONS]
        ys = [r["success_rate"] for r in rows]
        ax.errorbar(xs, ys, yerr=[r["se"] for r in rows], color=color, linewidth=2, marker="o", markersize=7,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, capsize=3, elinewidth=1)
        labels.append([ys[-1], f"{name} {ys[-1]:.0f}%", color])
    rows = [summary["success"][f"{f:.3f}"]["all"] for f, _, _ in FRACTIONS]
    ys = [r["success_rate"] for r in rows]
    ax.errorbar(xs, ys, yerr=[r["se"] for r in rows], color=INK, linewidth=3, marker="o", markersize=9,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, capsize=3, elinewidth=1.2, zorder=5)
    ax.annotate("全部 100%", (0, ys[0]), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=10,
                color=INK, fontweight="bold")
    ax.annotate(f"全部 {ys[1]:.1f}%", (1, ys[1]), xytext=(1, 104), ha="center", fontsize=10, color=INK,
                fontweight="bold", arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 0.8})
    labels.append([ys[-1], f"全部 {ys[-1]:.1f}%", INK])
    labels.sort(key=lambda item: item[0])
    for i in range(1, len(labels)):  # spread labels that would overlap
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 3.2)
    for y, text, color in labels:
        ax.text(2.12, y, text, color=color if color != SERIES[3] else "#9a6a00", va="center", fontsize=10)
    ax.set_xticks(xs, ["0\n(完整 392 个 token)", "1/3\n(去掉 131 个)", "2/3\n(去掉 261 个)"])
    ax.set_xlim(-0.25, 2.7)
    ax.set_ylim(52, 108)
    ax.set_ylabel("成功率 (%)")
    ax.set_title("随机去掉图像 token 后的 LIBERO 成功率（每格 50 回合，误差线 = 标准误）", loc="left", fontsize=11,
                 color=INK)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig1_success_vs_mask.png", dpi=170)
    plt.close(fig)


def fig_task_grid(episodes: dict, out: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("fail", ["#f4f3ef", "#fbd9c8", "#f3a17a", "#eb6834", "#b84616", "#7a2c0b"])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    for ax, (suite, name) in zip(axes.ravel(), SUITES):
        grid = np.zeros((10, 3), dtype=int)
        tasks = []
        for task in range(10):
            for col, (_, tag, _) in enumerate(FRACTIONS):
                grid[task, col] = sum(episodes[(suite, task, trial, tag)]["success"] for trial in range(5))
            text = episodes[(suite, task, 0, "0000")]["task"]
            tasks.append(text if len(text) <= 46 else text[:44] + "…")
        failures = 5 - grid
        ax.imshow(failures, cmap=cmap, vmin=0, vmax=5, aspect="auto")
        for (r, c), value in np.ndenumerate(grid):
            ax.text(c, r, f"{value}/5", ha="center", va="center", fontsize=9,
                    color="#fcfcfb" if failures[r, c] >= 3 else INK)
        ax.set_xticks(range(3), [label for _, _, label in FRACTIONS])
        ax.set_yticks(range(10), [f"{i}  {t}" for i, t in enumerate(tasks)], fontsize=7.5)
        ax.set_title(f"{name}（成功次数 / 5）", loc="left", fontsize=11, color=INK)
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 3), minor=True)
        ax.set_yticks(np.arange(-0.5, 10), minor=True)
        ax.grid(which="minor", color="#fcfcfb", linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.xaxis.tick_top()
    fig.suptitle("每个任务的成功次数（列：被去掉的图像 token 比例；颜色越深失败越多）", x=0.01, ha="left", fontsize=12,
                 color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig2_task_grid.png", dpi=170)
    plt.close(fig)


def fig_deviation(episodes: dict, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
    bins = np.linspace(0, 0.4, 41)
    rng = np.random.default_rng(0)
    for (fraction, tag, label), color, xpos in ((FRACTIONS[1], SERIES[0], 0), (FRACTIONS[2], SERIES[1], 1)):
        calls, ep_dev, ep_ok = [], [], []
        for key, meta in episodes.items():
            if key[3] != tag:
                continue
            z = np.load(meta["npz"])
            dev = np.abs(z["action_masked"] - z["action_unmasked"]).mean(axis=(1, 2))
            calls.append(dev)
            ep_dev.append(dev.mean())
            ep_ok.append(meta["success"])
        calls = np.concatenate(calls)
        ax1.hist(np.clip(calls, 0, 0.399), bins=bins, histtype="stepfilled", alpha=0.25, color=color)  # >= 0.4 in last bin
        ax1.hist(np.clip(calls, 0, 0.399), bins=bins, histtype="step", linewidth=2, color=color,
                 label=f"去掉 {label}（{len(calls)} 次推理，中位数 {np.median(calls):.3f}）")
        ax1.axvline(np.median(calls), color=color, linestyle="--", linewidth=1.2)
        ep_dev, ep_ok = np.array(ep_dev), np.array(ep_ok)
        for offset, mask, marker_color in ((-0.18, ep_ok, color), (0.18, ~ep_ok, INK)):
            jitter = rng.uniform(-0.09, 0.09, mask.sum())
            ax2.scatter(xpos + offset + jitter, ep_dev[mask], s=12, color=marker_color, alpha=0.45, linewidths=0)
            ax2.plot([xpos + offset - 0.12, xpos + offset + 0.12], [ep_dev[mask].mean()] * 2, color=marker_color,
                     linewidth=3)
            ax2.annotate(f"均值 {ep_dev[mask].mean():.3f}\nn={mask.sum()}", (xpos + offset + 0.13, ep_dev[mask].mean()),
                         textcoords="offset points", xytext=(3, 0), ha="left", va="center", fontsize=8.5, color=INK2,
                         bbox={"boxstyle": "round,pad=0.15", "facecolor": "#fcfcfb", "edgecolor": "none", "alpha": 0.9})
    ax1.set_xlabel("动作偏差：同一观测下，去掉 token 与不去掉的动作差（归一化单位，16×7 平均）")
    ax1.set_ylabel("推理次数")
    ax1.legend(frameon=False, loc="upper right", fontsize=9)
    ax1.set_title("每次推理的动作偏差分布", loc="left", fontsize=11, color=INK)
    ax2.set_xticks([-0.18, 0.18, 0.82, 1.18], ["成功", "失败", "成功", "失败"])
    ax2.text(0, -0.14, "去掉 1/3", ha="center", transform=ax2.get_xaxis_transform(), color=SERIES[0], fontsize=10)
    ax2.text(1, -0.14, "去掉 2/3", ha="center", transform=ax2.get_xaxis_transform(), color=SERIES[1], fontsize=10)
    ax2.set_xlim(-0.5, 1.75)
    ax1.text(0.399, ax1.get_ylim()[1] * 0.12, "≥0.4\n合并", ha="right", fontsize=8, color=MUTED)
    ax2.set_ylabel("回合平均动作偏差")
    ax2.set_title("失败回合的偏差更大（横线 = 均值）", loc="left", fontsize=11, color=INK)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig3_action_deviation.png", dpi=170)
    plt.close(fig)


def fig_attention_layers(summary: dict, out: Path) -> None:
    shares = np.array(summary["attention"]["share_per_layer"])  # [layers, 5]
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
    top20 = np.array(summary["attention"]["image_top20_mass_per_layer"])
    ax2.plot(layers, top20, color=SERIES[0], linewidth=2, marker="o", markersize=5)
    ax2.axhline(20 / 392, color=MUTED, linestyle="--", linewidth=1)
    ax2.text(24.4, 20 / 392 + 0.02, "均匀分布时 = 5.1%", ha="right", fontsize=8.5, color=MUTED)
    ax2.set_ylim(0, 0.8)
    ax2.set_ylabel("前 20 token 占比")
    ax2.set_xlabel("动作专家层号（0–4 为 double block，5–24 为 single block）")
    ax2.set_xticks(layers)
    ax2.grid(axis="x", visible=False)
    ax1.set_title("动作专家每层注意力的去向（不去掉 token 时，200 回合平均）", loc="left", fontsize=11, color=INK,
                  pad=30)
    fig.tight_layout()
    fig.savefig(out / "fig4_attention_by_layer.png", dpi=170)
    plt.close(fig)


def frame_at(meta: dict, t: int) -> np.ndarray:
    video = next(Path(meta["npz"]).parent.glob(f"*episode=trial{meta['trial']}_frac*{meta['npz'][-8:-4]}*.mp4"))
    frame = imageio.get_reader(str(video)).get_data(int(t) - WAIT)  # [256, 512, 3]: agent view | wrist view
    halves = [Image.fromarray(frame[:, :256]).resize((224, 224), Image.BILINEAR),
              Image.fromarray(frame[:, 256:]).resize((224, 224), Image.BILINEAR)]
    return np.concatenate([np.asarray(h) for h in halves], axis=1)


def overlay(ax, frame: np.ndarray, heat: np.ndarray, title: str) -> None:
    heat = heat / max(heat.max(), 1e-12)
    up = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize((448, 224), Image.BILINEAR)) / 255.0
    ax.imshow(frame * 0.55 / 255.0 + 0.0)
    ax.imshow(up, cmap="inferno", alpha=np.clip(up * 1.1, 0, 0.85), vmin=0, vmax=1)
    ax.axvline(223.5, color="#fcfcfb", linewidth=1.5)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.axis("off")


EXAMPLES = [("libero_spatial", 0), ("libero_goal", 3), ("libero_10", 8)]
LAYER_GROUPS = [("早期层 0–4", slice(0, 5)), ("中间层 8–15", slice(8, 16)), ("后期层 16–24", slice(16, 25))]


def fig_overlays(episodes: dict, out: Path) -> None:
    fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(13, 2.45 * len(EXAMPLES) + 0.5))
    for row, (suite, task) in enumerate(EXAMPLES):
        meta = episodes[(suite, task, 0, "0000")]
        z = np.load(meta["npz"])
        call = int(len(z["t"]) * 0.45)
        frame = frame_at(meta, z["t"][call])
        image_att = z["attn_masked"].astype(np.float32)[call, :, TEXT_SLOTS:].reshape(-1, GRID_H, GRID_W)
        header = f"{dict(SUITES)[suite]} 任务 {task}（第 {z['t'][call]} 步）：{meta['task']}"
        for col, (label, layers) in enumerate(LAYER_GROUPS):
            overlay(axes[row, col], frame, image_att[layers].mean(axis=0), f"{header if col == 0 else ''}\n{label}")
    fig.suptitle("动作专家把注意力放在画面的哪里（左：第三人称相机，右：腕部相机；越亮注意力越高，每格单独归一化）",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.2, w_pad=0.6)
    fig.savefig(out / "fig5_attention_overlays.png", dpi=150)
    plt.close(fig)


def fig_mask_examples(episodes: dict, out: Path) -> None:
    suite, task = "libero_goal", 3
    base = episodes[(suite, task, 0, "0000")]
    z0 = np.load(base["npz"])
    frame = frame_at(base, z0["t"][0])
    middle = z0["attn_masked"].astype(np.float32)[0, 8:16, TEXT_SLOTS:].mean(axis=0).reshape(GRID_H, GRID_W)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.4))
    axes[0, 0].imshow(frame)
    axes[0, 0].set_title("原始输入（224×448，切成 14×28 = 392 个 token）", fontsize=9.5, loc="left", color=INK)
    overlay(axes[0, 1], frame, middle, "中间层（8–15）注意力：集中在腕部相机的夹爪附近")
    order = np.argsort(middle.ravel())[::-1]
    for ax, (fraction, tag, label) in zip(axes[1], FRACTIONS[1:]):
        meta = episodes[(suite, task, 0, tag)]
        z = np.load(meta["npz"])
        mask = np.unpackbits(z["token_mask"][0])[:392].astype(bool)  # first call: same state as the unmasked run
        shown = frame.copy().astype(np.float32)
        for index in np.nonzero(mask)[0]:
            r, c = divmod(index, GRID_W)
            shown[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] = shown[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] * 0.15 + 70
        top20_removed = int(mask[order[:20]].sum())
        ax.imshow(shown.clip(0, 255).astype(np.uint8))
        ok = sum(episodes[(suite, task, trial, tag)]["success"] for trial in range(5))
        ax.set_title(f"随机去掉 {label}（暗块）：注意力前 20 的 token 里去掉了 {top20_removed} 个；该任务成功 {ok}/5",
                     fontsize=9.5, loc="left", color=INK)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"去掉 token 的样子：{base['task']}", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig6_mask_examples.png", dpi=150)
    plt.close(fig)


INDEX = """<!doctype html><meta charset="utf-8"><title>LIBERO token mask figures</title>
<style>body{{font-family:"Noto Sans CJK SC",sans-serif;background:#fcfcfb;color:#0b0b0b;max-width:1180px;margin:0 auto;
padding:24px 16px}}h1{{font-size:20px}}figure{{margin:0 0 36px}}img{{max-width:100%;border:1px solid #ecebe7}}
figcaption{{color:#52514e;font-size:14px;margin-top:6px;line-height:1.6}}</style>
<h1>LIBERO 随机去掉图像 token（ImageWAM-FLUX.2-4B，600 回合，OSMesa 渲染）</h1>
{figures}"""
CAPTIONS = [
    ("fig1_success_vs_mask.png", "去掉 1/3：97.5%，与完整输入无显著差异（配对 p = 0.062）；去掉 2/3：74.5%（p = 9e-16），Goal 与 Long 掉得最多。"),
    ("fig2_task_grid.png", "逐任务看，掉点集中在需要找特定物体（开抽屉、放置到指定位置）和多子目标的长任务。"),
    ("fig3_action_deviation.png", "同一观测下去掉 token 前后的动作差；去掉 2/3 时偏差约是 1/3 的 2.5 倍，失败回合的偏差更大。"),
    ("fig4_attention_by_layer.png", "早期和后期层主要看文本，中间层主要看腕部相机；中间层里 392 个图像 token 中前 20 个就占了约一半图像注意力。"),
    ("fig5_attention_overlays.png", "注意力叠加在真实画面上：中间层集中在腕部相机里的夹爪和被抓物体。"),
    ("fig6_mask_examples.png", "随机 mask 不区分重要与否，去掉 2/3 时会把夹爪附近的关键 token 一起去掉。"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run / "analysis" / "summary.json").read_text())
    episodes = load_episodes(args.run)
    fig_success(summary, args.out)
    fig_task_grid(episodes, args.out)
    fig_deviation(episodes, args.out)
    fig_attention_layers(summary, args.out)
    fig_overlays(episodes, args.out)
    fig_mask_examples(episodes, args.out)
    figures = "\n".join(f'<figure><img src="{name}" alt="{name}"><figcaption>{text}</figcaption></figure>'
                        for name, text in CAPTIONS)
    (args.out / "index.html").write_text(INDEX.format(figures=figures), encoding="utf-8")
    print("written:", sorted(p.name for p in args.out.iterdir()))


if __name__ == "__main__":
    main()
