"""Figures of the LIBERO quantization study (run in `.venv-libero`); deletes `figures/` and rebuilds it.

  01_success.png          success rate per suite: bfloat16, W4A4 and W2A2 (rotation), paired McNemar tests
  02_open_loop_screen.png open-loop screen of all configs: action deviation and gripper sign flips
  03_deviation.png        closed-loop paired action deviation (quantized vs. bfloat16 on the same observation)
  04_task_grid.png        successes per task
  summary.json, index.html

  python make_figures.py
"""

import argparse
import glob
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SUITES = [("libero_spatial", "Spatial"), ("libero_object", "Object"), ("libero_goal", "Goal"), ("libero_10", "Long")]
CONDITIONS = [("bf16", "bfloat16（不量化）", "#52514e"), ("rot_w4a4", "W4A4（旋转）", "#2a78d6"),
              ("rot_w2a2", "W2A2（旋转）", "#eb6834")]
# Paired action deviation of token removal (median, same metric), from research/libero_token_mask/RESULTS.md
REFERENCES = [(0.032, "随机删 1/3 token（成功率 97.5%）"), (0.075, "随机删 2/3 token（74.5%）")]
SCREEN_GROUPS = {"fp_rot": "只旋转（全精度）", "rtn_w8a8": "W8A8 RTN", "rot_w8a8": "W8A8 旋转",
                 "rot_w4a16": "W4A16 旋转（只量化权重）", "rot_w16a4": "W16A4 旋转（只量化激活）",
                 "rtn_w4a4": "W4A4 RTN", "sq_w4a4": "W4A4 SmoothQuant", "rot_w4a4": "W4A4 旋转",
                 "sqrot_w4a4": "W4A4 SmoothQuant + 旋转", "rot_w3a3": "W3A3 旋转", "rot_w2a16": "W2A16 旋转（只量化权重）",
                 "rot_w16a2": "W16A2 旋转（只量化激活）", "rtn_w2a2": "W2A2 RTN", "rot_w2a2": "W2A2 旋转",
                 "sqrot_w2a2": "W2A2 SmoothQuant + 旋转"}


def load(run: Path, baseline: Path) -> dict:
    episodes = {}
    for path in glob.glob(str(baseline / "*/task*/trial*_frac0000.json")):
        meta = json.loads(Path(path).read_text())
        episodes[("bf16", meta["suite"], meta["task_id"], meta["trial"])] = meta | {"npz": None}
    for path in glob.glob(str(run / "*/*/task*/trial*.json")):
        meta = json.loads(Path(path).read_text())
        episodes[(meta["config"], meta["suite"], meta["task_id"], meta["trial"])] = meta | {"npz": path[:-5] + ".npz"}
    return episodes


def summarize(episodes: dict) -> dict:
    base = {k[1:]: m["success"] for k, m in episodes.items() if k[0] == "bf16"}
    summary = {}
    for name, label, _ in CONDITIONS:
        rows = [m for k, m in episodes.items() if k[0] == name]
        entry = {"label": label, "success": {}}
        for suite in ["all"] + [s for s, _ in SUITES]:
            subset = rows if suite == "all" else [m for m in rows if m["suite"] == suite]
            wins = sum(m["success"] for m in subset)
            pairs = [(base[(m["suite"], m["task_id"], m["trial"])], m["success"]) for m in subset]
            lost = sum(b and not s for b, s in pairs)
            gained = sum(s and not b for b, s in pairs)
            p = binomtest(min(lost, gained), lost + gained, 0.5).pvalue if lost + gained else 1.0
            rate = wins / len(subset)
            entry["success"][suite] = {"n": len(subset), "wins": wins, "rate": round(100 * rate, 1),
                                       "se": round(100 * np.sqrt(rate * (1 - rate) / len(subset)), 1), "lost": lost,
                                       "gained": gained, "mcnemar_p": float(p)}
        if name != "bf16":
            dev = np.concatenate([np.abs(np.load(m["npz"])["action_quant"] - np.load(m["npz"])["action_full"])
                                  .mean(axis=(1, 2)) for m in rows])
            entry["deviation"] = {"calls": int(len(dev)), "median": float(np.median(dev)), "mean": float(dev.mean()),
                                  "p10": float(np.percentile(dev, 10)), "p90": float(np.percentile(dev, 90)),
                                  "values": dev}
            entry["seconds_quant_median"] = float(np.median([m["seconds_quant_mean"] for m in rows]))
            entry["seconds_full_median"] = float(np.median([m["seconds_full_mean"] for m in rows]))
            hosts = {}
            for m in rows:
                key = m.get("gpu", "NVIDIA H100 NVL")
                hosts[key] = hosts.get(key, 0) + 1
            entry["gpus"] = hosts
        summary[name] = entry
    return summary


def fig_success(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    columns = [("all", "全部")] + SUITES
    width = 0.26
    for i, (name, label, color) in enumerate(CONDITIONS):
        rates = [summary[name]["success"][c]["rate"] for c, _ in columns]
        bars = ax.bar(np.arange(len(columns)) + (i - 1) * width, rates, width, color=color, label=label)
        for bar, (c, _) in zip(bars, columns):
            cell = summary[name]["success"][c]
            text = f"{cell['rate']:.0f}" if c != "all" else f"{cell['rate']:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, text, ha="center", fontsize=8.5,
                    color=INK2)
    ax.set_xticks(np.arange(len(columns)), [name for _, name in columns])
    ax.set_ylim(0, 112)
    ax.set_ylabel("成功率 (%)")
    w4 = summary["rot_w4a4"]["success"]["all"]
    ax.set_title(f"量化后的 LIBERO 成功率（每组 200 回合）。W4A4 与 bfloat16 配对：掉 {w4['lost']} 个、多 {w4['gained']} 个，"
                 f"McNemar p = {w4['mcnemar_p']:.3f}；W2A2 全部失败", loc="left", fontsize=10, color=INK, pad=28)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "01_success.png", dpi=170)
    plt.close(fig)


def fig_screen(screen: dict, out: Path) -> None:
    names = [n for n in SCREEN_GROUPS if n in screen["configs"]]
    dev = [screen["configs"][n]["deviation_mean"] for n in names]
    flips = [100 * screen["configs"][n]["gripper_flip"] for n in names]
    bit_colors = {16: "#52514e", 8: "#1baf7a", 4: "#2a78d6", 3: "#7b5ea7", 2: "#eb6834"}
    colors = [bit_colors[min(int(b) for b in re.search(r"w(\d+)a(\d+)", n).groups()) if "_w" in n else 16]
              for n in names]  # colored by the lower of the weight and activation bit widths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.4 * len(names) + 1.8), sharey=True,
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(names))[::-1]
    ax1.barh(y, dev, color=colors)
    ax1.set_xscale("log")
    for value, yy in zip(dev, y):
        ax1.text(value * 1.08, yy, f"{value:.4f}", va="center", fontsize=8, color=INK2)
    for value, text in REFERENCES:
        ax1.axvline(value, color=MUTED, linestyle="--", linewidth=1)
        ax1.text(value * 1.05, len(names) - 0.3, text, fontsize=7.5, color=MUTED, rotation=90, va="top")
    ax1.set_yticks(y, [SCREEN_GROUPS[n] for n in names])
    ax1.set_xlabel("与 bfloat16 的动作偏差（均值，归一化单位，对数坐标）")
    ax1.set_title(f"开环筛选：{screen['states']} 个保存的仿真状态，同一观测和初始噪声", loc="left", fontsize=10, color=INK)
    ax2.barh(y, flips, color=colors)
    for value, yy in zip(flips, y):
        ax2.text(value + 0.6, yy, f"{value:.1f}%", va="center", fontsize=8, color=INK2)
    ax2.set_xlabel("12 步执行动作中夹爪开合指令翻转的比例 (%)")
    ax2.set_xlim(0, max(flips) * 1.25 + 1)
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "02_open_loop_screen.png", dpi=170)
    plt.close(fig)


def fig_deviation(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    bins = np.logspace(-3, 0, 60)
    for name, label, color in CONDITIONS[1:]:
        d = summary[name]["deviation"]
        ax.hist(np.clip(d["values"], 1e-3, 0.999), bins=bins, color=color, alpha=0.35, histtype="stepfilled")
        ax.hist(np.clip(d["values"], 1e-3, 0.999), bins=bins, color=color, histtype="step", linewidth=1.8,
                label=f"{label}：中位数 {d['median']:.3f}（{d['calls']} 次推理）")
    for value, text in REFERENCES:
        ax.axvline(value, color=MUTED, linestyle="--", linewidth=1)
        ax.text(value * 1.05, ax.get_ylim()[1] * 0.95, text, fontsize=8, color=MUTED, rotation=90, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("闭环中每次推理的动作偏差：量化 vs. 同一观测下的 bfloat16（归一化单位，16×7 平均）")
    ax.set_ylabel("推理次数")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("闭环成对动作偏差（虚线：删除 token 时的中位偏差，作参照）", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "03_deviation.png", dpi=170)
    plt.close(fig)


def fig_tasks(episodes: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.2))
    for ax, (suite, name) in zip(axes, SUITES):
        grid = np.array([[sum(episodes[(c, suite, t, trial)]["success"] for trial in range(5))
                          for c, _, _ in CONDITIONS] for t in range(10)])
        ax.imshow(5 - grid, cmap="Oranges", vmin=0, vmax=5, aspect="auto")
        for (r, c), v in np.ndenumerate(grid):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=9, color="#fcfcfb" if v < 3 else INK)
        ax.set_xticks(range(len(CONDITIONS)), ["bf16", "W4A4", "W2A2"])
        ax.set_yticks(range(10), [str(t) for t in range(10)])
        ax.set_title(name, loc="left", fontsize=11)
        ax.grid(False)
        ax.xaxis.tick_top()
    fig.suptitle("每个任务的成功次数（满分 5，行 = 任务编号）", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "04_task_grid.png", dpi=160)
    plt.close(fig)


INDEX = """<!doctype html><meta charset="utf-8"><title>LIBERO quantization</title>
<style>body{{font-family:"Noto Sans CJK SC",sans-serif;background:#fcfcfb;color:#0b0b0b;max-width:1180px;margin:0 auto;
padding:24px 16px}}h1{{font-size:20px}}figure{{margin:0 0 36px}}img{{max-width:100%;border:1px solid #ecebe7}}
figcaption,li{{color:#52514e;font-size:14px;line-height:1.6}}</style>
<h1>LIBERO：ImageWAM-FLUX.2-4B 的极限量化（模拟量化，OSMesa 渲染，每组 200 回合）</h1>
<ul><li>量化范围：两个专家 transformer block 内的 140 个线性层（4.31 B 权重）；嵌入、调制、输出层、注意力矩阵乘、VAE、文本编码器保持 bfloat16。</li>
<li>旋转：每个线性层输入乘块对角随机 Hadamard（1024 一块），权重乘同一矩阵；权重按输出通道对称量化（逐行搜索截断比例），激活按 token 动态对称量化。</li></ul>
{figures}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=HERE / "runs" / "20261005_quant_grid")
    parser.add_argument("--screen", type=Path, default=HERE / "runs" / "screen.json")
    parser.add_argument("--baseline", type=Path,
                        default=HERE.parent / "libero_token_mask" / "runs" / "20261004_osmesa_grid")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)
    episodes = load(args.run, args.baseline)
    summary = summarize(episodes)
    screen = json.loads(args.screen.read_text())
    fig_success(summary, args.out)
    fig_screen(screen, args.out)
    fig_deviation(summary, args.out)
    fig_tasks(episodes, args.out)
    for entry in summary.values():
        if "deviation" in entry:
            entry["deviation"].pop("values")
    screen_summary = {name: {k: v for k, v in c.items() if k != "rows"} for name, c in screen["configs"].items()}
    (args.out / "summary.json").write_text(json.dumps({"closed_loop": summary, "open_loop_screen": screen_summary},
                                                      indent=1, ensure_ascii=False))
    w4 = summary["rot_w4a4"]
    captions = {
        "01_success.png": f"W4A4（旋转）{w4['success']['all']['rate']:.1f}%，与 bfloat16 无显著差异；W2A2 全部失败。",
        "02_open_loop_screen.png": "W4A4 的误差主要来自激活（W16A4 0.020 vs. W4A16 0.0075）；旋转把 W4A4 从 RTN 的 0.18 降到 0.022。"
                                   "2 bit 激活单独就使夹爪指令翻转约 29%，2 bit 权重单独只有 0.048。",
        "03_deviation.png": "闭环中 W4A4 的偏差比随机删 1/3 token 还小；W2A2 的偏差大于删 2/3 token。",
        "04_task_grid.png": "W4A4 只在 Spatial 的 3 个任务和 Goal 的 1 个任务上各失败 1 次。",
    }
    figures = "\n".join(f'<figure><img src="{n}" alt="{n}"><figcaption>{c}</figcaption></figure>'
                        for n, c in captions.items())
    (args.out / "index.html").write_text(INDEX.format(figures=figures), encoding="utf-8")
    for name, entry in summary.items():
        s = entry["success"]["all"]
        print(name, f"SR {s['rate']} lost {s['lost']} gained {s['gained']} p {s['mcnemar_p']:.3g}",
              entry.get("deviation", ""), entry.get("gpus", ""))
    print("written:", sorted(p.name for p in args.out.iterdir()))


if __name__ == "__main__":
    main()
