"""Figures of the LIBERO test-time register study (run in `.venv-libero`); deletes `figures/` and rebuilds it.

  01_sink_shift.png       per-layer attention on the template and on the registers (image and action queries)
  02_open_loop_screen.png open-loop screen of all configs: action deviation and where the image queries' sink goes
  03_mechanism.png        register neurons (cumulative share of the massive update) and token norms per config
  04_success.png          closed-loop success rate per suite, paired McNemar tests against the unmodified model
  05_deviation.png        closed-loop paired action deviation (registers vs. unmodified on the same observation)
  06_task_grid.png        successes per task
  summary.json, index.html

  python make_figures.py
"""

import argparse
import glob
import json
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
CONDITIONS = [("none", "原模型", "#52514e"), ("ttr_n99_r16", "register：移神经元（99%，16 个）", "#1baf7a"),
              ("ttr_dir_r16", "register：移方向（16 个）", "#2a78d6"), ("ablate_dir", "删大激活方向，不加 register", "#eb6834")]
# Paired action deviation of token removal (median, same metric), from research/libero_token_mask/RESULTS.md
REFERENCES = [(0.032, "随机删 1/3 token（成功率 97.5%）"), (0.075, "随机删 2/3 token（74.5%）")]


def screen_label(name: str) -> str:
    if name == "none":
        return "原模型（重复一次）"
    if name.startswith("tokens_r"):
        return f"只加 {name[8:]} 个零 token"
    if name == "ablate_dir":
        return "删大激活方向"
    if name.startswith("ablate_n"):
        return f"删神经元（{name[8:]}%）"
    if name.startswith("ttr_dir_r"):
        return f"移方向 → {name[9:]} 个 register"
    share, r = name[5:].split("_r")
    return f"移神经元（{share}%）→ {r} 个 register"


def screen_color(name: str) -> str:
    if name == "none" or name.startswith("tokens"):
        return "#8a8984"
    if name.startswith("ablate"):
        return "#eb6834"
    if name.startswith("ttr_dir"):
        return "#2a78d6"
    return {"50": "#9fd8c0", "90": "#4cc398", "99": "#1baf7a"}[name[5:7]]


def load(run: Path, baseline: Path) -> dict:
    episodes = {}
    for path in glob.glob(str(baseline / "*/task*/trial*_frac0000.json")):
        meta = json.loads(Path(path).read_text())
        episodes[("none", meta["suite"], meta["task_id"], meta["trial"])] = meta | {"npz": None}
    for path in glob.glob(str(run / "*/*/task*/trial*.json")):
        meta = json.loads(Path(path).read_text())
        episodes[(meta["config"], meta["suite"], meta["task_id"], meta["trial"])] = meta | {"npz": path[:-5] + ".npz"}
    return episodes


def summarize(episodes: dict) -> dict:
    base = {k[1:]: m["success"] for k, m in episodes.items() if k[0] == "none"}
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
        if name != "none":
            dev = np.concatenate([np.abs(z["action_register"] - z["action_full"]).mean(axis=(1, 2))
                                  for z in (np.load(m["npz"]) for m in rows)])
            entry["deviation"] = {"calls": int(len(dev)), "median": float(np.median(dev)), "mean": float(dev.mean()),
                                  "p10": float(np.percentile(dev, 10)), "p90": float(np.percentile(dev, 90)),
                                  "values": dev}
            entry["seconds_register_median"] = float(np.median([m["seconds_register_mean"] for m in rows]))
            entry["seconds_full_median"] = float(np.median([m["seconds_full_mean"] for m in rows]))
        summary[name] = entry
    return summary


def fig_sink_shift(screen: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    layers = np.arange(25)
    for ax, key, title in zip(axes, ("image_query_attention", "action_query_attention"),
                              ("参考帧图像 token（query）的注意力", "动作 token（query，第 1 步去噪）的注意力")):
        for name, label, color in CONDITIONS:
            att = screen["configs"][name][key]
            ax.plot(layers, 100 * np.array(att["template"]), color=color, linewidth=1.8, label=f"{label}：模板")
            if name.startswith("ttr"):
                ax.plot(layers, 100 * np.array(att["register"]), color=color, linewidth=1.8, linestyle="--",
                        label=f"{label}：register")
        ax.axvspan(-0.5, 4.5, color="#ecebe7", alpha=0.5, linewidth=0)
        ax.text(2, 96, "双流层", ha="center", fontsize=8.5, color=MUTED)
        ax.set_xlim(-0.5, 24.5)
        ax.set_ylim(0, 100)
        ax.set_xlabel("层")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("注意力份额 (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"sink 从提示模板转移到 register：实线 = 落在模板 18 个 token 上，虚线 = 落在 register 上"
                 f"（{screen['states']} 个保存状态的平均）", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "01_sink_shift.png", dpi=170)
    plt.close(fig)


def fig_screen(screen: dict, out: Path) -> None:
    names = list(screen["configs"])
    configs = screen["configs"]
    dev = [configs[n]["deviation_mean"] for n in names]
    colors = [screen_color(n) for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 0.36 * len(names) + 1.9), sharey=True,
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(names))[::-1]
    ax1.barh(y, np.maximum(dev, 1e-4), color=colors)
    ax1.set_xscale("log")
    ax1.set_xlim(1e-4, 0.2)
    for value, yy in zip(dev, y):
        ax1.text(max(value, 1e-4) * 1.08, yy, f"{value:.4f}", va="center", fontsize=8, color=INK2)
    for (value, text), style in zip(REFERENCES, ("--", ":")):
        ax1.axvline(value, color=MUTED, linestyle=style, linewidth=1.2, label=f"{text}：{value}")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    ax1.set_yticks(y, [screen_label(n) for n in names])
    ax1.set_xlabel("与原模型的动作偏差（均值，归一化单位，对数坐标）")
    ax1.set_title(f"开环筛选：{screen['states']} 个保存的仿真状态，同一观测和初始噪声", loc="left", fontsize=10, color=INK)
    parts = [("template", "模板", "#52514e"), ("register", "register", "#2a78d6"), ("task", "任务词", "#c9c7bf")]
    left = np.zeros(len(names))
    for key, label, color in parts:
        share = np.array([100 * np.mean(configs[n]["image_query_attention"][key][5:11]) for n in names])
        ax2.barh(y, share, left=left, color=color, label=label)
        left += share
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("第 5–10 层图像 query 的注意力份额 (%)")
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right")
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "02_open_loop_screen.png", dpi=170)
    plt.close(fig)


def fig_mechanism(neurons: dict, screen: dict, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.2]})
    cumulative = np.array(neurons["cumulative_share"])
    k = np.arange(1, len(cumulative) + 1)
    ax1.plot(k, 100 * cumulative, color="#1baf7a", linewidth=2)
    for share, count in neurons["neurons_for_share"].items():
        if share in ("0.5", "0.9", "0.99"):
            ax1.axhline(100 * float(share), color="#ecebe7", linewidth=1)
            ax1.text(1.1, 100 * float(share) + 1, f"{float(share):.0%}：{count} 个神经元", fontsize=8.5, color=INK2)
    ax1.set_xscale("log")
    ax1.set_xlim(1, len(cumulative))
    ax1.set_ylim(0, 105)
    ax1.set_xlabel("按贡献排序的第 4 层文本 MLP 神经元个数（共 9216）")
    ax1.set_ylabel("模板 token 大激活更新的累计份额 (%)")
    channels = ", ".join(f"{c}" for c in list(neurons["direction_top_channels"])[:4])
    ax1.set_title(f"register 神经元不稀疏；大激活方向 {neurons['direction_share_top4']:.0%} 在通道 {channels}",
                  loc="left", fontsize=10, color=INK)
    groups = [("template", "模板"), ("task", "任务词"), ("register", "register"), ("image", "图像")]
    shown = [c for c in CONDITIONS]
    width = 0.2
    for i, (name, label, color) in enumerate(shown):
        norms = screen["configs"][name]["first_single_norm"]
        values = [norms.get(g, np.nan) for g, _ in groups]
        ax2.bar(np.arange(len(groups)) + (i - 1.5) * width, values, width, color=color, label=label)
    ax2.set_yscale("log")
    ax2.set_xticks(np.arange(len(groups)), [label for _, label in groups])
    ax2.set_ylabel("进入第 1 个单流层时的残差范数（中位数）")
    ax2.set_title("大激活从模板 token 移到 register", loc="left", fontsize=10, color=INK)
    ax2.legend(frameon=False, fontsize=8.5)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "03_mechanism.png", dpi=170)
    plt.close(fig)


def fig_success(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    columns = [("all", "全部")] + SUITES
    width = 0.2
    for i, (name, label, color) in enumerate(CONDITIONS):
        rates = [summary[name]["success"][c]["rate"] for c, _ in columns]
        cell = summary[name]["success"]["all"]
        if name != "none":
            label = f"{label}（掉 {cell['lost']}、多 {cell['gained']}，p = {cell['mcnemar_p']:.2g}）"
        bars = ax.bar(np.arange(len(columns)) + (i - 1.5) * width, rates, width, color=color, label=label)
        for bar, (c, _) in zip(bars, columns):
            cell = summary[name]["success"][c]
            text = f"{cell['rate']:.0f}" if c != "all" else f"{cell['rate']:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, text, ha="center", fontsize=8,
                    color=INK2)
    ax.set_xticks(np.arange(len(columns)), [name for _, name in columns])
    ax.set_ylim(0, 112)
    ax.set_ylabel("成功率 (%)")
    ax.set_title("闭环成功率（每组 200 回合；括号内为与原模型逐回合配对的 McNemar 检验）", loc="left", fontsize=10,
                 color=INK, pad=44)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "04_success.png", dpi=170)
    plt.close(fig)


def fig_deviation(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.0))
    bins = np.logspace(-3, 0, 60)
    for name, label, color in CONDITIONS[1:]:
        d = summary[name]["deviation"]
        values = np.clip(d["values"], 1e-3, 0.999)
        ax.hist(values, bins=bins, color=color, alpha=0.3, histtype="stepfilled")
        ax.hist(values, bins=bins, color=color, histtype="step", linewidth=1.8,
                label=f"{label}：中位数 {d['median']:.3f}（{d['calls']} 次推理）")
    for (value, text), style in zip(REFERENCES, ("--", ":")):
        ax.axvline(value, color=MUTED, linestyle=style, linewidth=1.2, label=f"{text}：{value}")
    ax.set_xscale("log")
    ax.set_xlabel("闭环中每次推理的动作偏差：register vs. 同一观测下的原模型（归一化单位，16×7 平均）")
    ax.set_ylabel("推理次数")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.set_title("闭环成对动作偏差（虚线：删除 token 时的中位偏差，作参照）", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "05_deviation.png", dpi=170)
    plt.close(fig)


def fig_tasks(episodes: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2))
    for ax, (suite, name) in zip(axes, SUITES):
        grid = np.array([[sum(episodes[(c, suite, t, trial)]["success"] for trial in range(5))
                          for c, _, _ in CONDITIONS] for t in range(10)])
        ax.imshow(5 - grid, cmap="Oranges", vmin=0, vmax=5, aspect="auto")
        for (r, c), v in np.ndenumerate(grid):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=9, color="#fcfcfb" if v < 3 else INK)
        ax.set_xticks(range(len(CONDITIONS)), ["原模型", "移神经元", "移方向", "删方向"], fontsize=8.5)
        ax.set_yticks(range(10), [str(t) for t in range(10)])
        ax.set_title(name, loc="left", fontsize=11)
        ax.grid(False)
        ax.xaxis.tick_top()
    fig.suptitle("每个任务的成功次数（满分 5，行 = 任务编号）", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "06_task_grid.png", dpi=160)
    plt.close(fig)


INDEX = """<!doctype html><meta charset="utf-8"><title>LIBERO test-time registers</title>
<style>body{{font-family:"Noto Sans CJK SC",sans-serif;background:#fcfcfb;color:#0b0b0b;max-width:1180px;margin:0 auto;
padding:24px 16px}}h1{{font-size:20px}}figure{{margin:0 0 36px}}img{{max-width:100%;border:1px solid #ecebe7}}
figcaption,li{{color:#52514e;font-size:14px;line-height:1.6}}</style>
<h1>LIBERO：ImageWAM-FLUX.2-4B 的训练无关 register（test-time register，OSMesa 渲染，每组 200 回合）</h1>
<ul><li>sink 位置：图像和动作 token 把注意力停在提示模板的 18 个 token 上；这些 token 在第 4 层（最后一个双流层）文本 MLP 被写入大激活（4 个通道）。</li>
<li>register：在 proprio 之后追加零初始化 token；把第 4 层的大激活（神经元或方向）从所有文本 token 移到 register 上。不训练、不改权重。</li></ul>
{figures}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=HERE / "runs" / "20261005_register_grid")
    parser.add_argument("--screen", type=Path, default=HERE / "runs" / "screen.json")
    parser.add_argument("--neurons", type=Path, default=HERE / "runs" / "register_neurons.json")
    parser.add_argument("--baseline", type=Path,
                        default=HERE.parent / "libero_token_mask" / "runs" / "20261004_osmesa_grid")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)
    episodes = load(args.run, args.baseline)
    summary = summarize(episodes)
    screen = json.loads(args.screen.read_text())
    neurons = json.loads(args.neurons.read_text())
    fig_sink_shift(screen, args.out)
    fig_screen(screen, args.out)
    fig_mechanism(neurons, screen, args.out)
    fig_success(summary, args.out)
    fig_deviation(summary, args.out)
    fig_tasks(episodes, args.out)
    for entry in summary.values():
        if "deviation" in entry:
            entry["deviation"].pop("values")
    screen_summary = {name: {k: v for k, v in c.items() if k != "rows"} for name, c in screen["configs"].items()}
    neuron_summary = {k: v for k, v in neurons.items() if k not in ("ranked_neurons", "cumulative_share")}
    (args.out / "summary.json").write_text(json.dumps(
        {"closed_loop": summary, "open_loop_screen": screen_summary, "register_neurons": neuron_summary},
        indent=1, ensure_ascii=False))
    s = {n: summary[n]["success"]["all"] for n, _, _ in CONDITIONS}
    d = {n: summary[n]["deviation"]["median"] for n, _, _ in CONDITIONS[1:]}
    captions = {
        "01_sink_shift.png": "register 接走了图像 token 落在模板上的注意力；动作 token 也有一部分转到 register 上。",
        "02_open_loop_screen.png": "只加零 token 几乎不改变输出；移走 90% 以上的大激活时，加 register 的偏差都小于不加，且 register 越多偏差越小。",
        "03_mechanism.png": "大激活在第 4 层由上百个神经元共同写入，集中在 4 个通道；移动后模板 token 范数降到任务词水平，register 成为最大范数 token。",
        "04_success.png": "；".join(f"{summary[n]['label']} {s[n]['rate']:.1f}%" for n, _, _ in CONDITIONS) + "。",
        "05_deviation.png": "闭环成对偏差中位数：" + "；".join(f"{summary[n]['label']} {d[n]:.3f}" for n in d) + "。",
        "06_task_grid.png": "每个任务 5 个初始状态的成功次数。",
    }
    figures = "\n".join(f'<figure><img src="{n}" alt="{n}"><figcaption>{c}</figcaption></figure>'
                        for n, c in captions.items())
    (args.out / "index.html").write_text(INDEX.format(figures=figures), encoding="utf-8")
    for name, entry in summary.items():
        a = entry["success"]["all"]
        print(name, f"SR {a['rate']} lost {a['lost']} gained {a['gained']} p {a['mcnemar_p']:.3g}",
              {k: round(v, 4) for k, v in entry.get("deviation", {}).items() if k != "calls"})
    print("written:", sorted(p.name for p in args.out.iterdir()))


if __name__ == "__main__":
    main()
