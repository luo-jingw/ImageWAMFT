"""Reads the RoboTwin evaluation runs of several models and writes the comparison tables (plan.md E7).

  .venv/bin/python launch/robotwin_eval_compare.py \
      --result flux2_4b=ImageWAM/evaluate_results/robotwin/model/flux2_4b_release_full \
      --result qwenimage21=ImageWAM/evaluate_results/robotwin/step_065000/qwenimage21_step065000_full \
      --reference launch/robotwin_table12_flux2_4b.csv --output <file.md>

A model may list several run directories separated by commas; each task must then come from exactly one of them.
Inputs per run directory (one manager run): `<task>/_result_{clean,random}.txt` and
`<task>/episode_end_states.jsonl`. The clean and randomized phases of a task share that file; a record's phase is the
`randomized-true|false` part of its video file name. Directories renamed by a resume (`<task>.incomplete_<k>`) are
skipped.

Checks, printed: every task of a model comes from exactly one run directory; the episode records agree with the
result file; per pair of models and phase, how many seeds both ran and how many of those had the same instruction.
The evaluation process never seeds Python `random`, which draws the instructions and is used by cuRobo's graph
planner, so neither the instructions nor the seeds that pass the expert check are fixed by the seed
(docs/robotwin_training_setup.md -> Evaluation Protocol).

Statistics:
- A model's mean is the mean of its per-task success rates; its standard error is sqrt(sum_t p_t (1 - p_t) / n_t) / T.
- The difference between two models is the difference of their means, with standard error sqrt(se_a^2 + se_b^2).
- The paired difference uses only the episodes with the same (task, phase, seed). Per task it is the mean of the
  per-episode differences; the standard error is sqrt(sum_t var_t / n_t) / T over the T tasks.
- Against the reference (100 trials per task), each task gets a two-proportion z with the pooled rate.
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

PHASES = ("clean", "random")


@dataclass(frozen=True)
class EpisodeRecord:
    task: str
    phase: str
    episode_idx: int
    seed: int
    success: bool
    instruction: str
    take_action_cnt: int


@dataclass
class TaskResult:
    task: str
    phase: str
    success_rate: float
    episodes: list[EpisodeRecord]
    run_dir: Path


@dataclass
class ModelResults:
    name: str
    results: dict[tuple[str, str], TaskResult]


def read_rate(path: Path) -> float:
    values = []
    for line in path.read_text().splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    if not values:
        raise ValueError(f"no success rate in {path}")
    return values[-1]


def record_phase(record: dict) -> str:
    video = record.get("video_path") or ""
    if "randomized-true" in video:
        return "random"
    if "randomized-false" in video:
        return "clean"
    raise ValueError(f"cannot tell the phase of episode {record.get('episode_idx')} from video_path={video!r}")


def read_task_dir(task_dir: Path) -> list[TaskResult]:
    records: dict[str, list[EpisodeRecord]] = {phase: [] for phase in PHASES}
    states = task_dir / "episode_end_states.jsonl"
    if states.exists():
        for line in states.read_text().splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            phase = record_phase(raw)
            instruction = raw.get("instruction")
            records[phase].append(
                EpisodeRecord(
                    task=task_dir.name,
                    phase=phase,
                    episode_idx=int(raw["episode_idx"]),
                    seed=int(raw["seed"]),
                    success=bool(raw["success"]),
                    instruction=instruction if isinstance(instruction, str) else json.dumps(instruction),
                    take_action_cnt=int(raw["take_action_cnt"]),
                )
            )
    results = []
    for phase in PHASES:
        result_file = task_dir / f"_result_{phase}.txt"
        if result_file.exists():
            results.append(TaskResult(task_dir.name, phase, read_rate(result_file), records[phase], task_dir.parent))
    return results


def load_model(name: str, run_dirs: list[Path]) -> ModelResults:
    results: dict[tuple[str, str], TaskResult] = {}
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        for task_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and ".incomplete_" not in p.name):
            for result in read_task_dir(task_dir):
                key = (result.task, result.phase)
                if key in results:
                    raise ValueError(f"{name}: {key} in both {results[key].run_dir} and {run_dir}")
                results[key] = result
    return ModelResults(name, results)


def read_reference(path: Path) -> dict[tuple[str, str], float]:
    reference = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            for phase in PHASES:
                reference[(row["task"], phase)] = float(row[phase]) / 100.0
    return reference


def check_model(model: ModelResults) -> None:
    for (task, phase), result in sorted(model.results.items()):
        n = len(result.episodes)
        rate = sum(e.success for e in result.episodes) / n if n else float("nan")
        seeds = [e.seed for e in result.episodes]
        problems = []
        if n == 0 or abs(rate - result.success_rate) > 1e-9:
            problems.append(f"records give {rate:.4f} over {n} episodes, result file {result.success_rate:.4f}")
        if len(set(seeds)) != n:
            problems.append("repeated seeds")
        if problems:
            print(f"CHECK {model.name} {task} {phase}: " + "; ".join(problems))


def check_pairs(models: list[ModelResults]) -> None:
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            for phase in PHASES:
                keys = sorted(k for k in set(a.results) & set(b.results) if k[1] == phase)
                if not keys:
                    continue
                episodes = shared = same_instruction = same_lists = 0
                for key in keys:
                    seeds_a = [e.seed for e in a.results[key].episodes]
                    inst_b = {e.seed: e.instruction for e in b.results[key].episodes}
                    episodes += len(seeds_a)
                    shared += sum(1 for s in seeds_a if s in inst_b)
                    same_instruction += sum(1 for e in a.results[key].episodes if inst_b.get(e.seed) == e.instruction)
                    same_lists += int(seeds_a == [e.seed for e in b.results[key].episodes])
                print(f"pair {a.name} / {b.name} {phase}: {len(keys)} tasks, identical seed lists in {same_lists}; "
                      f"{shared} of {episodes} seeds shared, {same_instruction} of those with the same instruction")


def model_mean(model: ModelResults, tasks: list[str], phase: str) -> tuple[float, float, int]:
    rates, variance = [], 0.0
    for task in tasks:
        result = model.results[(task, phase)]
        n = len(result.episodes)
        rates.append(result.success_rate)
        variance += result.success_rate * (1.0 - result.success_rate) / n
    return sum(rates) / len(rates), math.sqrt(variance) / len(rates), len(rates)


def paired_difference(a: ModelResults, b: ModelResults, tasks: list[str], phase: str) -> tuple[float, float, int]:
    diffs, variance, pairs = [], 0.0, 0
    for task in tasks:
        success_b = {e.seed: e.success for e in b.results[(task, phase)].episodes}
        d = [float(e.success) - float(success_b[e.seed]) for e in a.results[(task, phase)].episodes if e.seed in success_b]
        if not d:
            continue
        mean = sum(d) / len(d)
        var = sum((x - mean) ** 2 for x in d) / (len(d) - 1) if len(d) > 1 else 0.0
        diffs.append(mean)
        variance += var / len(d)
        pairs += len(d)
    return sum(diffs) / len(diffs), math.sqrt(variance) / len(diffs), pairs


def pooled_z(p: float, n: int, r: float, m: int) -> float:
    pooled = (p * n + r * m) / (n + m)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n + 1.0 / m))
    return 0.0 if se == 0.0 else (p - r) / se


def complete_tasks(models: list[ModelResults], phase: str) -> list[str]:
    tasks = None
    for model in models:
        present = {task for (task, ph) in model.results if ph == phase}
        tasks = present if tasks is None else tasks & present
    return sorted(tasks or [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", action="append", required=True, help="<name>=<run dir>[,<run dir>...]")
    parser.add_argument("--reference", type=Path, help="CSV with task,clean,random in percent (100 trials per task)")
    parser.add_argument("--reference-trials", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    models = []
    for spec in args.result:
        name, dirs = spec.split("=", 1)
        models.append(load_model(name, [Path(d) for d in dirs.split(",") if d]))
    reference = read_reference(args.reference) if args.reference else {}

    for model in models:
        per_phase = {phase: sum(1 for (_, ph) in model.results if ph == phase) for phase in PHASES}
        runs = sorted({str(r.run_dir) for r in model.results.values()})
        print(f"{model.name}: tasks clean {per_phase['clean']}, random {per_phase['random']}; {len(runs)} run directories")
        check_model(model)
    check_pairs(models)

    lines = []
    for phase in PHASES:
        tasks = complete_tasks(models, phase)
        if not tasks:
            continue
        has_ref = all((t, phase) in reference for t in tasks)
        header = ["task"] + (["Table 12"] if has_ref else []) + [m.name for m in models]
        lines += [f"### {phase} ({len(tasks)} tasks)", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for task in tasks:
            cells = [task] + ([f"{100 * reference[(task, phase)]:.0f}"] if has_ref else [])
            cells += [f"{100 * m.results[(task, phase)].success_rate:.0f}" for m in models]
            lines.append("| " + " | ".join(cells) + " |")
        mean_cells = ["mean"] + ([f"{100 * sum(reference[(t, phase)] for t in tasks) / len(tasks):.2f}"] if has_ref else [])
        for model in models:
            mean, se, _ = model_mean(model, tasks, phase)
            mean_cells.append(f"{100 * mean:.2f} ± {100 * se:.2f}")
        lines += ["| " + " | ".join(mean_cells) + " |", ""]

        if len(models) > 1:
            lines += ["| pair | difference of means (points) | paired, shared seeds (points) | shared episodes |",
                      "|---|---|---|---|"]
            for i, a in enumerate(models):
                for b in models[i + 1 :]:
                    mean_a, se_a, _ = model_mean(a, tasks, phase)
                    mean_b, se_b, _ = model_mean(b, tasks, phase)
                    diff, se, pairs = paired_difference(b, a, tasks, phase)
                    lines.append(f"| {b.name} − {a.name} | {100 * (mean_b - mean_a):+.2f} ± "
                                 f"{100 * math.sqrt(se_a ** 2 + se_b ** 2):.2f} | {100 * diff:+.2f} ± {100 * se:.2f} | {pairs} |")
            lines.append("")

        if has_ref:
            for model in models:
                far = []
                for task in tasks:
                    result = model.results[(task, phase)]
                    z = pooled_z(result.success_rate, len(result.episodes), reference[(task, phase)], args.reference_trials)
                    if abs(z) > 3.0:
                        far.append(f"{task} ({100 * result.success_rate:.0f} vs {100 * reference[(task, phase)]:.0f}, z {z:+.1f})")
                below = sum(1 for t in tasks if model.results[(t, phase)].success_rate < reference[(t, phase)])
                above = sum(1 for t in tasks if model.results[(t, phase)].success_rate > reference[(t, phase)])
                lines.append(f"{model.name} against Table 12: {below} tasks below, {above} above; |z| > 3: "
                             + (", ".join(far) if far else "none"))
            lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
