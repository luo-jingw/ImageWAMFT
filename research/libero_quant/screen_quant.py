"""Open-loop screen of the quantization configs on saved LIBERO states (run in `.venv-libero`, see PLAN.md).

States: the simulator states saved by the no-reduction episodes of `runs/20261004_osmesa_grid` (token-mask study).
  - evaluation: tasks 0-4 of every suite, trial 0, calls 1, 3 and 6 (60 states);
  - SmoothQuant calibration: tasks 5-9 of every suite, trial 0, calls 0 and 4 (40 states, disjoint from evaluation).
Per config and state, against the full-precision inference on the same restored observation and noise:
  - mean absolute action deviation (normalized units, 16 × 7 values);
  - share of the 12 executed steps whose binarized gripper command differs;
  - inference seconds.
Also checks that the rotation alone (`fp_rot`) reproduces bfloat16 up to rounding.

  python screen_quant.py --out runs/screen.json [--configs rot_w4a4,...]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval_quant import E, build_policy, infer_actions
from fake_quant import CONFIGS, InputAbsmaxCollector, QuantSwitch, quantize_model, target_linears

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "libero_token_mask" / "runs" / "20261004_osmesa_grid"
EVAL_TASKS, EVAL_CALLS = range(5), (1, 3, 6)
CALIB_TASKS, CALIB_CALLS = range(5, 10), (0, 4)


def restored_inputs(policy: E.Policy, tasks: range, calls: tuple[int, ...]) -> list[tuple[str, E.Prepared]]:
    benchmark_dict = E.benchmark.get_benchmark_dict()
    inputs = []
    for suite in E.SUITES:
        task_suite = benchmark_dict[suite]()
        for task_id in tasks:
            z = np.load(BASELINE / suite / f"task{task_id:02d}" / "trial0_frac0000.npz")
            env, desc = E.get_libero_env(task_suite.get_task(task_id), E.LIBERO_ENV_RESOLUTION, policy.cfg.get("seed"))
            env.reset()
            for call in calls:
                if call < len(z["t"]):
                    obs = env.set_init_state(z["sim_state"][call])
                    inputs.append((f"{suite}/task{task_id:02d}/call{call}", policy.prepare(obs, desc)))
            env.close()
    return inputs


def gripper_flip(policy: E.Policy, action: np.ndarray, reference: np.ndarray) -> float:
    return float(np.mean(np.sign(policy.executable(action)[:12, -1]) != np.sign(policy.executable(reference)[:12, -1])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", type=str, default=",".join(CONFIGS))
    args = parser.parse_args()
    policy = build_policy()
    mot = policy.model.mot
    print(f"target linears: {len(target_linears(mot))}, "
          f"{sum(m.weight.numel() for m in target_linears(mot).values()) / 1e9:.2f} B weights", flush=True)
    evaluation = restored_inputs(policy, EVAL_TASKS, EVAL_CALLS)
    reference = {key: infer_actions(policy, prepared) for key, prepared in evaluation}
    calibration = restored_inputs(policy, CALIB_TASKS, CALIB_CALLS)
    collector = InputAbsmaxCollector(mot)
    for _, prepared in calibration:
        infer_actions(policy, prepared)
    collector.remove()
    report = {"states": len(evaluation), "calibration_states": len(calibration), "configs": {}}
    switch = QuantSwitch()
    for name in args.configs.split(","):
        start = time.perf_counter()
        handle = quantize_model(mot, CONFIGS[name], calibration_absmax=collector.absmax, switch=switch)
        quantize_seconds = time.perf_counter() - start
        rows = []
        for key, prepared in evaluation:
            action, seconds = infer_actions(policy, prepared)
            full, full_seconds = reference[key]
            rows.append({"state": key, "deviation": float(np.abs(action - full).mean()),
                         "max_abs": float(np.abs(action - full).max()), "gripper_flip": gripper_flip(policy, action, full),
                         "finite": bool(np.isfinite(action).all()), "seconds": seconds, "seconds_full": full_seconds})
        handle.remove()
        torch.cuda.empty_cache()
        dev = np.array([r["deviation"] for r in rows])
        report["configs"][name] = {
            "deviation_mean": float(dev.mean()), "deviation_median": float(np.median(dev)),
            "deviation_p90": float(np.percentile(dev, 90)), "max_abs": float(max(r["max_abs"] for r in rows)),
            "gripper_flip": float(np.mean([r["gripper_flip"] for r in rows])),
            "finite": all(r["finite"] for r in rows), "seconds_median": float(np.median([r["seconds"] for r in rows])),
            "seconds_full_median": float(np.median([r["seconds_full"] for r in rows])),
            "quantize_seconds": quantize_seconds, "rows": rows}
        c = report["configs"][name]
        print(f"{name:12s} deviation mean {c['deviation_mean']:.4f} median {c['deviation_median']:.4f} "
              f"p90 {c['deviation_p90']:.4f} max {c['max_abs']:.3f} | gripper flips {c['gripper_flip']:.3f} | "
              f"{c['seconds_median']:.2f} s vs {c['seconds_full_median']:.2f} s | quantize {quantize_seconds:.0f} s",
              flush=True)
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
