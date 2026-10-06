"""LIBERO rollouts of ImageWAM-FLUX.2-4B with test-time registers (run in `.venv-libero`, see PLAN.md).

Setup, preprocessing, episode loop and action post-processing follow `research/libero_quant/eval_quant.py` (and
through it the token-mask evaluator and the official `eval_libero_single.py`). The model with registers produces the
executed actions; at every call a paired inference of the unmodified model from the same observation and initial noise
is recorded.

  python eval_register.py --out-dir <dir> --conditions ttr_dir_r16 --worker-index i --num-workers N
"""

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "libero_quant"))
from eval_quant import E, build_policy, infer_actions  # noqa: E402

import register as R  # noqa: E402


def run_episode(policy: E.Policy, handle: R.Registers, env, initial_state, task_description: str,
                suite: str) -> tuple[bool, list, dict]:
    cfg = policy.cfg
    max_steps = E.official._get_max_steps(suite)
    replan_steps = int(cfg.EVALUATION.replan_steps)
    num_steps_wait = int(cfg.EVALUATION.num_steps_wait)
    env.reset()
    obs = env.set_init_state(initial_state)
    replay, pending = [], []
    calls = {key: [] for key in ("t", "proprio_raw", "sim_state", "action_register", "action_full", "executed",
                                 "seconds_register", "seconds_full")}
    t, done = 0, False
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, done, _ = env.step(E.get_libero_dummy_action())
            t += 1
            continue
        if not pending:
            calls["sim_state"].append(np.asarray(env.get_sim_state(), dtype=np.float64).copy())
            prepared = policy.prepare(obs, task_description)
            handle.enabled = True
            action_register, seconds_register = infer_actions(policy, prepared)
            handle.enabled = False
            action_full, seconds_full = infer_actions(policy, prepared)
            handle.enabled = True
            executed = policy.executable(action_register)
            calls["t"].append(t)
            calls["proprio_raw"].append(E.official._extract_sim_state(obs))
            calls["action_register"].append(action_register)
            calls["action_full"].append(action_full)
            calls["executed"].append(executed)
            calls["seconds_register"].append(seconds_register)
            calls["seconds_full"].append(seconds_full)
            pending = executed[:replan_steps].tolist()
            replay.append(prepared.imgs.copy())
        else:
            replay.append(E.get_libero_image(obs).copy())
        obs, _, done, _ = env.step(pending.pop(0))
        if done:
            break
        t += 1
    calls["episode_steps"] = [t]
    return bool(done), replay, calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--conditions", type=str, default="ttr_dir_r16", help="register config names (register.py)")
    parser.add_argument("--suites", type=str, default=",".join(E.SUITES))
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--trials", type=int, default=E.NUM_TRIALS)
    args = parser.parse_args()
    grid = R.configs()
    names = [n for n in args.conditions.split(",") if n]
    if any(n not in grid for n in names):
        raise ValueError(f"unknown register config in {names}; known: {sorted(grid)}")
    units = [(name, suite, task_id) for suite in args.suites.split(",") for task_id in range(10) for name in names]
    units = sorted(units[args.worker_index :: args.num_workers], key=lambda unit: names.index(unit[0]))
    policy = build_policy()
    benchmark_dict = E.benchmark.get_benchmark_dict()
    current, handle = None, None
    for name, suite, task_id in units:
        if name != current:
            if handle is not None:
                handle.remove()
            handle, current = R.install(policy.model, grid[name]), name
        task_suite = benchmark_dict[suite]()
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = E.get_libero_env(task, E.LIBERO_ENV_RESOLUTION, policy.cfg.get("seed"))
        unit_dir = args.out_dir / name / suite / f"task{task_id:02d}"
        unit_dir.mkdir(parents=True, exist_ok=True)
        for trial in range(args.trials):
            json_path = unit_dir / f"trial{trial}.json"
            if json_path.exists():
                continue
            start = time.time()
            success, replay, calls = run_episode(policy, handle, env, initial_states[trial], task_description, suite)
            np.savez_compressed(unit_dir / f"trial{trial}.npz",
                                **{key: np.stack(value) for key, value in calls.items() if value})
            E.save_rollout_video(str(unit_dir), replay, f"trial{trial}", success=success,
                                 task_description=task_description)
            deviation = np.abs(np.stack(calls["action_register"]) - np.stack(calls["action_full"])).mean(axis=(1, 2))
            meta = {"config": name, "suite": suite, "task_id": task_id, "task": task_description, "trial": trial,
                    "initial_state_index": trial, "success": success, "inference_calls": len(calls["t"]),
                    "episode_steps": int(calls["episode_steps"][0]),
                    "paired_deviation_median": float(np.median(deviation)),
                    "seconds_register_mean": float(np.mean(calls["seconds_register"])),
                    "seconds_full_mean": float(np.mean(calls["seconds_full"])),
                    "wall_seconds": round(time.time() - start, 1), "worker": args.worker_index,
                    "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0)}
            json_path.write_text(json.dumps(meta, indent=1))
            print(json.dumps(meta), flush=True)
        env.close()


if __name__ == "__main__":
    main()
