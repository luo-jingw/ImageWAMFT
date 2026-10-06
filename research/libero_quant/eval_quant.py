"""LIBERO rollouts of ImageWAM-FLUX.2-4B with simulated W/A quantization (run in `.venv-libero`, see PLAN.md).

Setup, preprocessing, episode loop and action post-processing follow `research/libero_token_mask/eval_token_mask.py`
(and through it the official `eval_libero_single.py`). The quantized model (`fake_quant.py`) produces the executed
actions; at every call a paired full-precision inference from the same observation and initial noise is recorded.

  python eval_quant.py --out-dir <dir> --conditions rot_w4a4,rot_w2a2 --worker-index i --num-workers N
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
sys.path.insert(0, str(HERE.parent / "libero_token_mask"))
import eval_token_mask as E  # noqa: E402

from fake_quant import CONFIGS, QuantSwitch, quantize_model  # noqa: E402


def build_policy() -> E.Policy:
    cfg = E.eval_config()
    E.set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    policy = E.Policy(cfg)
    policy.model.mot.action_attention_capture = None  # no attention recording: the fast SDPA path
    return policy


@torch.no_grad()
def infer_actions(policy: E.Policy, prepared: E.Prepared) -> tuple[np.ndarray, float]:
    """Normalized action chunk [16, 7] and wall seconds, with the official evaluation arguments."""
    cfg = policy.cfg
    torch.cuda.synchronize()
    start = time.perf_counter()
    pred = policy.model.infer_action(
        prompt=prepared.prompt, input_image=prepared.image, action_horizon=int(cfg.EVALUATION.action_horizon),
        negative_prompt=str(cfg.EVALUATION.get("negative_prompt", "")),
        text_cfg_scale=float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        num_inference_steps=int(cfg.EVALUATION.num_inference_steps), proprio=prepared.proprio, sigma_shift=None,
        seed=int(cfg.seed), rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")), tiled=False)
    torch.cuda.synchronize()
    return pred["action"].float().numpy(), time.perf_counter() - start


def run_episode(policy: E.Policy, switch: QuantSwitch, env, initial_state, task_description: str,
                suite: str) -> tuple[bool, list, dict]:
    cfg = policy.cfg
    max_steps = E.official._get_max_steps(suite)
    replan_steps = int(cfg.EVALUATION.replan_steps)
    num_steps_wait = int(cfg.EVALUATION.num_steps_wait)
    env.reset()
    obs = env.set_init_state(initial_state)
    replay, pending = [], []
    calls = {key: [] for key in ("t", "proprio_raw", "sim_state", "action_quant", "action_full", "executed",
                                 "seconds_quant", "seconds_full")}
    t, done = 0, False
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, done, _ = env.step(E.get_libero_dummy_action())
            t += 1
            continue
        if not pending:
            calls["sim_state"].append(np.asarray(env.get_sim_state(), dtype=np.float64).copy())
            prepared = policy.prepare(obs, task_description)
            switch.enabled = True
            action_quant, seconds_quant = infer_actions(policy, prepared)
            switch.enabled = False
            action_full, seconds_full = infer_actions(policy, prepared)
            switch.enabled = True
            executed = policy.executable(action_quant)
            calls["t"].append(t)
            calls["proprio_raw"].append(E.official._extract_sim_state(obs))
            calls["action_quant"].append(action_quant)
            calls["action_full"].append(action_full)
            calls["executed"].append(executed)
            calls["seconds_quant"].append(seconds_quant)
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
    parser.add_argument("--conditions", type=str, default="rot_w4a4,rot_w2a2", help=f"of {sorted(CONFIGS)}")
    parser.add_argument("--suites", type=str, default=",".join(E.SUITES))
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--trials", type=int, default=E.NUM_TRIALS)
    args = parser.parse_args()
    names = [n for n in args.conditions.split(",") if n]
    if any(n not in CONFIGS for n in names):
        raise ValueError(f"unknown quantization config in {names}")
    # Shards interleave conditions across tasks; within a worker, units run config by config so the weights are
    # quantized once per config.
    units = [(name, suite, task_id) for suite in args.suites.split(",") for task_id in range(10) for name in names]
    units = sorted(units[args.worker_index :: args.num_workers], key=lambda unit: names.index(unit[0]))
    policy = build_policy()
    benchmark_dict = E.benchmark.get_benchmark_dict()
    current, handle, switch = None, None, QuantSwitch()
    for name, suite, task_id in units:
        if name != current:
            if handle is not None:
                handle.remove()
                torch.cuda.empty_cache()
            handle, current = quantize_model(policy.model.mot, CONFIGS[name], switch=switch), name
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
            success, replay, calls = run_episode(policy, switch, env, initial_states[trial], task_description, suite)
            np.savez_compressed(unit_dir / f"trial{trial}.npz",
                                **{key: np.stack(value) for key, value in calls.items() if value})
            E.save_rollout_video(str(unit_dir), replay, f"trial{trial}", success=success,
                                 task_description=task_description)
            meta = {"config": name, "suite": suite, "task_id": task_id, "task": task_description, "trial": trial,
                    "initial_state_index": trial, "success": success, "inference_calls": len(calls["t"]),
                    "episode_steps": int(calls["episode_steps"][0]),
                    "seconds_quant_mean": float(np.mean(calls["seconds_quant"])),
                    "seconds_full_mean": float(np.mean(calls["seconds_full"])),
                    "wall_seconds": round(time.time() - start, 1), "worker": args.worker_index,
                    "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0)}
            json_path.write_text(json.dumps(meta, indent=1))
            print(json.dumps(meta), flush=True)
        env.close()


if __name__ == "__main__":
    main()
