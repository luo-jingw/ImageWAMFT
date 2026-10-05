"""Open-loop dose test: reusing stale K/V vs. removing the same tokens (run in `.venv-libero`).

On saved states of no-reduction episodes (runs/20261004_osmesa_grid), restores the simulator at calls i-1 and i of an
episode. The K/V canvas is computed from the observation of call i-1; at call i the same random token subsets
(nested, n tokens) are either reused from that canvas or removed. Reported against the full inference at call i:
mean absolute action deviation and the share of the 12 executed steps whose binarized gripper command differs.
The restored call-i full inference must equal the action the episode recorded (checks the state restore).

  python kv_dose.py --baseline runs/20261004_osmesa_grid --out runs/20261004_reduce_grid/kv_dose.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_token_mask as E  # noqa: E402
from libero.libero import benchmark  # noqa: E402

DOSES = (0, 13, 33, 65, 131, 196, 261, 383)
CALLS = (1, 3, 6)  # call i; the canvas comes from call i - 1


def gripper_flip(policy, action, reference) -> float:
    return float(np.mean(np.sign(policy.executable(action)[:12, -1]) != np.sign(policy.executable(reference)[:12, -1])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=5, help="tasks 0..n-1 of every suite, trial 0")
    args = parser.parse_args()
    cfg = E.eval_config()
    E.set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    policy = E.Policy(cfg)
    benchmark_dict = benchmark.get_benchmark_dict()
    rows = []
    for suite in E.SUITES:
        task_suite = benchmark_dict[suite]()
        for task_id in range(args.tasks):
            z = np.load(args.baseline / suite / f"task{task_id:02d}" / "trial0_frac0000.npz")
            env, desc = E.get_libero_env(task_suite.get_task(task_id), E.LIBERO_ENV_RESOLUTION, cfg.get("seed"))
            env.reset()
            for call in CALLS:
                if call >= len(z["t"]):
                    continue
                previous_obs = env.set_init_state(z["sim_state"][call - 1])
                current_obs = env.set_init_state(z["sim_state"][call])
                policy.canvas.reset()
                policy.act(policy.prepare(previous_obs, desc), E.Reduction(kv_reuse=True), paired=False)
                canvas = ([k.clone() for k in policy.canvas.k], [v.clone() for v in policy.canvas.v])
                current = policy.prepare(current_obs, desc)
                full = policy.act(current, E.Reduction(), paired=False)["masked"]["action_norm"]
                restore_error = float(np.abs(full - z["action_masked"][call]).max())
                order = np.random.default_rng([E.SUITES.index(suite), task_id, call]).permutation(E.IMAGE_TOKENS)
                for dose in DOSES:
                    subset = torch.from_numpy(np.sort(order[:dose])).long()
                    results = {}
                    for mode in ("stale", "removed"):
                        if mode == "stale":
                            policy.canvas.k, policy.canvas.v = [k.clone() for k in canvas[0]], [v.clone() for v in canvas[1]]
                            reduction = E.Reduction(kv_reuse=True, stale=subset if dose else None)
                        else:
                            reduction = E.Reduction(drop=subset if dose else None)
                        action = policy.act(current, reduction, paired=False)["masked"]["action_norm"]
                        results[mode] = (float(np.abs(action - full).mean()), gripper_flip(policy, action, full))
                    rows.append({"suite": suite, "task_id": task_id, "call": call, "dose": dose,
                                 "restore_error": restore_error,
                                 "stale_deviation": results["stale"][0], "stale_gripper_flip": results["stale"][1],
                                 "removed_deviation": results["removed"][0],
                                 "removed_gripper_flip": results["removed"][1]})
                print(json.dumps(rows[-1]), flush=True)
            env.close()
    summary = {}
    for dose in DOSES:
        subset = [r for r in rows if r["dose"] == dose]
        summary[dose] = {key: float(np.mean([r[key] for r in subset])) for key in
                         ("stale_deviation", "removed_deviation", "stale_gripper_flip", "removed_gripper_flip")}
        summary[dose]["n"] = len(subset)
    args.out.write_text(json.dumps({"rows": rows, "summary": summary,
                                    "max_restore_error": max(r["restore_error"] for r in rows)}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
