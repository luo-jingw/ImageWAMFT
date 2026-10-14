"""Convert a Trainer state directory (DeepSpeed ZeRO-1/2) for resuming on a different number of processes (run in `.venv`).

A ZeRO-1/2 state stores the optimizer state in one shard per data-parallel rank, so it only loads on the same rank
count. This script writes a new state directory that loads on any rank count and any optimizer group split:

- `pytorch_model/`: DeepSpeed universal checkpoint (`deepspeed.checkpoint.ds_to_universal`): per-parameter fp32 weights,
  `exp_avg` and `exp_avg_sq`, loaded by DeepSpeed when the run's DeepSpeed config sets `checkpoint.load_universal`
  (`scripts/ds_configs/ds_zero2_load_universal_config.json`).
- `pytorch_model/zero/optimizer_state.pt`: DeepSpeed restores optimizer groups by zipping the saved `param_groups`
  with the new run's groups. The source run has one group; the new run splits parameters into several groups with
  identical hyperparameters (`optimizer_max_group_numel`). The single saved group is repeated once per trainable
  parameter, an upper bound on the group count, so every new group is restored.
- `trainer_state.json`: `batch_in_epoch` counts per-process batches, so it is scaled by old/new process count to keep
  the same sample offset (`ResumableEpochSampler`: offset = batch_in_epoch x batch size x processes).
- `scheduler.bin`, `random_states_*.pkl`: copied unchanged.

  python scripts/reshard_training_state.py --state-dir <run>/checkpoints/state/step_045000 \
      --output-dir <run>/checkpoints/state_universal/step_045000 --num-processes 4
"""

import argparse
import glob
import json
import os
import shutil
import time
from dataclasses import dataclass

import torch
from deepspeed.checkpoint import ds_to_universal

MODEL_TAG = "pytorch_model"


@dataclass
class TrainerProgress:
    global_step: int
    epoch: int
    batch_in_epoch: int


def read_progress(state_dir: str) -> TrainerProgress:
    with open(os.path.join(state_dir, "trainer_state.json")) as f:
        payload = json.load(f)
    return TrainerProgress(int(payload["global_step"]), int(payload["epoch"]), int(payload["batch_in_epoch"]))


def write_progress(state_dir: str, progress: TrainerProgress) -> None:
    with open(os.path.join(state_dir, "trainer_state.json"), "w") as f:
        json.dump(
            {"global_step": progress.global_step, "epoch": progress.epoch, "batch_in_epoch": progress.batch_in_epoch},
            f,
            indent=2,
        )


def count_source_ranks(model_dir: str) -> int:
    shards = glob.glob(os.path.join(model_dir, "*zero_pp_rank_*_mp_rank_00_optim_states.pt"))
    if not shards:
        raise FileNotFoundError(f"no ZeRO optimizer shards in {model_dir}")
    return len(shards)


def repeat_param_groups(zero_dir: str) -> tuple[int, int]:
    """Repeat the single saved optimizer group once per converted parameter; returns (saved groups, written groups)."""
    path = os.path.join(zero_dir, "optimizer_state.pt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    groups = state["param_groups"]
    if len(groups) != 1:
        raise ValueError(f"expected 1 saved optimizer group, found {len(groups)}")
    num_params = sum(1 for entry in os.scandir(zero_dir) if entry.is_dir())
    state["param_groups"] = [dict(groups[0]) for _ in range(num_params)]
    torch.save(state, path)
    return len(groups), num_params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-processes", type=int, required=True, help="process count of the resuming run")
    parser.add_argument("--extract-workers", type=int, default=4)
    parser.add_argument("--merge-workers", type=int, default=4)
    args = parser.parse_args()

    source_model = os.path.join(args.state_dir, MODEL_TAG)
    target_model = os.path.join(args.output_dir, MODEL_TAG)
    if os.path.exists(args.output_dir):
        raise FileExistsError(args.output_dir)
    old_ranks = count_source_ranks(source_model)
    progress = read_progress(args.state_dir)
    if (progress.batch_in_epoch * old_ranks) % args.num_processes:
        raise ValueError(f"batch_in_epoch {progress.batch_in_epoch} x {old_ranks} is not divisible by {args.num_processes}")
    new_batch_in_epoch = progress.batch_in_epoch * old_ranks // args.num_processes

    start = time.time()
    os.makedirs(args.output_dir)
    ds_to_universal.main(
        argparse.Namespace(
            input_folder=source_model,
            output_folder=target_model,
            num_extract_workers=args.extract_workers,
            num_merge_workers=args.merge_workers,
            keep_temp_folder=False,
            strict=True,
            inject_missing_state=True,
        )
    )
    convert_s = time.time() - start
    saved_groups, written_groups = repeat_param_groups(os.path.join(target_model, "zero"))

    for name in ["scheduler.bin", *sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.state_dir, "random_states_*.pkl")))]:
        shutil.copy2(os.path.join(args.state_dir, name), os.path.join(args.output_dir, name))
    with open(os.path.join(args.output_dir, "latest"), "w") as f:
        f.write(MODEL_TAG)
    write_progress(args.output_dir, TrainerProgress(progress.global_step, progress.epoch, new_batch_in_epoch))

    zero_dir = os.path.join(target_model, "zero")
    param_dirs = [e.path for e in os.scandir(zero_dir) if e.is_dir()]
    incomplete = [p for p in param_dirs if not all(os.path.isfile(os.path.join(p, f"{k}.pt")) for k in ("fp32", "exp_avg", "exp_avg_sq"))]
    size_gb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(args.output_dir) for f in fs) / 1e9
    print(
        json.dumps(
            {
                "global_step": progress.global_step,
                "source_ranks": old_ranks,
                "target_processes": args.num_processes,
                "batch_in_epoch": [progress.batch_in_epoch, new_batch_in_epoch],
                "batch_in_epoch_x_processes": [progress.batch_in_epoch * old_ranks, new_batch_in_epoch * args.num_processes],
                "param_dirs": len(param_dirs),
                "param_dirs_missing_state": len(incomplete),
                "param_groups": [saved_groups, written_groups],
                "output_gb": round(size_gb, 1),
                "convert_s": round(convert_s, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
