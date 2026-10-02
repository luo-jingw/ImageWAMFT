"""Mirror offline wandb runs into TensorBoard event files, following the runs while they train (run in `.venv`).

Reads the `run-*.wandb` record log of each run directory (no wandb login, no network) and writes every history
scalar (`train/*`, `eval/*`, `performance/*`) at its `_step`, plus the validation videos in `<run>/eval/`
(the single frame of `step_XXXXXX_rank_YYY.mp4`, stacked top to bottom: prediction, VAE reconstruction, ground
truth) as images.
The output directory `<logdir>/<name>/` is owned by this script and is rebuilt from scratch at start-up.

  python scripts/wandb_offline_to_tensorboard.py --logdir <dir> --run flux4b=<run dir> --run qwenimage21=<run dir>
"""

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

SKIP_KEYS = {"_step", "_timestamp", "_runtime"}
EVAL_VIDEO = re.compile(r"step_(\d+)_rank_(\d+)\.mp4$")


def read_history(wandb_file: str) -> list[dict]:
    """All history records of a (possibly still growing) wandb record log."""
    store = datastore.DataStore()
    store.open_for_scan(wandb_file)
    rows = []
    while True:
        try:
            data = store.scan_data()
        except Exception:  # partial block at the end of a live file
            break
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") == "history":
            rows.append({item.key or "/".join(item.nested_key): json.loads(item.value_json) for item in record.history.item})
    return rows


def first_frame(path: str) -> np.ndarray | None:
    try:
        from torchcodec.decoders import VideoDecoder

        return VideoDecoder(path)[0].numpy()  # [3, H, W] uint8
    except Exception:
        return None


class RunMirror:
    def __init__(self, name: str, run_dir: Path, logdir: Path) -> None:
        self.name = name
        self.run_dir = run_dir
        out = logdir / name
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("events.out.tfevents.*"):
            old.unlink()
        self.writer = SummaryWriter(str(out))
        self.last_step = -1
        self.videos_done: set[str] = set()

    def update(self) -> int:
        written = 0
        files = sorted(glob.glob(str(self.run_dir / "wandb" / "offline-run-*" / "run-*.wandb")), key=os.path.getmtime)
        for wandb_file in files:
            for row in read_history(wandb_file):
                step = row.get("_step")
                if step is None or step <= self.last_step:
                    continue
                walltime = row.get("_timestamp")
                for key, value in row.items():
                    if key in SKIP_KEYS or isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    self.writer.add_scalar(key, value, global_step=step, walltime=walltime)
                self.last_step = step
                written += 1
        for video in sorted(glob.glob(str(self.run_dir / "eval" / "step_*_rank_*.mp4"))):
            match = EVAL_VIDEO.search(video)
            if match is None or video in self.videos_done or time.time() - os.path.getmtime(video) < 30:
                continue
            frame = first_frame(video)
            if frame is not None:
                tag = f"eval_image/rank_{int(match.group(2))}_top-pred_middle-vae_bottom-gt"
                self.writer.add_image(tag, frame, global_step=int(match.group(1)))
            self.videos_done.add(video)
        self.writer.flush()
        return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="name=run directory (repeatable)")
    parser.add_argument("--interval", type=float, default=120.0, help="seconds between refreshes; 0 = once")
    args = parser.parse_args()
    os.environ.setdefault("WANDB_MODE", "offline")
    mirrors = []
    for spec in args.run:
        name, run_dir = spec.split("=", 1)
        mirrors.append(RunMirror(name, Path(run_dir), args.logdir))
    while True:
        for mirror in mirrors:
            written = mirror.update()
            print(f"{time.strftime('%F %T')} {mirror.name}: +{written} steps (last step {mirror.last_step})", flush=True)
        if args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
