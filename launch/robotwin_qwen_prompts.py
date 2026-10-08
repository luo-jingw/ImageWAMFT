"""Writes the RoboTwin training prompts as JSON shards for the Qwen-Image-2.1 text cache (plan.md Phase 3).

Each prompt is `DEFAULT_PROMPT.format(task=task)` for every task string of `meta/tasks.jsonl`, exactly as
`RobotVideoDataset` builds it (no stripping), so the cache file name `sha256(prompt)` matches the training lookup.
Output: <out-dir>/prompts_<i>_of_<n>.json with {"prompts": [...]}, read by
`ImageWAM/scripts/qwenimage21/precompute_qwenimage21_text_embeds.py --prompts-json`.

  .venv/bin/python launch/robotwin_qwen_prompts.py <dataset root> <out dir> --shards 4
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ImageWAM" / "src"))
from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    tasks = [json.loads(line)["task"] for line in (args.dataset_root / "meta" / "tasks.jsonl").read_text().splitlines()
             if line.strip()]
    prompts = list(dict.fromkeys(DEFAULT_PROMPT.format(task=task) for task in tasks))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.shards):
        shard = prompts[i :: args.shards]
        path = args.out_dir / f"prompts_{i}_of_{args.shards}.json"
        path.write_text(json.dumps({"prompts": shard}))
        print(f"{path}: {len(shard)} prompts")
    print(f"{len(tasks)} task strings, {len(prompts)} distinct prompts")


if __name__ == "__main__":
    main()
