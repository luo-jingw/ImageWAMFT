"""Export Qwen-Image-2.1 parity-fixture inputs from the G1 validation split (run in the main `.venv`).

Writes, into `--out-dir`:
  current.png / target.png  frames t and t+16 of one validation sample, RoboTwin-tiled 288x256 (no augmentation)
  prompts.json              the four LGG100 task strings wrapped in `DEFAULT_PROMPT`, in dataset order
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from PIL import Image

from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from imagewam.runtime import build_datasets
from imagewam.utils import misc

DATASETS = ("drawer", "biomanual", "simple-pick", "Stack-the-cubes")


def frame_to_image(frame: torch.Tensor) -> Image.Image:
    """[3,H,W] in [-1,1] -> RGB PIL image."""
    array = ((frame.float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(np.ascontiguousarray(array))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True, help="ImageWAM directory")
    parser.add_argument("--data-root", type=Path, required=True, help="directory holding LGG100/<dataset>")
    parser.add_argument("--norm-stats", type=Path, required=True, help="dataset_stats.json of the G1 training split")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-index", type=int, default=100)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(args.out_dir))
    roots = [str(args.data_root / "LGG100" / name) for name in DATASETS]
    dataset_dirs = "[" + ",".join(roots) + "]"
    with initialize_config_dir(config_dir=str(args.repo_root / "configs"), version_base="1.3"):
        cfg = compose(
            "train",
            overrides=[
                "task=g1_lgg100_flux2_klein_4b_base_imagewam",
                f"data.train.dataset_dirs={dataset_dirs}",
                f"data.val.dataset_dirs={dataset_dirs}",
                "data.train.qwen_text_cache_dir=null",
                "data.val.qwen_text_cache_dir=null",
                f"data.train.pretrained_norm_stats={args.norm_stats}",
                "model.flux2_model_path=unused",
                "model.ae_model_path=unused",
            ],
        )
    _, val_ds = build_datasets(cfg.data)
    sample = val_ds[args.val_index]
    video = sample["video"]  # [3, 2, 288, 256]
    frame_to_image(video[:, 0]).save(args.out_dir / "current.png")
    frame_to_image(video[:, -1]).save(args.out_dir / "target.png")

    prompts = []
    for root in roots:
        with open(Path(root) / "meta" / "tasks.jsonl", "r", encoding="utf-8") as f:
            task = json.loads(f.readline())["task"]
        prompts.append(DEFAULT_PROMPT.format(task=task))
    (args.out_dir / "prompts.json").write_text(json.dumps({"datasets": list(DATASETS), "prompts": prompts}, indent=2))
    print(f"wrote current.png/target.png ({tuple(video.shape)}) and {len(prompts)} prompts to {args.out_dir}")
    print(f"sample prompt: {sample['instruction']!r}")


if __name__ == "__main__":
    main()
