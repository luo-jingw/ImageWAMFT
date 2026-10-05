"""Held-out LGG100 samples (the run's validation split: 4 episodes, one per dataset root) in robot-side form."""

import tempfile
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from imagewam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from imagewam.utils import misc

from g1_server.protocol import CAMERA_KEYS, ObservationRequest


def validation_dataset(run_dir: Path) -> tuple[DictConfig, RobotVideoDataset]:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    val_cfg = cfg.data.val
    val_cfg.pretrained_norm_stats = str(run_dir / "dataset_stats.json")
    misc.register_work_dir(tempfile.mkdtemp(prefix="g1_server_check_"))  # the dataset writes its stats copy here
    dataset = instantiate(val_cfg)
    dataset.video_augmentation = None
    return cfg, dataset


def raw_request(dataset: RobotVideoDataset, index: int) -> tuple[ObservationRequest, np.ndarray]:
    """Sample `index` as the robot sends it: frame t of each camera (RGB uint8 [480, 640, 3]), the raw state at t and
    the task string; plus the recorded raw actions t..t+15 [16, 16]."""
    base = dataset.lerobot_dataset
    processor, base.processor = base.processor, None
    try:
        raw = base[index]
    finally:
        base.processor = processor
    views = tuple(raw["images"][key][0].permute(1, 2, 0).numpy() for key in CAMERA_KEYS)
    state = raw["state"]["default"][0].numpy().astype(np.float32)
    actions = raw["action"]["default"][:16].numpy().astype(np.float32)
    return ObservationRequest(views=views, state=state, prompt=str(raw["task"])), actions
