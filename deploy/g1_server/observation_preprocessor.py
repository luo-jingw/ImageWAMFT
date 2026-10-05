"""Robot observation → model inputs, with the transforms of the run's validation data pipeline.

Image path, per view, as `RobotVideoDataset` with `concat_multi_camera: robotwin` builds it:
  uint8 [3, H, W] → processor validation transforms (ToTensor, Resize to 240×320) → bilinear antialiased resize to the
  RoboTwin tile size (head 192×256, wrists 96×128) → head on top, left and right wrist below (288×256) →
  ResizeSmallestSideAspectPreserving and CenterCrop to `video_size` → Normalize(0.5, 0.5) to [-1, 1].
State: z-score with the run's `dataset_stats.json` through the processor normalizer. Actions: the inverse.
"""

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from omegaconf import DictConfig

from imagewam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from imagewam.datasets.lerobot.processors.imagewam_processor import ImageWAMProcessor
from imagewam.datasets.lerobot.robot_video_dataset import RobotVideoDataset

from g1_server.protocol import CAMERA_KEYS


class G1ObservationPreprocessor:
    def __init__(self, data_cfg: DictConfig, processor: ImageWAMProcessor, device: torch.device,
                 dtype: torch.dtype) -> None:
        keys = tuple(meta["key"] for meta in processor.shape_meta["images"])
        if keys != CAMERA_KEYS:
            raise ValueError(f"run cameras {keys} differ from the protocol order {CAMERA_KEYS}")
        if str(data_cfg.concat_multi_camera) != "robotwin":
            raise ValueError(f"expected concat_multi_camera robotwin, got {data_cfg.concat_multi_camera}")
        self.processor = processor
        self.view_transforms = list(processor.val_transforms)
        self.tile_sizes = RobotVideoDataset._robotwin_camera_sizes(str(data_cfg.robotwin_camera_layout))
        video_h, video_w = (int(v) for v in data_cfg.video_size)
        self.resize = ResizeSmallestSideAspectPreserving(args={"img_w": video_w, "img_h": video_h})
        self.crop = CenterCrop(args={"img_w": video_w, "img_h": video_h})
        self.normalize = Normalize(args={"mean": 0.5, "std": 0.5})
        self.state_key = processor.shape_meta["state"][0]["key"]
        self.action_key = processor.shape_meta["action"][0]["key"]
        self.device = device
        self.dtype = dtype

    def image(self, views: tuple[np.ndarray, ...]) -> torch.Tensor:
        """Three RGB uint8 [H, W, 3] views → [1, 3, 288, 256] in [-1, 1] on the model device."""
        tiles = []
        for view, size in zip(views, self.tile_sizes):
            frame = torch.from_numpy(np.ascontiguousarray(view)).permute(2, 0, 1).unsqueeze(0)  # [T=1, 3, H, W] uint8
            for transform in self.view_transforms:
                frame = transform(frame)
            tiles.append(transforms_F.resize(frame, size=size, interpolation=transforms_F.InterpolationMode.BILINEAR,
                                             antialias=True))
        video = torch.cat([tiles[0], torch.cat([tiles[1], tiles[2]], dim=-1)], dim=-2)
        video = self.normalize(self.crop(self.resize(video)))  # [1, 3, 288, 256]
        return video.to(device=self.device, dtype=self.dtype)

    def proprio(self, state: np.ndarray) -> torch.Tensor:
        """Raw 16-dim state → normalized [16] float32."""
        batch = {"state": {self.state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][self.state_key][0]

    def denormalize_actions(self, actions: torch.Tensor) -> np.ndarray:
        """Normalized [H, 16] → raw [H, 16] float32."""
        normalizer = self.processor.normalizer.normalizers["action"][self.action_key]
        raw = normalizer.backward(actions.to(dtype=torch.float32, device="cpu").unsqueeze(0))[0]
        return raw.numpy().astype(np.float32)

    @staticmethod
    def to_uint8_image(image: torch.Tensor) -> np.ndarray:
        """[1, 3, H, W] in [-1, 1] → uint8 [H, W, 3], for display."""
        pixels = ((image[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        return pixels.permute(1, 2, 0).cpu().numpy()
