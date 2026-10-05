"""One trained G1 ImageWAM model at one checkpoint step: observation → de-normalized action chunk.

The model is built from the run's own `config.yaml` (`instantiate(cfg.model)`, as `run_training` does) and loaded with
`ImageWAM.load_checkpoint`. Observations go through `G1ObservationPreprocessor`, text through `PromptFeatureStore`,
and actions come from `ImageWAM.infer_action` with the cached text features.
"""

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from imagewam.models.backbones.imagewam import ImageWAM

from g1_server.inference_settings import InferenceSettings
from g1_server.model_registry import PROJECT_ROOT, ModelSpec, weights_path
from g1_server.observation_preprocessor import G1ObservationPreprocessor
from g1_server.prompt_features import (Flux2PromptEncoder, PromptEncoder, PromptFeatureStore, QwenImage21PromptEncoder,
                                       training_tasks)
from g1_server.protocol import ObservationRequest, ServerTiming

IMAGEWAM_ROOT = PROJECT_ROOT / "ImageWAM"  # working directory of training; relative config paths resolve here
SERVER_CACHE_ROOT = PROJECT_ROOT / "deploy" / "g1_server" / "cache"
MODEL_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}


@dataclass(frozen=True)
class PolicyOutput:
    actions: np.ndarray  # float32 [H, 16], de-normalized
    actions_normalized: np.ndarray  # float32 [H, 16], model output
    model_image: np.ndarray  # uint8 [288, 256, 3], the image the model saw
    task: str
    prompt_source: str
    seed: int
    model: str  # display name of the model that produced the chunk
    step: int
    num_inference_steps: int
    timing: ServerTiming


@dataclass(frozen=True)
class ModelInputs:
    image: torch.Tensor  # [1, 3, 288, 256] in [-1, 1], model device and dtype
    proprio: torch.Tensor  # [16] normalized, model device
    context: torch.Tensor  # [context_len, text_dim]
    context_mask: torch.Tensor  # [context_len] bool
    task: str
    prompt_source: str
    preprocess_ms: float
    text_ms: float


def model_config(cfg: DictConfig) -> DictConfig:
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    init_path = model_cfg.get("action_dit_pretrained_path")
    if init_path and not Path(init_path).is_absolute():
        model_cfg["action_dit_pretrained_path"] = str(IMAGEWAM_ROOT / init_path)
    return OmegaConf.create(model_cfg)


class G1Policy:
    def __init__(self, spec: ModelSpec, step: int, cfg: DictConfig, model: ImageWAM,
                 preprocessor: G1ObservationPreprocessor, prompts: PromptFeatureStore, load_seconds: float) -> None:
        self.spec = spec
        self.step = step
        self.cfg = cfg
        self.model = model
        self.preprocessor = preprocessor
        self.prompts = prompts
        self.load_seconds = load_seconds
        self.tasks = training_tasks(list(cfg.data.val.dataset_dirs))
        self.default_sigma_shift = float(cfg.model.action_scheduler.infer_shift)

    @classmethod
    def load(cls, spec: ModelSpec, step: int, device: torch.device) -> "G1Policy":
        start = time.perf_counter()
        cfg = OmegaConf.load(spec.run_dir / "config.yaml")
        model = instantiate(model_config(cfg), model_dtype=MODEL_DTYPE[str(cfg.mixed_precision)], device=str(device))
        model.load_checkpoint(str(weights_path(spec, step)))
        model = model.to(device).eval()
        processor = instantiate(cfg.data.val.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(spec.run_dir / "dataset_stats.json")))
        preprocessor = G1ObservationPreprocessor(cfg.data.val, processor, device, model.torch_dtype)
        cache_format = str(cfg.data.val.qwen_text_cache_format)
        context_len = int(cfg.data.val.qwen_context_len)
        server_cache = SERVER_CACHE_ROOT / spec.name
        encoder: PromptEncoder
        if spec.stack == "flux2":
            encoder = Flux2PromptEncoder(model)
        else:
            encoder = QwenImage21PromptEncoder(server_cache, cache_format, context_len)
        prompts = PromptFeatureStore(Path(cfg.data.val.qwen_text_cache_dir), server_cache, cache_format, context_len,
                                     encoder)
        torch.cuda.synchronize(device)
        return cls(spec, step, cfg, model, preprocessor, prompts, time.perf_counter() - start)

    def inputs(self, request: ObservationRequest, settings: InferenceSettings) -> ModelInputs:
        start = time.perf_counter()
        image = self.preprocessor.image(request.views)
        proprio = self.preprocessor.proprio(request.state).to(self.model.device)
        torch.cuda.synchronize(self.model.device)
        preprocessed = time.perf_counter()
        task = settings.prompt_override or request.prompt
        features = self.prompts.features(task)
        return ModelInputs(image=image, proprio=proprio, context=features.context, context_mask=features.mask,
                           task=task, prompt_source=features.source, preprocess_ms=(preprocessed - start) * 1e3,
                           text_ms=(time.perf_counter() - preprocessed) * 1e3)

    @torch.no_grad()
    def infer(self, request: ObservationRequest, settings: InferenceSettings) -> PolicyOutput:
        start = time.perf_counter()
        inputs = self.inputs(request, settings)
        seed = settings.seed if settings.seed is not None else int(np.random.default_rng().integers(2**31 - 1))
        model_start = time.perf_counter()
        pred = self.model.infer_action(prompt=None, input_image=inputs.image, proprio=inputs.proprio,
                                       context=inputs.context, context_mask=inputs.context_mask,
                                       action_horizon=settings.action_horizon,
                                       num_inference_steps=settings.num_inference_steps,
                                       sigma_shift=settings.sigma_shift, seed=seed, rand_device="cpu")
        torch.cuda.synchronize(self.model.device)
        model_ms = (time.perf_counter() - model_start) * 1e3
        actions_normalized = pred["action"].float()
        actions = self.preprocessor.denormalize_actions(actions_normalized)
        timing = ServerTiming(preprocess_ms=inputs.preprocess_ms, text_ms=inputs.text_ms, model_ms=model_ms,
                              total_ms=(time.perf_counter() - start) * 1e3)
        return PolicyOutput(actions=actions, actions_normalized=actions_normalized.numpy(),
                            model_image=G1ObservationPreprocessor.to_uint8_image(inputs.image), task=inputs.task,
                            prompt_source=inputs.prompt_source, seed=seed, model=self.spec.label, step=self.step,
                            num_inference_steps=settings.num_inference_steps, timing=timing)

    @torch.no_grad()
    def predict_image(self, request: ObservationRequest, settings: InferenceSettings) -> np.ndarray:
        """The model's predicted frame 16 steps ahead, uint8 [288, 256, 3] (joint image and action inference)."""
        inputs = self.inputs(request, settings)
        seed = 42 if settings.seed is None else settings.seed
        pred = self.model.infer(prompt=None, input_image=inputs.image, proprio=inputs.proprio, context=inputs.context,
                                context_mask=inputs.context_mask, num_frames=2, action_horizon=settings.action_horizon, text_cfg_scale=1.0,
                                action_cfg_scale=1.0, num_inference_steps=settings.num_inference_steps,
                                sigma_shift=settings.sigma_shift, seed=seed, tiled=False)
        return np.asarray(pred["video"][-1].convert("RGB"), dtype=np.uint8)

    def unload(self) -> None:
        """Drops the GPU references; the policy is unusable afterwards."""
        del self.model, self.prompts
        gc.collect()
        torch.cuda.empty_cache()
