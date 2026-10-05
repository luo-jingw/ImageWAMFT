"""Text features of a task string for one loaded model.

A task string is wrapped in `DEFAULT_PROMPT` and looked up by SHA-256, with the file naming of the training text
caches: first the run's training cache (the four LGG100 task strings, bit-identical to training), then the server's
per-model cache. A miss is encoded by the model's prompt encoder and stored in the server cache:
  - FLUX.2: the model's own Qwen3 text encoder (`ImageWAM._encode_flux2_prompts`, loaded on first use);
  - Qwen-Image-2.1: `precompute_qwenimage21_text_embeds.py` in `.venv-qwenimage` (transformers 5), as a subprocess.
"""

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from imagewam.models.backbones.imagewam import ImageWAM

from g1_server.model_registry import PROJECT_ROOT


@dataclass(frozen=True)
class PromptFeatures:
    task: str
    context: torch.Tensor  # [context_len, text_dim] bfloat16, CPU
    mask: torch.Tensor  # [context_len] bool, CPU
    source: str  # "training_cache", "server_cache" or the encoder's source


class PromptEncoder(Protocol):
    source: str

    def encode(self, wrapped_prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """`DEFAULT_PROMPT`-wrapped task string → ([context_len, text_dim] bfloat16, [context_len] bool) on CPU."""


class Flux2PromptEncoder:
    source = "flux2_qwen3_online"

    def __init__(self, model: ImageWAM) -> None:
        self.model = model

    @torch.no_grad()
    def encode(self, wrapped_prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, mask = self.model._encode_flux2_prompts([wrapped_prompt])
        return hidden[0].to(device="cpu", dtype=torch.bfloat16), mask[0].to(device="cpu", dtype=torch.bool)


class QwenImage21PromptEncoder:
    """Runs the training precompute script for one prompt; it writes into `cache_dir` with the cache file naming."""

    source = "qwen3vl_subprocess"

    def __init__(self, cache_dir: Path, cache_format: str, context_len: int) -> None:
        self.cache_dir = cache_dir
        self.cache_format = cache_format
        self.context_len = context_len

    def encode(self, wrapped_prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"prompts": [wrapped_prompt]}, handle)
        command = [str(PROJECT_ROOT / ".venv-qwenimage" / "bin" / "python"),
                   str(PROJECT_ROOT / "ImageWAM" / "scripts" / "qwenimage21" / "precompute_qwenimage21_text_embeds.py"),
                   "--model-dir", str(PROJECT_ROOT / "BaseModel" / "Qwen-Image-2.1"), "--prompts-json", handle.name,
                   "--cache-dir", str(self.cache_dir), "--context-len", str(self.context_len), "--device", "cuda:0"]
        try:
            subprocess.run(command, check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"}, capture_output=True,
                           text=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Qwen-Image-2.1 text encoding failed:\n{error.stderr[-3000:]}") from error
        finally:
            os.unlink(handle.name)
        payload = torch.load(self.cache_dir / cache_file_name(wrapped_prompt, self.cache_format, self.context_len),
                             map_location="cpu")
        return payload["text_hidden_states"], payload["text_attention_mask"].bool()


def cache_file_name(wrapped_prompt: str, cache_format: str, context_len: int) -> str:
    return f"{hashlib.sha256(wrapped_prompt.encode('utf-8')).hexdigest()}.{cache_format}_len{context_len}.pt"


def training_tasks(dataset_dirs: list[str]) -> list[str]:
    """Task strings of the training datasets (LeRobot `meta/tasks.jsonl`), in dataset order."""
    tasks = []
    for directory in dataset_dirs:
        for line in (Path(directory) / "meta" / "tasks.jsonl").read_text().splitlines():
            task = json.loads(line)["task"]
            if task not in tasks:
                tasks.append(task)
    return tasks


class PromptFeatureStore:
    def __init__(self, training_cache_dir: Path, server_cache_dir: Path, cache_format: str, context_len: int,
                 encoder: PromptEncoder) -> None:
        self.training_cache_dir = training_cache_dir
        self.server_cache_dir = server_cache_dir
        self.cache_format = cache_format
        self.context_len = context_len
        self.encoder = encoder
        self.memo: dict[str, PromptFeatures] = {}

    def features(self, task: str) -> PromptFeatures:
        if task in self.memo:
            return self.memo[task]
        wrapped = DEFAULT_PROMPT.format(task=task)
        name = cache_file_name(wrapped, self.cache_format, self.context_len)
        found = None
        for directory, source in ((self.training_cache_dir, "training_cache"), (self.server_cache_dir, "server_cache")):
            if (directory / name).exists():
                payload = torch.load(directory / name, map_location="cpu")
                found = PromptFeatures(task, payload["text_hidden_states"], payload["text_attention_mask"].bool(),
                                       source)
                break
        if found is None:
            context, mask = self.encoder.encode(wrapped)
            self.server_cache_dir.mkdir(parents=True, exist_ok=True)
            if not (self.server_cache_dir / name).exists():
                torch.save({"text_hidden_states": context, "text_attention_mask": mask}, self.server_cache_dir / name)
            found = PromptFeatures(task, context, mask, self.encoder.source)
        if found.context.shape[0] != self.context_len or found.mask.shape != (self.context_len,):
            raise ValueError(f"text features {tuple(found.context.shape)} do not match context_len {self.context_len}")
        self.memo[task] = found
        return found
