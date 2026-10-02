"""Precompute Qwen-Image-2.1 text-only Qwen3-VL prompt features (run in `.venv-qwenimage`).

Features follow the text-to-image path of `QwenImage21Pipeline._get_qwen_prompt_embeds` (diffusers @ 578c9b2):
template without image slots, last decoder layer before the final norm, system tokens dropped (plan.md D1).

Input: `--prompts-json` with {"prompts": [...]}, the exact dataset prompts (`DEFAULT_PROMPT`-wrapped task
strings, e.g. written by `export_fixture_inputs.py`).
Output: `<cache-dir>/<sha256(prompt)>.qwen3vl_qwenimage21_len<L>.pt` with
  text_hidden_states [L, 4096] bf16 (zero-padded) and text_attention_mask [L] bool,
loaded by `RobotVideoDataset` with `qwen_text_cache_format: qwen3vl_qwenimage21`.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from diffusers import QwenImage21Pipeline


def cache_path(cache_dir: Path, prompt: str, context_len: int) -> Path:
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache_dir / f"{hashed}.qwen3vl_qwenimage21_len{context_len}.pt"


@torch.no_grad()
def encode(pipe: QwenImage21Pipeline, prompt: str, context_len: int, device: torch.device) -> dict[str, torch.Tensor]:
    embeds, mask, image_pad_mask = pipe._get_qwen_prompt_embeds([prompt], None, device)
    if bool(image_pad_mask.any()):
        raise ValueError("text-to-image template must contain no image slots")
    valid = int(mask.sum()) if mask is not None else int(embeds.shape[1])
    if valid > context_len:
        raise ValueError(f"prompt has {valid} tokens > context_len {context_len}: {prompt!r}")
    hidden = embeds[0, :valid].to(torch.bfloat16).cpu()
    padded = torch.zeros(context_len, hidden.shape[1], dtype=torch.bfloat16)
    padded[:valid] = hidden
    attention_mask = torch.zeros(context_len, dtype=torch.bool)
    attention_mask[:valid] = True
    return {"text_hidden_states": padded, "text_attention_mask": attention_mask}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prompts = json.loads(args.prompts_json.read_text())["prompts"]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    pipe = QwenImage21Pipeline.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16).to(device)
    for prompt in prompts:
        path = cache_path(args.cache_dir, prompt, args.context_len)
        if path.exists() and not args.overwrite:
            print(f"skip (exists): {path.name}")
            continue
        payload = encode(pipe, prompt, args.context_len, device)
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        print(f"wrote {path.name}: tokens={int(payload['text_attention_mask'].sum())} prompt={prompt[:80]!r}")


if __name__ == "__main__":
    main()
