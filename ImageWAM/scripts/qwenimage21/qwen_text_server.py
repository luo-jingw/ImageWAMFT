"""Qwen-Image-2.1 text features for evaluation workers that cannot load Qwen3-VL themselves.

The Qwen-Image-2.1 stack has no in-process text encoder, and Qwen3-VL needs transformers 5 (`.venv-qwenimage`), while the
RoboTwin evaluation workers run in `.venv-robotwin`. This process loads the pipeline once and serves requests on a local
TCP port (`multiprocessing.connection`). A request is one `DEFAULT_PROMPT`-wrapped prompt; the reply is the path of its
cache file in `--cache-dir`, written by `encode()` of `precompute_qwenimage21_text_embeds.py` with the same file naming,
so the features equal the training cache's for the same prompt. Existing files are reused.

  .venv-qwenimage/bin/python ImageWAM/scripts/qwenimage21/qwen_text_server.py --model-dir BaseModel/Qwen-Image-2.1 \
      --cache-dir <dir> --port 29610 --device cuda:0
"""

import argparse
import os
import threading
from multiprocessing.connection import Listener
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline

from precompute_qwenimage21_text_embeds import cache_path, encode

AUTHKEY = b"imagewamft-qwen-text"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    # Only the processor and the Qwen3-VL text encoder are used; without the 7B transformer the server holds about
    # 17 GB of GPU memory instead of 31 GB, and the features are bitwise identical.
    pipe = QwenImage21Pipeline.from_pretrained(args.model_dir, transformer=None, torch_dtype=torch.bfloat16).to(device)
    lock = threading.Lock()

    def features_file(prompt: str) -> str:
        path = cache_path(args.cache_dir, prompt, args.context_len)
        with lock:
            if not path.exists():
                payload = encode(pipe, prompt, args.context_len, device)
                tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
                torch.save(payload, tmp)
                os.replace(tmp, path)
                print(f"wrote {path.name}: tokens={int(payload['text_attention_mask'].sum())} prompt={prompt!r}",
                      flush=True)
        return str(path)

    def serve(conn) -> None:
        with conn:
            while True:
                try:
                    prompt = conn.recv()
                except EOFError:
                    return
                try:
                    conn.send({"path": features_file(str(prompt))})
                except Exception as error:  # reported to the worker, which raises it
                    conn.send({"error": repr(error)})

    with Listener(("127.0.0.1", args.port), authkey=AUTHKEY) as listener:
        print(f"ready on 127.0.0.1:{args.port} cache={args.cache_dir}", flush=True)
        while True:
            threading.Thread(target=serve, args=(listener.accept(),), daemon=True).start()


if __name__ == "__main__":
    main()
