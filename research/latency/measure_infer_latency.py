"""Inference latency breakdown of a trained G1 ImageWAM run (batch 1, bf16, one GPU, run from ImageWAM/).

Times, with CUDA synchronization, the full `infer_action` at 1 and at 10 action denoising steps; the difference gives
the per-step cost of the action expert over the cached prefix, the remainder the VAE encode + prefix prefill. Also
times the VAE encode alone and reports the prefix token counts (valid text, padded text, image).

  python ../research/latency/measure_infer_latency.py --run-dir <run> --checkpoint <weights .pt>
"""

import argparse
import statistics
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from imagewam.runtime import build_datasets
from imagewam.utils import misc


def timed(fn, repeats: int) -> list[float]:
    out = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - start) * 1000)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    misc.register_work_dir(str(args.work_dir))
    cfg.data.train.pretrained_norm_stats = str(args.run_dir / "dataset_stats.json")
    _, val = build_datasets(cfg.data)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(args.checkpoint))
    model.eval()
    sample = val[100]
    image = sample["video"][:, 0].unsqueeze(0)
    kwargs = dict(prompt=None, input_image=image, action_horizon=16, proprio=sample["proprio"][0],
                  context=sample["text_hidden_states"], context_mask=sample["text_attention_mask"], seed=0)
    with torch.no_grad():
        if model.stack == "flux2":
            encode = lambda: model._encode_flux2_image_tokens(image.cuda().to(torch.bfloat16), time_value=10.0)  # noqa: E731
        else:
            encode = lambda: model._qwenimage21_stack().encode_image(image)  # noqa: E731
        for steps in (1, 10):
            timed(lambda: model.infer_action(num_inference_steps=steps, **kwargs), 3)  # warm-up
        t1 = timed(lambda: model.infer_action(num_inference_steps=1, **kwargs), args.repeats)
        t10 = timed(lambda: model.infer_action(num_inference_steps=10, **kwargs), args.repeats)
        tenc = timed(encode, args.repeats)
    m1, m10, menc = statistics.median(t1), statistics.median(t10), statistics.median(tenc)
    per_step = (m10 - m1) / 9
    valid_text = int(sample["text_attention_mask"].sum())
    print(f"stack={model.stack} gpu={torch.cuda.get_device_name()}")
    print(f"prefix tokens: text {valid_text} valid + {sample['text_attention_mask'].numel() - valid_text} padded + 1 proprio, "
          f"image {image.shape[-2] // 16 * image.shape[-1] // 16}")
    print(f"infer_action 1 step {m1:.1f} ms | 10 steps {m10:.1f} ms | per action step {per_step:.1f} ms | "
          f"VAE encode {menc:.1f} ms | prefill (+ overhead) {m1 - per_step - menc:.1f} ms")
    print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
