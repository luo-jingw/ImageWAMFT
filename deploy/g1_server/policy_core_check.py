"""Checks the server's policy core against the training pipeline on held-out LGG100 samples (run in `.venv`).

For evenly spaced samples of the run's validation split (4 held-out episodes, one per dataset root):
  1. inputs (CPU): the server image tensor built from the raw 480×640 frames vs. the dataset's `video[:, 0]`, and the
     server proprio vs. the dataset's normalized `proprio[0]`;
  2. per model (GPU, without --cpu-only): `G1Policy.infer` vs. `ImageWAM.infer_action` on the dataset sample with the
     cached text features (same seed and steps), action L1 against the recorded actions, FLUX.2 online text features
     vs. the training cache, load time, GPU memory and latency.

  PYTHONPATH=deploy .venv/bin/python deploy/g1_server/policy_core_check.py [--models flux2_4b,qwenimage21,flux2_9b]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

from g1_server.g1_policy import G1Policy
from g1_server.heldout_samples import raw_request, validation_dataset
from g1_server.inference_settings import InferenceSettings
from g1_server.model_registry import MODEL_SPECS, available_steps, spec_by_name
from g1_server.observation_preprocessor import G1ObservationPreprocessor


def check_inputs(run_dir: Path, indices: list[int]) -> dict:
    cfg, dataset = validation_dataset(run_dir)
    processor = instantiate(cfg.data.val.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(run_dir / "dataset_stats.json")))
    preprocessor = G1ObservationPreprocessor(cfg.data.val, processor, torch.device("cpu"), torch.float32)
    image_diff, proprio_diff, shapes = [], [], set()
    for index in indices:
        sample = dataset[index]
        request, _ = raw_request(dataset, index)
        image = preprocessor.image(request.views)[0]
        shapes.add(tuple(request.views[0].shape))
        image_diff.append(float((image - sample["video"][:, 0].float()).abs().max()))
        proprio = preprocessor.proprio(request.state)
        proprio_diff.append(float((proprio - sample["proprio"][0].float()).abs().max()))
    return {"samples": len(indices), "raw_view_shape": sorted(shapes), "image_max_abs_diff": max(image_diff),
            "proprio_max_abs_diff": max(proprio_diff), "image_shape": list(image.shape)}


@torch.no_grad()
def check_model(name: str, indices: list[int], device: torch.device) -> dict:
    spec = spec_by_name(name)
    step = available_steps(spec)[-1]
    torch.zeros(1, device=device)  # initializes CUDA; peak-memory statistics need an initialized allocator
    torch.cuda.reset_peak_memory_stats(device)
    policy = G1Policy.load(spec, step, device)
    load_memory = torch.cuda.max_memory_allocated(device) / 2**30
    _, dataset = validation_dataset(spec.run_dir)
    settings = InferenceSettings()
    rows = []
    for index in indices:
        sample = dataset[index]
        request, recorded = raw_request(dataset, index)
        out = policy.infer(request, settings)
        reference = policy.model.infer_action(
            prompt=None, input_image=sample["video"][:, 0].unsqueeze(0).to(device=device, dtype=policy.model.torch_dtype),
            proprio=sample["proprio"][0], context=sample["text_hidden_states"], context_mask=sample["text_attention_mask"],
            action_horizon=16, num_inference_steps=settings.num_inference_steps, seed=settings.seed, rand_device="cpu")
        reference_raw = policy.preprocessor.denormalize_actions(reference["action"].float())
        rows.append({"server_vs_reference_max_abs": float(np.abs(out.actions - reference_raw).max()),
                     "l1_vs_recorded": float(np.abs(out.actions - recorded).mean()),
                     "prompt_source": out.prompt_source, "total_ms": out.timing.total_ms,
                     "model_ms": out.timing.model_ms})
    result = {"model": spec.label, "step": step, "load_seconds": round(policy.load_seconds, 1),
              "peak_memory_after_load_gib": round(load_memory, 1),
              "server_vs_reference_max_abs": max(r["server_vs_reference_max_abs"] for r in rows),
              "l1_vs_recorded_rad": float(np.mean([r["l1_vs_recorded"] for r in rows])),
              "prompt_sources": sorted({r["prompt_source"] for r in rows}),
              "model_ms_median": float(np.median([r["model_ms"] for r in rows[1:]])),
              "total_ms_median": float(np.median([r["total_ms"] for r in rows[1:]]))}
    image = policy.predict_image(request, settings)
    result["predicted_image_shape"] = list(image.shape)
    if spec.stack == "flux2":
        diffs = []
        request, _ = raw_request(dataset, indices[0])
        inputs = policy.inputs(request, settings)
        for task in policy.tasks:
            cached = policy.prompts.features(task)
            online, mask = policy.prompts.encoder.encode(DEFAULT_PROMPT.format(task=task))
            valid = cached.mask
            difference = online[valid].float() - cached.context[valid].float()
            actions = []
            for context in (cached.context, online):
                pred = policy.model.infer_action(prompt=None, input_image=inputs.image, proprio=inputs.proprio,
                                                 context=context, context_mask=cached.mask, action_horizon=16,
                                                 num_inference_steps=settings.num_inference_steps, seed=settings.seed,
                                                 rand_device="cpu")
                actions.append(policy.preprocessor.denormalize_actions(pred["action"].float()))
            diffs.append({"mask_equal": bool(torch.equal(mask, cached.mask)),
                          "relative_l2": float(difference.norm() / cached.context[valid].float().norm()),
                          "max_abs": float(difference.abs().max()),
                          "action_max_abs_rad": float(np.abs(actions[0] - actions[1]).max()),
                          "action_mean_abs_rad": float(np.abs(actions[0] - actions[1]).mean())})
        result["flux2_online_text_vs_cache"] = diffs
    result["peak_memory_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 1)
    policy.unload()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=str, default=",".join(spec.name for spec in MODEL_SPECS))
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run_dir = MODEL_SPECS[0].run_dir
    _, dataset = validation_dataset(run_dir)
    indices = [int(i) for i in np.linspace(0, len(dataset) - 17, args.samples)]
    report = {"indices": indices, "inputs": check_inputs(run_dir, indices)}
    print(json.dumps(report["inputs"]), flush=True)
    if not args.cpu_only:
        device = torch.device("cuda:0")
        report["models"] = []
        for name in args.models.split(","):
            start = time.perf_counter()
            report["models"].append(check_model(name, indices, device))
            print(json.dumps(report["models"][-1]), f"({time.perf_counter() - start:.0f} s)", flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
