"""LIBERO rollouts of ImageWAM-FLUX.2-4B with a random fraction of the reference-image tokens masked (run in
`.venv-libero`, see PLAN.md).

The configuration, observation preprocessing, action post-processing and episode loop follow
`ImageWAM/experiments/libero/eval_libero_single.py` and `scripts/flux2/run_eval_flux2_libero.sh`; the official helpers
are imported and called directly. Added per inference call:
  - token mask: a random subset of the 392 reference-image tokens is removed from every attention row (prefix prefill
    and action denoising) by disabling its column in the FLUX.2 MoT attention mask;
  - paired control: for masked levels, a second inference from the same observation and the same initial noise
    without the mask (recorded, not executed);
  - attention summary: action-to-prefix attention per layer, averaged over heads, action tokens and denoising steps;
  - the MuJoCo state before the call, the token mask, both action chunks, the executed actions and the wall time.

  python eval_token_mask.py --out-dir <dir> --worker-index i --num-workers N
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGEWAM = PROJECT_ROOT / "ImageWAM"
FLUX2_SRC = IMAGEWAM / "third_party" / "flux2"
for path in (FLUX2_SRC, FLUX2_SRC / "src", IMAGEWAM / "experiments" / "libero", IMAGEWAM / "src", IMAGEWAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from experiments.libero import eval_libero_single as official  # noqa: E402
from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    save_rollout_video,
)
from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from imagewam.utils.pytorch_utils import set_global_seed  # noqa: E402

CHECKPOINT_DIR = IMAGEWAM / "checkpoints" / "imagewam_release" / "libero" / "flux2_klein_4b"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
FRACTIONS = (0.0, 1.0 / 3.0, 2.0 / 3.0)
NUM_TRIALS = 5
IMAGE_TOKENS = 392  # 224 x 448 input, 16 x 16 pixels per token


def eval_config():
    overrides = [
        "task=libero_flux2_klein_4b_base_imagewam",
        f"ckpt={CHECKPOINT_DIR / 'model.pt'}",
        f"EVALUATION.dataset_stats_path={CHECKPOINT_DIR / 'dataset_stats.json'}",
        f"model.flux2_src_path={FLUX2_SRC}",
        f"model.flux2_model_path={PROJECT_ROOT / 'BaseModel/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors'}",
        f"model.ae_model_path={PROJECT_ROOT / 'BaseModel/flux2/FLUX.2-dev/ae.safetensors'}",
        "model.variant=klein-base-4b",
        f"model.qwen3_model_spec={PROJECT_ROOT / 'BaseModel/Qwen3-4B'}",
        "model.load_text_encoder=true",
        "model.pack_proprio_after_text=true",
        "EVALUATION.action_horizon=16",
        "EVALUATION.replan_steps=12",
        "model.proprio_dim=8",
        "data.train.qwen_context_len=512",
        "data.train.qwen_text_cache_format=qwen3_flux2",
    ]
    with initialize_config_dir(config_dir=str(IMAGEWAM / "configs"), version_base="1.3"):
        return compose("sim_libero_omnigen2", overrides=overrides)


@dataclass
class TokenMask:
    """Reference-image token indices hidden from every attention row; None = no masking."""

    drop: torch.Tensor | None = None


def install_token_mask(model, token_mask: TokenMask) -> None:
    original = model._build_mot_attention_mask_flux2

    def build(*args, **kwargs):
        masks = original(*args, **kwargs)
        if token_mask.drop is not None:
            if int(kwargs["cond_len"]) != IMAGE_TOKENS:
                raise ValueError(f"expected {IMAGE_TOKENS} reference tokens, got {kwargs['cond_len']}")
            columns = int(kwargs["txt_len"]) + token_mask.drop.to(kwargs["device"])
            for mask in masks.values():
                mask[:, :, columns] = False
        return masks

    model._build_mot_attention_mask_flux2 = build


class ActionAttentionRecorder:
    """MoT `action_attention_capture` interface: per layer, action-to-key attention averaged over batch, heads, action
    queries and denoising steps; keeps the prefix part and the action-to-action total."""

    def __init__(self) -> None:
        self.prefix_len = 0
        self.prefix: dict[int, torch.Tensor] = {}
        self.self_mass: dict[int, float] = {}
        self.counts: dict[int, int] = {}

    def configure(self, *, condition_slice, condition_grid, prefix_len, action_len, metadata) -> None:
        self.prefix_len = int(prefix_len)
        self.prefix, self.self_mass, self.counts = {}, {}, {}

    def start_step(self, step_idx: int) -> None:
        return None

    def should_capture(self, layer_idx: int) -> bool:
        return True

    def update(self, attn_probs: torch.Tensor, *, layer_idx: int, block_type: str, prefix_len: int, action_len: int) -> None:
        mean = attn_probs.float().mean(dim=(0, 1, 2))
        self.prefix[layer_idx] = self.prefix.get(layer_idx, 0) + mean[:prefix_len].cpu()
        self.self_mass[layer_idx] = self.self_mass.get(layer_idx, 0.0) + float(mean[prefix_len:].sum())
        self.counts[layer_idx] = self.counts.get(layer_idx, 0) + 1

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        layers = sorted(self.prefix)
        prefix = torch.stack([self.prefix[i] / self.counts[i] for i in layers]).to(torch.float16).numpy()
        self_mass = np.array([self.self_mass[i] / self.counts[i] for i in layers], dtype=np.float32)
        return prefix, self_mass


class Policy:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        model_cfg = official._resolve_model_cfg(cfg)
        self.model = instantiate(model_cfg, model_dtype=torch.bfloat16, device="cuda")
        official._load_model_checkpoint(self.model, str(CHECKPOINT_DIR / "model.pt"))
        self.model = self.model.to("cuda").eval()
        self.processor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(CHECKPOINT_DIR / "dataset_stats.json")))
        self.token_mask = TokenMask()
        install_token_mask(self.model, self.token_mask)
        self.recorder = ActionAttentionRecorder()
        self.model.mot.action_attention_capture = self.recorder
        self.input_h, self.input_w = (int(v) for v in cfg.data.train.video_size)

    def _infer(self, image, proprio, prompt: str, drop: torch.Tensor | None) -> dict:
        self.token_mask.drop = drop
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            pred = self.model.infer_action(
                prompt=prompt,
                input_image=image,
                action_horizon=int(self.cfg.EVALUATION.action_horizon),
                negative_prompt=str(self.cfg.EVALUATION.get("negative_prompt", "")),
                text_cfg_scale=float(self.cfg.EVALUATION.get("text_cfg_scale", 1.0)),
                num_inference_steps=int(self.cfg.EVALUATION.num_inference_steps),
                proprio=proprio,
                sigma_shift=None,
                seed=int(self.cfg.seed),
                rand_device=str(self.cfg.EVALUATION.get("rand_device", "cpu")),
                tiled=False,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.token_mask.drop = None
        attention, self_mass = self.recorder.result()
        return {"action_norm": pred["action"].float().numpy(), "seconds": elapsed, "attention": attention,
                "attention_self": self_mass}

    def executable(self, action_norm: np.ndarray) -> np.ndarray:
        """Official post-processing (`_predict_action_chunk`): de-normalize, flip and binarize the gripper."""
        action = official._denormalize_action(torch.from_numpy(action_norm), self.processor)[0]
        action[..., -1] = action[..., -1] * 2 - 1
        action = invert_gripper_action(action)
        if bool(self.cfg.EVALUATION.get("binarize_gripper", False)):
            action[..., -1] = np.sign(action[..., -1])
        return action

    def act(self, obs: dict, task_description: str, drop: torch.Tensor | None, paired: bool) -> dict:
        prompt = DEFAULT_PROMPT.format(task=task_description)
        image, proprio, imgs = official._obs_to_model_input(
            obs, cfg=self.cfg, processor=self.processor, width=self.input_w, height=self.input_h,
            device="cuda", dtype=self.model.torch_dtype,
        )
        masked = self._infer(image, proprio, prompt, drop)
        record = {"imgs": imgs, "masked": masked, "executed": self.executable(masked["action_norm"])}
        if paired:
            record["unmasked"] = self._infer(image, proprio, prompt, None)
        return record


def run_episode(policy: Policy, env, initial_state, task_description: str, suite: str, fraction: float,
                rng: np.random.Generator) -> tuple[bool, list, dict]:
    cfg = policy.cfg
    max_steps = official._get_max_steps(suite)
    replan_steps = int(cfg.EVALUATION.replan_steps)
    num_steps_wait = int(cfg.EVALUATION.num_steps_wait)
    num_drop = int(round(fraction * IMAGE_TOKENS))
    env.reset()
    obs = env.set_init_state(initial_state)
    replay, pending = [], []
    calls = {key: [] for key in ("t", "proprio_raw", "sim_state", "token_mask", "action_masked", "action_unmasked",
                                 "executed", "attn_masked", "attn_unmasked", "attn_self_masked", "attn_self_unmasked",
                                 "seconds_masked", "seconds_unmasked")}
    t, done = 0, False
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue
        if not pending:
            sim_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            token_mask = np.zeros(IMAGE_TOKENS, dtype=bool)
            drop = None
            if num_drop > 0:
                indices = rng.permutation(IMAGE_TOKENS)[:num_drop]
                token_mask[indices] = True
                drop = torch.from_numpy(np.sort(indices)).long()
            record = policy.act(obs, task_description, drop, paired=num_drop > 0)
            calls["t"].append(t)
            calls["proprio_raw"].append(official._extract_sim_state(obs))
            calls["sim_state"].append(sim_state)
            calls["token_mask"].append(token_mask)
            calls["action_masked"].append(record["masked"]["action_norm"])
            calls["executed"].append(record["executed"])
            calls["attn_masked"].append(record["masked"]["attention"])
            calls["attn_self_masked"].append(record["masked"]["attention_self"])
            calls["seconds_masked"].append(record["masked"]["seconds"])
            if "unmasked" in record:
                calls["action_unmasked"].append(record["unmasked"]["action_norm"])
                calls["attn_unmasked"].append(record["unmasked"]["attention"])
                calls["attn_self_unmasked"].append(record["unmasked"]["attention_self"])
                calls["seconds_unmasked"].append(record["unmasked"]["seconds"])
            pending = record["executed"][:replan_steps].tolist()
            replay.append(record["imgs"].copy())
        else:
            replay.append(get_libero_image(obs).copy())
        obs, _, done, _ = env.step(pending.pop(0))
        if done:
            break
        t += 1
    calls["episode_steps"] = [t]
    return bool(done), replay, calls


def jobs() -> list[tuple[str, int, float]]:
    """(suite, task id, fraction) units, fractions interleaved so partial results stay balanced."""
    units = []
    for suite in SUITES:
        for task_id in range(10):
            for fraction in FRACTIONS:
                units.append((suite, task_id, fraction))
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--units", type=str, default=None, help="comma-separated suite:task:fraction subset")
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    args = parser.parse_args()
    cfg = eval_config()
    set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    units = jobs()
    if args.units:
        wanted = [u.split(":") for u in args.units.split(",")]
        units = [(s, int(t), float(f)) for s, t, f in wanted]
    units = units[args.worker_index :: args.num_workers]
    policy = Policy(cfg)
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite, task_id, fraction in units:
        task_suite = benchmark_dict[suite]()
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
        unit_dir = args.out_dir / suite / f"task{task_id:02d}"
        unit_dir.mkdir(parents=True, exist_ok=True)
        tag = f"frac{round(fraction * 1000):04d}"  # no decimal point: frac0000, frac0333, frac0667
        for trial in range(args.trials):
            name = f"trial{trial}_{tag}"
            json_path, npz_path = unit_dir / f"{name}.json", unit_dir / f"{name}.npz"
            if json_path.exists():
                continue
            seed = [SUITES.index(suite), task_id, trial]
            rng = np.random.default_rng(seed)
            start = time.time()
            success, replay, calls = run_episode(policy, env, initial_states[trial], task_description, suite,
                                                 fraction, rng)
            arrays = {key: np.stack(value) for key, value in calls.items() if value}
            arrays["token_mask"] = np.packbits(arrays["token_mask"], axis=1)
            np.savez_compressed(npz_path, **arrays)
            save_rollout_video(str(unit_dir), replay, name, success=success,
                               task_description=task_description)
            meta = {"suite": suite, "task_id": task_id, "task": task_description, "trial": trial,
                    "initial_state_index": trial, "mask_fraction": fraction,
                    "masked_tokens": int(round(fraction * IMAGE_TOKENS)), "image_tokens": IMAGE_TOKENS,
                    "success": success, "inference_calls": len(calls["t"]),
                    "episode_steps": int(calls["episode_steps"][0]), "mask_rng_seed": seed,
                    "wall_seconds": round(time.time() - start, 1), "worker": args.worker_index}
            json_path.write_text(json.dumps(meta, indent=1))
            print(json.dumps(meta), flush=True)
        env.close()


if __name__ == "__main__":
    main()
