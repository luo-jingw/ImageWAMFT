"""LIBERO rollouts of ImageWAM-FLUX.2-4B with fewer computed reference-image tokens (run in `.venv-libero`, see PLAN.md).

The configuration, observation preprocessing, action post-processing and episode loop follow
`ImageWAM/experiments/libero/eval_libero_single.py` and `scripts/flux2/run_eval_flux2_libero.sh`; the official helpers
are imported and called directly. Strategies, applied to the 392 reference-image tokens (14 x 28 grid):
  - random:    a random subset, resampled at every inference call, is removed from every attention row (prefix
               prefill and action denoising) by disabling its column in the FLUX.2 MoT attention mask;
  - grid:      the same removal with a fixed regular pattern ((row + col) % 3 or % 2);
  - tome:      ToMe bipartite soft matching (research/third_party/ToMe) on the VAE latent tokens merges tokens into
               size-weighted averages, positions averaged the same way, with ToMe proportional attention (+log size
               on the merged tokens' attention logits); the sequence gets shorter;
  - kv_random: a random subset, resampled per call, is not recomputed: in every prefix layer its keys and values are
               taken from the per-episode K/V canvas, which holds each token's K/V from the latest call that computed
               it. The first call of an episode computes every token;
  - kv_topk:   only the tokens with the most action-to-image attention at the previous call (mean over layers) are
               recomputed; all others come from the canvas;
  - kv_randk:  as kv_topk with a random recomputed subset of the same size;
  - attn_topk: per call, a scoring pass on the full input (prefix prefill and the first action denoising step, which
               equals step 1 of the 10-step schedule) gives each token's action-to-image attention (mean over layers,
               heads and action queries); the n lowest-scored tokens are removed as in `random`;
  - evit:      EViT token reorganization with the same score: the n lowest-scored tokens are fused into one token,
               the score-weighted average of their VAE latents and positions, appended after the kept tokens (no
               proportional attention);
  - tofu:      Token Fusion MLERP merging: ToMe matching and proportional attention, but each merged token is rescaled
               to the largest norm among the tokens it merges.
A condition is `strategy:n`, n the number of removed, merged-away or reused tokens. Recorded per inference call: a
paired inference from the same observation and initial noise without reduction (not executed), the action-to-prefix
attention per layer, the MuJoCo state, the reduction (token mask or merge groups), both action chunks, the executed
actions and the wall time.

  python eval_token_mask.py --out-dir <dir> --conditions random:196,kv_topk:383 --worker-index i --num-workers N
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGEWAM = PROJECT_ROOT / "ImageWAM"
FLUX2_SRC = IMAGEWAM / "third_party" / "flux2"
TOME_SRC = PROJECT_ROOT / "research" / "third_party" / "ToMe"
for path in (FLUX2_SRC, FLUX2_SRC / "src", IMAGEWAM / "experiments" / "libero", IMAGEWAM / "src", IMAGEWAM, TOME_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg  # noqa: E402

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
NUM_TRIALS = 5
IMAGE_TOKENS = 392  # 224 x 448 input, 16 x 16 pixels per token
GRID_H, GRID_W = 14, 28
STRATEGIES = ("random", "grid", "tome", "kv_random", "kv_topk", "kv_randk", "attn_topk", "evit", "tofu")
KV_STRATEGIES = ("kv_random", "kv_topk", "kv_randk")
SCORED_STRATEGIES = ("attn_topk", "evit")  # need the per-call attention score
TOME_ROUNDS = {131: (131,), 196: (196,), 261: (196, 65)}  # ToMe merges at most half of the tokens per round
DEFAULT_CONDITIONS = ("random:196,grid:131,grid:196,grid:261,tome:131,tome:196,tome:261,"
                      "kv_random:131,kv_random:261,kv_topk:383,kv_randk:383")


def grid_drop_indices(removed: int) -> np.ndarray:
    """Fixed regular patterns on the token grid: 131 removes (r+c)%3==0, 196 the checkerboard (r+c)%2==0, 261 keeps
    only (r+c)%3==0."""
    rows, cols = np.divmod(np.arange(IMAGE_TOKENS), GRID_W)
    diagonal = rows + cols
    pattern = {131: diagonal % 3 == 0, 196: diagonal % 2 == 0, 261: diagonal % 3 != 0}[removed]
    return np.nonzero(pattern)[0]


def parse_condition(text: str) -> tuple[str, int]:
    strategy, count = text.split(":")
    removed = int(count)
    if strategy not in STRATEGIES or not 0 <= removed < IMAGE_TOKENS:
        raise ValueError(f"bad condition {text!r}")
    if strategy in ("grid", "tome", "tofu") and removed not in TOME_ROUNDS:
        raise ValueError(f"{strategy} supports {sorted(TOME_ROUNDS)} tokens, got {removed}")
    return strategy, removed


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
class Reduction:
    """How the reference-image tokens of one inference are reduced; the default reduces nothing."""

    drop: torch.Tensor | None = None  # token indices hidden from every attention row
    merge_rounds: tuple[int, ...] = ()  # ToMe tokens merged per round
    merge_mode: str = "mean"  # "mean" (ToMe size-weighted average) or "mlerp" (Token Fusion norm-preserving)
    fuse_drop: torch.Tensor | None = None  # EViT: token indices fused into one appended token
    fuse_weights: torch.Tensor | None = None  # EViT: fusion weights of fuse_drop (the attention score)
    kv_reuse: bool = False  # read stale tokens from the K/V canvas and store this call's image K/V in it
    stale: torch.Tensor | None = None  # token indices whose K/V come from the canvas


@dataclass
class MergeState:
    """Outcome of the last ToMe merge: token sizes [M] and the merged group of every original token [392]."""

    size: torch.Tensor | None = None
    group: np.ndarray | None = None
    proportional: bool = True  # add log(size) to the merged tokens' attention logits


@dataclass
class KVCanvas:
    """Per prefix layer, the K/V of the 392 image tokens as last computed in this episode."""

    k: list = field(default_factory=list)
    v: list = field(default_factory=list)

    def reset(self) -> None:
        self.k, self.v = [], []


def install_token_mask(model, active: list[Reduction]) -> None:
    original = model._build_mot_attention_mask_flux2

    def build(*args, **kwargs):
        masks = original(*args, **kwargs)
        drop = active[0].drop
        if drop is not None:
            if int(kwargs["cond_len"]) != IMAGE_TOKENS:
                raise ValueError(f"expected {IMAGE_TOKENS} reference tokens, got {kwargs['cond_len']}")
            columns = int(kwargs["txt_len"]) + drop.to(kwargs["device"])
            for mask in masks.values():
                mask[:, :, columns] = False
        return masks

    model._build_mot_attention_mask_flux2 = build


def install_token_merge(model, active: list[Reduction], state: MergeState) -> None:
    """ToMe on the reference latent tokens, plus proportional attention through a float attention mask."""
    encode_original = model._encode_flux2_image_tokens

    def encode(image, *, time_value):
        tokens, ids = encode_original(image, time_value=time_value)
        state.size, state.group, state.proportional = None, None, True
        reduction = active[0]
        if time_value != 10.0:  # 10.0 marks the reference image in infer_action_flux2
            return tokens, ids
        if reduction.fuse_drop is not None:
            drop = reduction.fuse_drop.to(tokens.device)
            keep = torch.ones(tokens.shape[1], dtype=torch.bool, device=tokens.device)
            keep[drop] = False
            weights = reduction.fuse_weights.to(device=tokens.device, dtype=torch.float32)
            weights = (weights / weights.sum())[:, None]
            fused_x = (tokens[0, drop].float() * weights).sum(dim=0, keepdim=True)
            fused_pos = (ids[0, drop].float() * weights).sum(dim=0, keepdim=True)
            x = torch.cat([tokens[0, keep].float(), fused_x])[None]
            pos = torch.cat([ids[0, keep].float(), fused_pos])[None]
            kept = int(keep.sum())
            group = np.full(tokens.shape[1], kept, dtype=np.int16)
            group[keep.cpu().numpy()] = np.arange(kept)
            state.size = torch.cat([torch.ones(kept), torch.tensor([float(len(drop))])]).to(tokens.device)
            state.group, state.proportional = group, False
            return x.to(tokens.dtype), pos.to(ids.dtype)
        rounds = reduction.merge_rounds
        if not rounds:
            return tokens, ids
        x, pos, size, source = tokens.float(), ids.float(), None, None
        for r in rounds:
            merge, _ = bipartite_soft_matching(x, r)
            source = merge_source(merge, x, source)
            if reduction.merge_mode == "mlerp":
                max_norm = merge(x.norm(dim=-1, keepdim=True), mode="amax")
            pos, _ = merge_wavg(merge, pos, size)
            x, size = merge_wavg(merge, x, size)
            if reduction.merge_mode == "mlerp":
                x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6) * max_norm
        state.size = size[0, :, 0]
        state.group = source[0].argmax(dim=0).cpu().numpy().astype(np.int16)
        return x.to(tokens.dtype), pos.to(ids.dtype)

    model._encode_flux2_image_tokens = encode
    mask_original = model._build_mot_attention_mask_flux2

    def build(*args, **kwargs):
        masks = mask_original(*args, **kwargs)
        if state.size is None or not state.proportional:
            return masks
        txt_len, cond_len = int(kwargs["txt_len"]), int(kwargs["cond_len"])
        if cond_len != state.size.numel():
            raise ValueError(f"merged {state.size.numel()} reference tokens, mask expects {cond_len}")
        bias = torch.log(state.size.to(device=kwargs["device"], dtype=torch.float32))
        out = {}
        for key, mask in masks.items():
            additive = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
            additive = additive.masked_fill(~mask, float("-inf"))
            additive[:, :, txt_len:txt_len + cond_len] += bias
            out[key] = additive
        return out

    model._build_mot_attention_mask_flux2 = build
    mot = model.mot
    attention_original = mot._mixed_attention

    def mixed(q_cat, k_cat, v_cat, attention_mask, return_attn_probs=False):
        if attention_mask.dtype == torch.bool:
            return attention_original(q_cat, k_cat, v_cat, attention_mask, return_attn_probs=return_attn_probs)
        if mot.num_kv_heads != mot.num_heads:
            raise ValueError("float attention masks are implemented for equal query and key/value heads")
        batch, q_len, _ = q_cat.shape
        k_len = k_cat.shape[1]
        heads, dim = mot.num_heads, mot.attn_head_dim
        bias = attention_mask if attention_mask.ndim == 4 else attention_mask.unsqueeze(1)
        q = q_cat.view(batch, q_len, heads, dim).transpose(1, 2)
        k = k_cat.view(batch, k_len, heads, dim).transpose(1, 2)
        v = v_cat.view(batch, k_len, heads, dim).transpose(1, 2)
        if return_attn_probs:
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * dim ** -0.5 + bias.float()
            probs = torch.softmax(scores, dim=-1).to(v.dtype)
            out = torch.matmul(probs, v).transpose(1, 2).reshape(batch, q_len, heads * dim)
            return out, probs
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return out.transpose(1, 2).reshape(batch, q_len, heads * dim)

    mot._mixed_attention = mixed


def install_kv_reuse(model, active: list[Reduction], canvas: KVCanvas) -> None:
    """During a prefix prefill with `kv_reuse`, each layer's image keys and values of the stale tokens are replaced by
    the canvas entries before attention, in place, so the cached prefix the action expert reads carries them too; then
    the layer's image K/V are stored as the new canvas. The stale tokens' own hidden states only feed their own,
    overwritten K/V, so this equals not computing them."""
    mot = model.mot
    prefill_original = mot.prefill_flux2_video_cache
    attention_inner = mot._mixed_attention
    layer = [None]

    def prefill(*args, **kwargs):
        if not active[0].kv_reuse:
            return prefill_original(*args, **kwargs)
        layer[0] = 0
        try:
            return prefill_original(*args, **kwargs)
        finally:
            layer[0] = None

    def mixed(q_cat, k_cat, v_cat, attention_mask, return_attn_probs=False):
        index = layer[0]
        if index is not None:
            layer[0] = index + 1
            start = k_cat.shape[1] - IMAGE_TOKENS  # prefix order is [text + proprio, image]
            stale = active[0].stale
            if stale is not None and index < len(canvas.k):
                stale = stale.to(k_cat.device)
                k_cat[:, start + stale] = canvas.k[index][:, stale]
                v_cat[:, start + stale] = canvas.v[index][:, stale]
            k_img, v_img = k_cat[:, start:].clone(), v_cat[:, start:].clone()
            if index < len(canvas.k):
                canvas.k[index], canvas.v[index] = k_img, v_img
            else:
                canvas.k.append(k_img)
                canvas.v.append(v_img)
        return attention_inner(q_cat, k_cat, v_cat, attention_mask, return_attn_probs=return_attn_probs)

    mot.prefill_flux2_video_cache = prefill
    mot._mixed_attention = mixed


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


@dataclass
class Prepared:
    """Model input of one observation."""

    image: torch.Tensor
    proprio: torch.Tensor
    prompt: str
    imgs: np.ndarray  # the frames saved in the rollout video


class Policy:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        model_cfg = official._resolve_model_cfg(cfg)
        self.model = instantiate(model_cfg, model_dtype=torch.bfloat16, device="cuda")
        official._load_model_checkpoint(self.model, str(CHECKPOINT_DIR / "model.pt"))
        self.model = self.model.to("cuda").eval()
        self.processor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(CHECKPOINT_DIR / "dataset_stats.json")))
        self.active = [Reduction()]
        self.merge_state = MergeState()
        self.canvas = KVCanvas()
        install_token_mask(self.model, self.active)
        install_token_merge(self.model, self.active, self.merge_state)
        install_kv_reuse(self.model, self.active, self.canvas)
        self.recorder = ActionAttentionRecorder()
        self.model.mot.action_attention_capture = self.recorder
        self.input_h, self.input_w = (int(v) for v in cfg.data.train.video_size)

    def _infer(self, image, proprio, prompt: str, reduction: Reduction, steps: int | None = None) -> dict:
        self.active[0] = reduction
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            pred = self.model.infer_action(
                prompt=prompt,
                input_image=image,
                action_horizon=int(self.cfg.EVALUATION.action_horizon),
                negative_prompt=str(self.cfg.EVALUATION.get("negative_prompt", "")),
                text_cfg_scale=float(self.cfg.EVALUATION.get("text_cfg_scale", 1.0)),
                num_inference_steps=int(steps or self.cfg.EVALUATION.num_inference_steps),
                proprio=proprio,
                sigma_shift=None,
                seed=int(self.cfg.seed),
                rand_device=str(self.cfg.EVALUATION.get("rand_device", "cpu")),
                tiled=False,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.active[0] = Reduction()
        attention, self_mass = self.recorder.result()
        out = {"action_norm": pred["action"].float().numpy(), "seconds": elapsed, "attention": attention,
               "attention_self": self_mass}
        if reduction.merge_rounds or reduction.fuse_drop is not None:
            out["merge_group"] = self.merge_state.group
            out["merge_size"] = self.merge_state.size.cpu().numpy().astype(np.float32)
        return out

    def executable(self, action_norm: np.ndarray) -> np.ndarray:
        """Official post-processing (`_predict_action_chunk`): de-normalize, flip and binarize the gripper."""
        action = official._denormalize_action(torch.from_numpy(action_norm), self.processor)[0]
        action[..., -1] = action[..., -1] * 2 - 1
        action = invert_gripper_action(action)
        if bool(self.cfg.EVALUATION.get("binarize_gripper", False)):
            action[..., -1] = np.sign(action[..., -1])
        return action

    def prepare(self, obs: dict, task_description: str) -> Prepared:
        image, proprio, imgs = official._obs_to_model_input(
            obs, cfg=self.cfg, processor=self.processor, width=self.input_w, height=self.input_h,
            device="cuda", dtype=self.model.torch_dtype,
        )
        return Prepared(image=image, proprio=proprio, prompt=DEFAULT_PROMPT.format(task=task_description), imgs=imgs)

    def score(self, prepared: Prepared) -> np.ndarray:
        """Action-to-image attention of the first denoising step on the full input, mean over layers [392]."""
        scored = self._infer(prepared.image, prepared.proprio, prepared.prompt, Reduction(), steps=1)
        return scored["attention"][:, -IMAGE_TOKENS:].astype(np.float32).mean(axis=0)

    def act(self, prepared: Prepared, reduction: Reduction, paired: bool) -> dict:
        masked = self._infer(prepared.image, prepared.proprio, prepared.prompt, reduction)
        record = {"imgs": prepared.imgs, "masked": masked, "executed": self.executable(masked["action_norm"])}
        if paired:
            record["unmasked"] = self._infer(prepared.image, prepared.proprio, prepared.prompt, Reduction())
        return record


def call_reduction(strategy: str, removed: int, rng: np.random.Generator, first_call: bool,
                   previous_attention: np.ndarray | None,
                   score: np.ndarray | None = None) -> tuple[Reduction, np.ndarray]:
    """The reduction of one inference call and its 392-token mask (True = removed, fused or reused; all False for
    ToMe-style merges, no reduction and the first call of a K/V-reuse episode, which computes every token)."""
    token_mask = np.zeros(IMAGE_TOKENS, dtype=bool)
    if removed == 0:
        return Reduction(), token_mask
    if strategy == "tome":
        return Reduction(merge_rounds=TOME_ROUNDS[removed]), token_mask
    if strategy == "tofu":
        return Reduction(merge_rounds=TOME_ROUNDS[removed], merge_mode="mlerp"), token_mask
    if strategy in SCORED_STRATEGIES:
        indices = np.sort(np.argsort(-score, kind="stable")[IMAGE_TOKENS - removed:])  # the lowest-scored tokens
        token_mask[indices] = True
        selected = torch.from_numpy(indices).long()
        if strategy == "evit":
            return Reduction(fuse_drop=selected, fuse_weights=torch.from_numpy(score[indices])), token_mask
        return Reduction(drop=selected), token_mask
    if strategy in KV_STRATEGIES and first_call:
        return Reduction(kv_reuse=True), token_mask
    if strategy == "grid":
        indices = grid_drop_indices(removed)
    elif strategy == "kv_topk":
        image_attention = previous_attention[:, -IMAGE_TOKENS:].astype(np.float32).mean(axis=0)
        indices = np.argsort(-image_attention, kind="stable")[IMAGE_TOKENS - removed:]
    else:
        indices = rng.permutation(IMAGE_TOKENS)[:removed]
    token_mask[indices] = True
    selected = torch.from_numpy(np.sort(indices)).long()
    if strategy in KV_STRATEGIES:
        return Reduction(kv_reuse=True, stale=selected), token_mask
    return Reduction(drop=selected), token_mask


def run_episode(policy: Policy, env, initial_state, task_description: str, suite: str, strategy: str, removed: int,
                rng: np.random.Generator) -> tuple[bool, list, dict]:
    cfg = policy.cfg
    max_steps = official._get_max_steps(suite)
    replan_steps = int(cfg.EVALUATION.replan_steps)
    num_steps_wait = int(cfg.EVALUATION.num_steps_wait)
    env.reset()
    obs = env.set_init_state(initial_state)
    policy.canvas.reset()
    replay, pending = [], []
    calls = {key: [] for key in ("t", "proprio_raw", "sim_state", "token_mask", "score", "merge_group", "merge_size",
                                 "action_masked", "action_unmasked", "executed", "attn_masked", "attn_unmasked",
                                 "attn_self_masked", "attn_self_unmasked", "seconds_masked", "seconds_unmasked")}
    t, done = 0, False
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue
        if not pending:
            sim_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            previous = calls["attn_masked"][-1] if calls["attn_masked"] else None
            prepared = policy.prepare(obs, task_description)
            score = policy.score(prepared) if strategy in SCORED_STRATEGIES and removed > 0 else None
            reduction, token_mask = call_reduction(strategy, removed, rng, not calls["t"], previous, score)
            record = policy.act(prepared, reduction, paired=removed > 0)
            if score is not None:
                calls["score"].append(score.astype(np.float16))
            calls["t"].append(t)
            calls["proprio_raw"].append(official._extract_sim_state(obs))
            calls["sim_state"].append(sim_state)
            calls["token_mask"].append(token_mask)
            calls["action_masked"].append(record["masked"]["action_norm"])
            calls["executed"].append(record["executed"])
            calls["attn_masked"].append(record["masked"]["attention"])
            calls["attn_self_masked"].append(record["masked"]["attention_self"])
            calls["seconds_masked"].append(record["masked"]["seconds"])
            if "merge_group" in record["masked"]:
                calls["merge_group"].append(record["masked"]["merge_group"])
                calls["merge_size"].append(record["masked"]["merge_size"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--conditions", type=str, default=DEFAULT_CONDITIONS, help="comma-separated strategy:n")
    parser.add_argument("--suites", type=str, default=",".join(SUITES))
    parser.add_argument("--tasks", type=str, default="0-9", help="task id range a-b")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    args = parser.parse_args()
    conditions = [parse_condition(c) for c in args.conditions.split(",") if c]
    first, last = (int(v) for v in args.tasks.split("-"))
    # Conditions interleaved inside each task so partial results stay balanced across conditions.
    units = [(strategy, removed, suite, task_id) for suite in args.suites.split(",")
             for task_id in range(first, last + 1) for strategy, removed in conditions]
    units = units[args.worker_index :: args.num_workers]
    cfg = eval_config()
    set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    policy = Policy(cfg)
    benchmark_dict = benchmark.get_benchmark_dict()
    for strategy, removed, suite, task_id in units:
        task_suite = benchmark_dict[suite]()
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
        unit_dir = args.out_dir / strategy / suite / f"task{task_id:02d}"
        unit_dir.mkdir(parents=True, exist_ok=True)
        for trial in range(args.trials):
            name = f"trial{trial}_rm{removed:03d}"
            json_path, npz_path = unit_dir / f"{name}.json", unit_dir / f"{name}.npz"
            if json_path.exists():
                continue
            seed = [SUITES.index(suite), task_id, trial]
            rng = np.random.default_rng(seed)
            start = time.time()
            success, replay, calls = run_episode(policy, env, initial_states[trial], task_description, suite,
                                                 strategy, removed, rng)
            arrays = {key: np.stack(value) for key, value in calls.items() if value}
            arrays["token_mask"] = np.packbits(arrays["token_mask"], axis=1)
            np.savez_compressed(npz_path, **arrays)
            save_rollout_video(str(unit_dir), replay, name, success=success, task_description=task_description)
            meta = {"strategy": strategy, "suite": suite, "task_id": task_id, "task": task_description,
                    "trial": trial, "initial_state_index": trial, "mask_fraction": round(removed / IMAGE_TOKENS, 4),
                    "masked_tokens": removed, "kept_tokens": IMAGE_TOKENS - removed, "image_tokens": IMAGE_TOKENS,
                    "success": success, "inference_calls": len(calls["t"]),
                    "episode_steps": int(calls["episode_steps"][0]), "mask_rng_seed": seed,
                    "wall_seconds": round(time.time() - start, 1), "worker": args.worker_index}
            json_path.write_text(json.dumps(meta, indent=1))
            print(json.dumps(meta), flush=True)
        env.close()


if __name__ == "__main__":
    main()
