"""Phase 12 checks for the Qwen-Image-2.1 ImageWAM stack on one GPU (run in `.venv`).

1. Prints the D2 attention mask of a small synthetic layout as a block matrix.
2. One training batch: loss terms finite; every trainable module group gets a non-zero gradient; the VAE gets none.
3. `infer_action` returns [16, 16].
4. With identical inputs, the action prediction from the cached t = 0 prefix equals a full joint forward over
   [text | reference | noisy target | action]; prints the maximum absolute difference.
"""

import argparse
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

from imagewam.models.backbones.qwenimage21_stack import QwenImage21Stack
from imagewam.runtime import build_datasets
from imagewam.utils import misc

DATASETS = ("drawer", "biomanual", "simple-pick", "Stack-the-cubes")


def print_mask() -> None:
    text_mask = torch.tensor([[True, True, False]])
    mask = QwenImage21Stack.attention_mask(text_mask, ref_len=2, target_len=2, action_len=2)[0].int()
    labels = ["T0", "T1", "Tp", "R0", "R1", "X0", "X1", "A0", "A1"]
    print("D2 mask (rows attend to columns; Tp = padded text):")
    print("     " + " ".join(f"{label:>2}" for label in labels))
    for label, row in zip(labels, mask.tolist()):
        print(f"  {label:>2} " + " ".join(f"{v:>2}" for v in row))


def group_of(name: str) -> str:
    parts = name.split(".")
    if parts[0] in ("video_expert", "action_expert"):
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--skip-train", action="store_true", help="only run checks 3 and 4")
    args = parser.parse_args()
    torch.manual_seed(0)
    print_mask()

    root = args.project_root
    misc.register_work_dir(str(args.work_dir))
    dirs = "[" + ",".join(str(root / "dataset/LGG100" / name) for name in DATASETS) + "]"
    model_root = root / "BaseModel/Qwen-Image-2.1"
    cache = root / "ImageWAM/data/g1_lgg100/qwenimage21_text_cache"
    with initialize_config_dir(config_dir=str(root / "ImageWAM/configs"), version_base="1.3"):
        cfg = compose(
            "train",
            overrides=[
                "task=g1_lgg100_qwenimage21_imagewam",
                f"data.train.dataset_dirs={dirs}",
                f"data.val.dataset_dirs={dirs}",
                f"data.train.qwen_text_cache_dir={cache}",
                f"data.val.qwen_text_cache_dir={cache}",
                f"data.train.pretrained_norm_stats={args.norm_stats}",
                "model.proprio_dim=16",
                f"model.transformer_dir={model_root / 'transformer'}",
                f"model.vae_dir={model_root / 'vae'}",
                f"model.action_dit_pretrained_path={root / 'ImageWAM/checkpoints/action_dit_qwenimage21_g1_init.pt'}",
            ],
        )
    _, val_ds = build_datasets(cfg.data)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = instantiate(cfg.model, model_dtype=dtype, device="cuda")
    # Trainer policy (trainer.py): everything frozen, then MoT and the proprio encoder trainable.
    model.requires_grad_(False)
    model.dit.train()
    model.dit.requires_grad_(True)
    model.proprio_encoder.train()
    model.proprio_encoder.requires_grad_(True)
    stride = len(val_ds) // args.batch_size
    batch = default_collate([val_ds[i * stride] for i in range(args.batch_size)])
    if not args.skip_train:
        train_check(model, batch)
    model.eval()
    inference_checks(model, default_collate([val_ds[0]]))


def train_check(model, batch) -> None:
    loss, terms = model.training_loss(batch)
    loss.backward()
    print(f"loss={loss.item():.4f} terms={terms} finite={bool(torch.isfinite(loss))}")
    stats: dict[str, list[float]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        g = stats.setdefault(group_of(name), [0, 0, 0.0])
        g[0] += 1
        if param.grad is not None and bool(param.grad.abs().sum() > 0):
            g[1] += 1
            g[2] += float(param.grad.float().pow(2).sum())
    for group, (count, nonzero, sq) in sorted(stats.items()):
        print(f"  grad {group:28s} tensors={count:4d} nonzero={nonzero:4d} norm={sq ** 0.5:.4e}")
    vae_grads = sum(p.grad is not None for p in model.vae.parameters())
    print(f"  VAE params with grad: {vae_grads}; VAE requires_grad: {any(p.requires_grad for p in model.vae.parameters())}")
    print(f"  peak memory after backward: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    model.zero_grad(set_to_none=True)


def inference_checks(model, sample) -> None:
    context, context_mask = sample["text_hidden_states"][0], sample["text_attention_mask"][0]
    proprio = sample["proprio"][0, 0]
    image = sample["video"][0][:, 0].unsqueeze(0)
    out = model.infer_action(
        prompt=None, input_image=image, action_horizon=16, proprio=proprio,
        context=context, context_mask=context_mask, num_inference_steps=10, seed=0,
    )
    print(f"infer_action: shape={tuple(out['action'].shape)} finite={bool(torch.isfinite(out['action']).all())}")

    stack = QwenImage21Stack(model)
    with torch.no_grad():
        inputs = stack.build_inputs(sample)
        noise = torch.randn_like(inputs.target_tokens)
        noisy_target = 0.4 * inputs.target_tokens + 0.6 * noise
        noisy_action = torch.randn(1, 16, 16, device=model.device, dtype=model.torch_dtype)
        t_action = torch.tensor([0.7], device=model.device)
        # Cached path.
        kv, prefix_len, positions = stack.prefill_prefix(inputs, 16)
        text_len = int(inputs.text_states.shape[1])
        action_pre = model.action_expert.pre_dit(noisy_action, t_action)
        cached = model.mot.forward_action_with_video_cache(
            action_tokens=action_pre.tokens, action_freqs=positions.action, action_t_mod=action_pre.modulation,
            action_context_payload=None, video_kv_cache=kv,
            attention_mask=stack.attention_mask(inputs.text_mask, prefix_len - text_len, 0, 16), video_seq_len=prefix_len,
        )
        pred_cached = model.action_expert.post_dit(cached, action_pre)
        # Full joint forward, with the noisy target present (training layout).
        video_pre = model.video_expert.pre_dit(
            target_latents=noisy_target, video_timestep=torch.tensor([0.6], device=model.device),
            text_states=inputs.text_states, text_mask=inputs.text_mask, ref_latents=inputs.ref_tokens, positions=positions,
        )
        mask = stack.attention_mask(inputs.text_mask, video_pre.ref_len, video_pre.target_len, 16)
        full = model.mot(
            embeds_all={"video": video_pre.tokens, "action": action_pre.tokens}, attention_mask=mask,
            freqs_all={"video": video_pre.rotary, "action": positions.action}, context_all={},
            t_mod_all={"video": video_pre.modulation, "action": action_pre.modulation},
        )
        pred_full = model.action_expert.post_dit(full["action"], action_pre)
    diff = (pred_cached.float() - pred_full.float()).abs().max().item()
    print(f"cached-prefix vs full joint forward: max abs diff {diff:.4e} (pred max abs {pred_full.float().abs().max().item():.3f})")


if __name__ == "__main__":
    main()
