"""Open-loop screen of test-time registers on restored LIBERO states (run in `.venv-libero`, see PLAN.md).

States: the 60 evaluation states of the quantization screen (tasks 0-4 of every suite, calls 1, 3 and 6 of the
no-reduction episodes of the token-mask study). Per config and state, against the unmodified model with the same
observation and initial noise:
  - mean absolute action deviation (normalized 16 x 7 chunk) and gripper flips over the 12 executed steps;
  - attention per layer: share of the reference-image queries (prefix pass) and of the action queries (first denoising
    step) on the template, task, tail, proprio, register and image keys (and action keys for action queries);
  - residual-stream norm entering the first single block, per token group.
`none` repeats the unmodified model (determinism floor).

  python screen_register.py --out runs/screen.json [--configs none,ttr_dir_r1,...]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "libero_quant"))
from eval_quant import build_policy, infer_actions  # noqa: E402
from screen_quant import EVAL_CALLS, EVAL_TASKS, gripper_flip, restored_inputs  # noqa: E402

import register as R  # noqa: E402

TEMPLATE, TAIL, IMAGE = 18, 9, 392
GROUPS = ("template", "task", "tail", "proprio", "register", "image", "action")


def group_slices(n_text: int, registers: int, txt_len: int) -> dict[str, slice]:
    return {"template": slice(0, TEMPLATE), "task": slice(TEMPLATE, n_text - TAIL), "tail": slice(n_text - TAIL, n_text),
            "proprio": slice(n_text, n_text + 1), "register": slice(n_text + 1, n_text + 1 + registers),
            "image": slice(txt_len, txt_len + IMAGE), "action": slice(txt_len + IMAGE, None)}


class Probe:
    """Attention shares and residual norms of one inference. Image-query attention is computed in the prefix pass
    (square video-expert attention calls); action-query attention through the MoT capture interface (step 0)."""

    def __init__(self, policy):
        self.mot = policy.model.mot
        self.active, self.slices, self.step = False, None, -1
        self.image_rows, self.action_rows, self.norms = [], [], {}
        original = self.mot._mixed_attention

        def mixed_attention(q_cat, k_cat, v_cat, attention_mask, return_attn_probs=False):
            out = original(q_cat, k_cat, v_cat, attention_mask, return_attn_probs=return_attn_probs)
            if self.active and q_cat.shape[1] == k_cat.shape[1]:
                self._image_queries(q_cat, k_cat, attention_mask)
            return out

        self.mot._mixed_attention = mixed_attention
        first_single = self.mot.mixtures["video"].single_blocks[0]
        original_qkv = first_single._qkv

        def qkv(x, *args, **kwargs):
            if self.active:
                norm = x[0].float().norm(dim=-1)
                self.norms = {g: float(norm[s].median()) for g, s in self.slices.items()
                              if g != "action" and s.stop != s.start}
            return original_qkv(x, *args, **kwargs)

        first_single._qkv = qkv

    def _shares(self, probs: torch.Tensor) -> dict[str, float]:  # probs [Lk], mean over heads and queries
        return {g: float(probs[s].sum()) for g, s in self.slices.items()}

    def _image_queries(self, q_cat, k_cat, attention_mask):
        mot, L = self.mot, q_cat.shape[1]
        H, D = mot.num_heads, mot.attn_head_dim
        mask = mot._format_attention_mask(attention_mask, 1, L, L, q_cat.device)[0, 0]  # [L, L]
        rows = self.slices["image"]
        q = q_cat[0, rows].view(-1, H, D).transpose(0, 1).float()
        k = k_cat[0].view(L, H, D).transpose(0, 1).float()
        scores = (q @ k.transpose(-2, -1)) * D ** -0.5
        probs = torch.softmax(scores.masked_fill(~mask[rows], float("-inf")), dim=-1)
        self.image_rows.append(self._shares(probs.mean(dim=(0, 1))))

    # MoT action_attention_capture interface
    def configure(self, *, condition_slice, condition_grid, prefix_len, action_len, metadata):
        self.step = -1

    def start_step(self, step_idx):
        self.step = step_idx

    def should_capture(self, layer_idx):
        return self.active and self.step == 0

    def update(self, attn_probs, *, layer_idx, block_type, prefix_len, action_len):
        self.action_rows.append(self._shares(attn_probs[0].float().mean(dim=(0, 1))))

    @torch.no_grad()
    def run(self, policy, prepared, registers):
        model = policy.model
        _, mask = model._prepare_flux2_infer_text(prepared.prompt, None, None)
        n_text = int(mask[0].sum())
        self.slices = group_slices(n_text, registers, mask.shape[1] + 1 + registers)
        self.image_rows, self.action_rows, self.active = [], [], True
        model.mot.action_attention_capture = self
        try:
            infer_actions(policy, prepared)
        finally:
            self.active = False
            model.mot.action_attention_capture = None
        if len(self.image_rows) != 25 or len(self.action_rows) != 25:
            raise RuntimeError(f"captured {len(self.image_rows)} prefix and {len(self.action_rows)} action layers")
        return self.image_rows, self.action_rows, self.norms


def mean_rows(rows_per_state: list[list[dict]]) -> dict[str, list[float]]:
    return {g: [float(np.mean([rows[layer][g] for rows in rows_per_state])) for layer in range(25)] for g in GROUPS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", type=str, default="none," + ",".join(R.configs()))
    args = parser.parse_args()
    grid = {"none": R.RegisterConfig("none"), **R.configs()}
    policy = build_policy()
    probe = Probe(policy)
    evaluation = restored_inputs(policy, EVAL_TASKS, EVAL_CALLS)
    reference = {key: infer_actions(policy, prepared)[0] for key, prepared in evaluation}
    report = {"states": len(evaluation), "neuron_counts": R.neuron_counts(), "configs": {}}
    if args.out.exists():
        report["configs"] = json.loads(args.out.read_text())["configs"]
    for name in args.configs.split(","):
        if name in report["configs"]:
            continue
        config = grid[name]
        handle = R.install(policy.model, config) if name != "none" else None
        start = time.perf_counter()
        rows, image_rows, action_rows, norms = [], [], [], []
        for key, prepared in evaluation:
            action, _ = infer_actions(policy, prepared)
            full = reference[key]
            rows.append({"state": key, "deviation": float(np.abs(action - full).mean()),
                         "gripper_flip": gripper_flip(policy, action, full), "finite": bool(np.isfinite(action).all())})
            image, act, norm = probe.run(policy, prepared, config.registers)
            image_rows.append(image), action_rows.append(act), norms.append(norm)
        if handle is not None:
            handle.remove()
        dev = np.array([r["deviation"] for r in rows])
        report["configs"][name] = {
            "config": vars(config), "deviation_mean": float(dev.mean()), "deviation_median": float(np.median(dev)),
            "deviation_p90": float(np.percentile(dev, 90)),
            "gripper_flip": float(np.mean([r["gripper_flip"] for r in rows])),
            "finite": all(r["finite"] for r in rows), "seconds": time.perf_counter() - start,
            "image_query_attention": mean_rows(image_rows), "action_query_attention": mean_rows(action_rows),
            "first_single_norm": {g: float(np.median([n[g] for n in norms if g in n])) for g in norms[0]},
            "rows": rows}
        c = report["configs"][name]
        img, act = c["image_query_attention"], c["action_query_attention"]
        print(f"{name:14s} deviation mean {c['deviation_mean']:.4f} median {c['deviation_median']:.4f} | "
              f"flips {c['gripper_flip']:.3f} | image queries, layers 5-10: template {np.mean(img['template'][5:11]):.3f} "
              f"register {np.mean(img['register'][5:11]):.3f} | action queries: template {np.mean(act['template']):.3f} "
              f"register {np.mean(act['register']):.3f} | norms {json.dumps({g: round(v) for g, v in c['first_single_norm'].items()})}",
              flush=True)
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
