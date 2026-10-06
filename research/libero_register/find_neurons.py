"""Rank the register neurons of the video expert (run in `.venv-libero`, see PLAN.md).

Calibration: 40 restored states (tasks 5-9 of every suite, calls 0 and 4; disjoint from the screen states). For every
valid text token p (text + proprio), the layer-`REGISTER_LAYER` text MLP adds u_p = gate * (W_down @ a_p) to the
residual stream. The massive direction d is the normalized sum of u_p over the 18 prompt-template head tokens. Neuron
n is ranked by its summed contribution sum_p a_p[n] * (gate * W_down[:, n]) . d over those tokens; the output also
records the cumulative share, the channels of d, and how much of each neuron's contribution falls on task tokens.
The direction used by `shift="direction"` is the normalized sum of the MLP outputs (before the gate) over the same
tokens.

  python find_neurons.py   ->   runs/register_neurons.json, runs/register_direction.npy
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "libero_quant"))
from eval_quant import build_policy  # noqa: E402
from screen_quant import CALIB_CALLS, CALIB_TASKS, restored_inputs  # noqa: E402

from register import DIRECTION_FILE, NEURONS_FILE, REGISTER_LAYER  # noqa: E402

TEMPLATE = 18  # "<|im_start|>user\nA video recorded from a robot's point of view executing the following instruction:"


def main() -> None:
    policy = build_policy()
    model = policy.model
    block = model.mot.mixtures["video"].double_blocks[REGISTER_LAYER]
    w_down = block.txt_mlp[2].weight.detach().float()  # [3072, 9216]
    captured = {}
    block.txt_mlp[1].register_forward_hook(lambda m, a, out: captured.__setitem__("act", out[0].float()))
    original = block._apply_residuals

    def residuals(img, txt, img_attn, txt_attn, mods):
        captured["gate"] = mods[7].reshape(-1).float()  # txt_mod2_gate
        return original(img, txt, img_attn, txt_attn, mods)

    block._apply_residuals = residuals
    inputs = restored_inputs(policy, CALIB_TASKS, CALIB_CALLS)
    head_sum, task_sum, update_sum = 0.0, 0.0, 0.0
    per_state = []
    for key, prepared in inputs:
        _, mask = model._prepare_flux2_infer_text(prepared.prompt, None, None)
        n_text = int(mask[0].sum())  # proprio sits at index n_text
        with torch.no_grad():
            model.infer_action(prompt=prepared.prompt, input_image=prepared.image, action_horizon=16,
                               num_inference_steps=1, proprio=prepared.proprio, seed=42, rand_device="cpu")
            act, gate = captured["act"], captured["gate"]
            update = (act[:TEMPLATE] @ w_down.T) * gate  # [18, 3072]
            update_sum = update_sum + update.sum(0)
            head_sum = head_sum + act[:TEMPLATE].sum(0)
            task_sum = task_sum + act[TEMPLATE:n_text - 9].sum(0)
            per_state.append(update.norm(dim=-1).cpu().numpy())
        print(key, flush=True)
    direction = update_sum / update_sum.norm()
    column = (w_down * gate[:, None]).T @ direction  # [9216]: d-component of each neuron's output column
    contribution = (head_sum * column).cpu().numpy()  # summed over template tokens and states
    task_contribution = (task_sum * column).cpu().numpy()
    order = np.argsort(-contribution)
    total = contribution.sum()
    cumulative = np.cumsum(contribution[order]) / total
    channels = torch.argsort(-direction.abs())[:8].tolist()
    template_norm = float(np.median(np.concatenate(per_state)))
    report = {
        "layer": REGISTER_LAYER, "calibration_states": len(inputs),
        "direction_top_channels": {int(c): round(float(direction[c]), 4) for c in channels},
        "direction_share_top4": float((direction[channels[:4]] ** 2).sum()),
        "template_update_norm_median": template_norm,
        "ranked_neurons": [int(n) for n in order],
        "cumulative_share": [round(float(c), 5) for c in cumulative[:2048]],
        "neurons_for_share": {str(s): int(np.searchsorted(cumulative, s) + 1) for s in (0.5, 0.8, 0.9, 0.95, 0.99)},
        "top_neurons": [{"neuron": int(n), "share": float(contribution[n] / total),
                         "task_to_template_ratio": float(task_contribution[n] / contribution[n])}
                        for n in order[:32]],
    }
    NEURONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    mlp_direction = w_down @ head_sum
    np.save(DIRECTION_FILE, (mlp_direction / mlp_direction.norm()).cpu().numpy().astype(np.float32))
    NEURONS_FILE.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("ranked_neurons", "cumulative_share")}, indent=1))


if __name__ == "__main__":
    main()
