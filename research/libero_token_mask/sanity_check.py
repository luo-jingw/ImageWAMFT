"""Checks the token-mask evaluator against the official LIBERO prediction path on one simulator state."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_token_mask as E  # noqa: E402
from libero.libero import benchmark  # noqa: E402

cfg = E.eval_config()
E.set_global_seed(int(cfg.seed), get_worker_init_fn=False)
policy = E.Policy(cfg)
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
env, desc = E.get_libero_env(suite.get_task(0), E.LIBERO_ENV_RESOLUTION, cfg.get("seed"))
env.reset()
obs = env.set_init_state(suite.get_task_init_states(0)[0])
for _ in range(30):
    obs, _, _, _ = env.step(E.get_libero_dummy_action())
official_action, _, _ = E.official._predict_action_chunk(
    obs=obs, task_description=desc, model=policy.model, processor=policy.processor, cfg=cfg,
    action_horizon=16, input_w=policy.input_w, input_h=policy.input_h, model_device="cuda")
record = policy.act(obs, desc, None, paired=False)
print("no-mask vs official executed action: max abs diff", float(np.abs(record["executed"] - official_action).max()))
rng = np.random.default_rng(0)
drop = torch.from_numpy(np.sort(rng.permutation(392)[:131])).long()
masked = policy.act(obs, desc, drop, paired=True)
dev = np.abs(masked["masked"]["action_norm"] - masked["unmasked"]["action_norm"])
print("paired unmasked vs no-mask call: max abs diff", float(np.abs(masked["unmasked"]["action_norm"] - record["masked"]["action_norm"]).max()))
print("1/3 mask vs unmasked (normalized action): mean abs", float(dev.mean()), "max", float(dev.max()))
att, self_mass = masked["unmasked"]["attention"], masked["unmasked"]["attention_self"]
print("attention", att.shape, att.dtype, "| prefix + self mass per layer:", np.round(att.astype(np.float32).sum(1) + self_mass, 4)[:5], "...")
att_m = masked["masked"]["attention"].astype(np.float32)
txt_len = att.shape[1] - 392
print("masked run: attention on dropped image tokens (should be 0):", float(att_m[:, txt_len + drop.numpy()].sum()))
print("timing (s): no-mask", round(record["masked"]["seconds"], 3), "masked", round(masked["masked"]["seconds"], 3))
