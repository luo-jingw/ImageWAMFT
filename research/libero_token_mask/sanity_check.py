"""Checks every reduction of eval_token_mask.py on LIBERO-Spatial task 0, initial state 0:
  - no reduction reproduces the official LIBERO prediction path;
  - random removal leaves zero attention on the removed tokens; grid patterns have 131/196/261 tokens;
  - ToMe with zero merges (sizes 1, float-mask path) matches no reduction up to kernel rounding; real merges shorten
    the image sequence to 261/196/131 tokens with finite actions and normalized attention;
  - K/V reuse: the first call (empty canvas) is exact; reusing K/V cached from the same observation is exact; after
    the scene moves, reuse deviates from fresh computation.
"""
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


def run(reduction, observation=None, paired=True):
    return policy.act(policy.prepare(obs if observation is None else observation, desc), reduction, paired=paired)


def deviation(record):
    return float(np.abs(record["masked"]["action_norm"] - record["unmasked"]["action_norm"]).mean())


def attention_total(record):
    return np.round(record["masked"]["attention"].astype(np.float32).sum(1) + record["masked"]["attention_self"], 3)


official_action, _, _ = E.official._predict_action_chunk(
    obs=obs, task_description=desc, model=policy.model, processor=policy.processor, cfg=cfg,
    action_horizon=16, input_w=policy.input_w, input_h=policy.input_h, model_device="cuda")
base = run(E.Reduction(), paired=False)
reference = base["masked"]["action_norm"]
print("[none] executed vs official: max abs diff", float(np.abs(base["executed"] - official_action).max()))

rng = np.random.default_rng(0)
reduction, mask = E.call_reduction("random", 131, rng, False, None)
record = run(reduction)
txt_len = record["masked"]["attention"].shape[1] - E.IMAGE_TOKENS
print("[random 131] paired unreduced vs none: max abs diff",
      float(np.abs(record["unmasked"]["action_norm"] - reference).max()),
      "| deviation", round(deviation(record), 4),
      "| attention on removed tokens", float(record["masked"]["attention"][:, txt_len + np.nonzero(mask)[0]].sum()))
for n in (131, 196, 261):
    reduction, mask = E.call_reduction("grid", n, rng, False, None)
    record = run(reduction)
    print(f"[grid {n}] removed {int(mask.sum())} | deviation {deviation(record):.4f}")

record = run(E.Reduction(merge_rounds=(0,)))
print("[tome 0 merges] max abs diff vs none", float(np.abs(record["masked"]["action_norm"] - reference).max()),
      "| sizes", np.unique(record["masked"]["merge_size"]))
for n in (131, 196, 261):
    reduction, _ = E.call_reduction("tome", n, rng, False, None)
    record = run(reduction)
    m = record["masked"]
    print(f"[tome {n}] image tokens {m['attention'].shape[1] - txt_len}, size sum {m['merge_size'].sum():.0f}, "
          f"groups {len(np.unique(m['merge_group']))}, finite {np.isfinite(m['action_norm']).all()}, "
          f"deviation {deviation(record):.4f}, attention row totals {attention_total(record)[[0, 12, 24]]}")

policy.canvas.reset()
reduction, mask = E.call_reduction("kv_random", 261, rng, True, None)
first = run(reduction)
print("[kv first call] stale", int(mask.sum()), "| max abs diff vs none",
      float(np.abs(first["masked"]["action_norm"] - reference).max()), "| canvas layers", len(policy.canvas.k),
      tuple(policy.canvas.k[0].shape))
reduction, mask = E.call_reduction("kv_random", 261, rng, False, None)
same = run(reduction)
print("[kv 261 stale, same observation] max abs diff vs none",
      float(np.abs(same["masked"]["action_norm"] - reference).max()))
for _ in range(12):
    obs, _, _, _ = env.step(base["executed"][_ % 16])
previous = same["masked"]["attention"]
for strategy, n in (("kv_random", 131), ("kv_random", 261), ("kv_topk", 383), ("kv_randk", 383)):
    canvas_k = [k.clone() for k in policy.canvas.k]
    canvas_v = [v.clone() for v in policy.canvas.v]
    reduction, mask = E.call_reduction(strategy, n, rng, False, previous)
    record = run(reduction)
    stale = np.nonzero(mask)[0]
    kept = torch.from_numpy(stale).long().to(canvas_k[0].device)
    unchanged = all(torch.equal(policy.canvas.k[i][:, kept], canvas_k[i][:, kept]) for i in range(len(canvas_k)))
    print(f"[{strategy} {n}] stale {len(stale)}, deviation after 12 steps {deviation(record):.4f}, "
          f"stale canvas entries unchanged {unchanged}")
    policy.canvas.k, policy.canvas.v = canvas_k, canvas_v
score = policy.score(policy.prepare(obs, desc))
print("[score] shape", score.shape, "sum", round(float(score.sum()), 4), "| top-5 tokens", np.argsort(-score)[:5].tolist())
for n in (131, 196, 261):
    reduction, mask = E.call_reduction("attn_topk", n, rng, False, None, score)
    record = run(reduction)
    print(f"[attn_topk {n}] removed {int(mask.sum())}, attention on removed "
          f"{float(record['masked']['attention'][:, txt_len + np.nonzero(mask)[0]].sum()):.1e}, "
          f"lowest kept score {score[~mask].min():.2e} >= highest removed {score[mask].max():.2e}, "
          f"deviation {deviation(record):.4f}")
    reduction, mask = E.call_reduction("evit", n, rng, False, None, score)
    record = run(reduction)
    m = record["masked"]
    print(f"[evit {n}] image tokens {m['attention'].shape[1] - txt_len} (kept {E.IMAGE_TOKENS - n} + 1 fused), "
          f"fused size {m['merge_size'][-1]:.0f}, groups {len(np.unique(m['merge_group']))}, "
          f"attention on fused token {float(m['attention'][:, -1].astype(np.float32).mean()):.4f}, "
          f"deviation {deviation(record):.4f}")
for n in (196, 261):
    reduction, _ = E.call_reduction("tofu", n, rng, False, None)
    tokens, ids = policy.model.__class__._encode_flux2_image_tokens(policy.model, policy.prepare(obs, desc).image,
                                                                   time_value=10.0)
    policy.active[0] = reduction
    merged, _ = policy.model._encode_flux2_image_tokens(policy.prepare(obs, desc).image, time_value=10.0)
    group = policy.merge_state.group
    policy.active[0] = E.Reduction()
    norms = tokens[0].float().norm(dim=-1).cpu().numpy()
    expected = np.array([norms[group == g].max() for g in range(merged.shape[1])])
    got = merged[0].float().norm(dim=-1).cpu().numpy()
    record = run(reduction)
    print(f"[tofu {n}] image tokens {merged.shape[1]}, merged norm vs max member norm: max rel diff "
          f"{float(np.abs(got - expected).max() / expected.max()):.1e}, deviation {deviation(record):.4f}")
top = np.argsort(-previous[:, -E.IMAGE_TOKENS:].astype(np.float32).mean(0))[:9]
print("[kv_topk] refreshed tokens (row, col):", [tuple(int(v) for v in divmod(i, E.GRID_W)) for i in top])
record = run(E.Reduction())
print("timing (s): none", round(base["masked"]["seconds"], 3), "| peak GPU memory GiB",
      round(torch.cuda.max_memory_allocated() / 2**30, 1))
