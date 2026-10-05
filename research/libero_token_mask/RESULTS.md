# Random Image-Token Masking on LIBERO: Results

Run: `runs/20261004_osmesa_grid/` (git-ignored). Settings as in `PLAN.md`:
- released ImageWAM-FLUX.2-4B LIBERO checkpoint, OSMesa rendering;
- 4 suites × 10 tasks × 5 initial states per masked fraction;
- inference seed 42; a paired unmasked inference at every call of the masked levels.

All 600 episodes completed with no worker crash. Summary and figure copies are in `figures/`.

## Success Rate

| Masked image tokens | All | Spatial | Object | Goal | Long |
|---|---|---|---|---|---|
| 0 | 100.0% (200/200) | 100 | 100 | 100 | 100 |
| 1/3 (131 of 392) | 97.5% ± 1.1 | 96.0 ± 2.8 | 100 | 94.0 ± 3.4 | 100 |
| 2/3 (261 of 392) | 74.5% ± 3.1 | 88.0 ± 4.6 | 88.0 ± 4.6 | 62.0 ± 6.9 | 60.0 ± 6.9 |

Paired against the unmasked episode with the same task and initial state (exact McNemar test):

| Masked | All | Spatial | Object | Goal | Long |
|---|---|---|---|---|---|
| 1/3 | 5 lost, 0 gained, p = 0.062 | 2 / 0, p = 0.5 | 0 / 0 | 3 / 0, p = 0.25 | 0 / 0 |
| 2/3 | 51 lost, 0 gained, p = 9e-16 | 6 / 0, p = 0.031 | 6 / 0, p = 0.031 | 19 / 0, p = 4e-6 | 20 / 0, p = 2e-6 |

Lowest per-task successes (out of 5) at 2/3:
- Goal: "open the top drawer and put the bowl inside" 0, "open the middle drawer of the cabinet" 1, "put the wine bottle on the rack" 2.
- Long: "put the white mug on the left plate and put the yellow and white mug on the right plate" 0, "put the yellow and white mug in the microwave and close it" 1, "put both moka pots on the stove" 2, "put the white mug on the plate and put the chocolate pudding to the right of the plate" 2.

## Paired Action Deviation

Mean absolute difference between the masked and the paired unmasked action chunk, in normalized action units, over 16 × 7 values.

| Masked | Calls | Mean | Median | 90th percentile | Episode mean, successes | Episode mean, failures |
|---|---|---|---|---|---|---|
| 1/3 | 2,754 | 0.039 | 0.032 | 0.062 | 0.037 | 0.069 |
| 2/3 | 4,509 | 0.096 | 0.075 | 0.172 | 0.084 | 0.108 |

The unmasked attention share that fell on the masked tokens correlates only weakly with the deviation of a call: Spearman ρ = 0.26 at 1/3 and 0.16 at 2/3. Random masks remove a nearly constant share (mean 0.333 and 0.666), so this test has little range.

## Attention Profile of the Unmasked Model

Action-to-prefix attention, averaged over heads, action tokens, denoising steps and calls. Full per-layer values are in `figures/summary.json` → `attention`.

| Layers | Valid text | Proprio | Agent-view image | Wrist image | Action tokens | Top-20 of 392 image tokens |
|---|---|---|---|---|---|---|
| 0 | 0.39 | 0.01 | 0.24 | 0.14 | 0.23 | 0.11 |
| 1–4 | 0.57–0.82 | 0.04–0.07 | 0.03–0.17 | 0.01–0.11 | 0.02–0.18 | 0.17–0.29 |
| 8–14 | 0.12–0.25 | ≤ 0.01 | 0.19–0.24 | 0.44–0.62 | 0.02–0.07 | 0.47–0.57 |
| 16–22 | 0.57–0.67 | 0.01–0.02 | 0.01–0.10 | 0.03–0.25 | 0.07–0.28 | 0.38–0.68 |
| 23–24 | 0.35–0.54 | 0.01–0.02 | 0.01–0.02 | 0.01–0.03 | 0.42–0.59 | 0.28–0.41 |

- **Image attention** (`figures/07_attention_on_frames.png`):
  - Layer 0 spreads over the center of the agent view.
  - Layers 8–15 concentrate on the lower center of the wrist view, the gripper and the grasped object. There, 20 of 392 tokens receive about half of the image attention.
  - Late layers keep a few wrist tokens.
- **Text attention**: it is not a single-token sink. The most-attended text token holds 6–12% of the text attention per layer. Its position is mostly 2, 5 or 6, inside the fixed `DEFAULT_PROMPT` template before the task words. How much of the text share carries task information is open.

## Conclusions

- Removing a random third of the reference-image tokens costs no measurable success: 97.5% vs. 100%, paired p = 0.062. The paired action deviation is 0.039.
- Removing two thirds costs 25.5 points (p = 9e-16). The loss concentrates in Goal (−38) and Long (−40), which require locating a specific object or multiple sub-goals in the scene, against −12 in Spatial and Object.
- Failed episodes have larger paired deviations than successful ones at both levels.
- The action expert reads the image mostly through a small set of wrist-view tokens in its middle layers. A random mask cannot test whether keeping those tokens preserves success; that needs an attention-guided mask at the same budget.
- The success rates at 0 and 1/3 sit at the ceiling of standard LIBERO, so small effects need a harder benchmark (LIBERO-Plus) or larger budgets.

## Open Questions

- An attention-guided mask at 2/3 or higher: keep the tokens with the most attention at the previous control step vs. random vs. the least attended.
- Per-camera masking: agent view only vs. wrist view only.
- Open-loop evaluation of mask strategies on the saved simulator states of the unmasked episodes, before closed-loop confirmation.
- Measured speed-up when masked tokens are removed from the sequence, including removal of the 512-slot text padding.

# Strategy Comparison: Removal, Merging and K/V Reuse

Run: `runs/20261004_reduce_grid/` (git-ignored). 11 new conditions × 200 episodes; every one of the 2,200 episodes completed without a worker crash. The no-reduction and random 131/261 conditions are those of the first grid. Design and sanity checks: `PLAN.md` → Strategy Comparison Grid. Figures: `figures/` (`make_figures.py`).

## Success Rate

Success rate in %. "Lost / gained" counts episodes that fail / succeed only with the reduction, paired with the no-reduction episode of the same task and initial state; p is the exact McNemar test.

| Condition | Tokens reduced | All | Spatial | Object | Goal | Long | Lost / gained | p |
|---|---|---|---|---|---|---|---|---|
| none | 0 | 100.0 | 100 | 100 | 100 | 100 | – | – |
| random removal | 131 (1/3) | 97.5 ± 1.1 | 96 | 100 | 94 | 100 | 5 / 0 | 0.062 |
| random removal | 196 (1/2) | 95.0 ± 1.5 | 98 | 96 | 94 | 92 | 10 / 0 | 0.002 |
| random removal | 261 (2/3) | 74.5 ± 3.1 | 88 | 88 | 62 | 60 | 51 / 0 | 9e-16 |
| grid removal | 131 | 97.5 ± 1.1 | 94 | 100 | 96 | 100 | 5 / 0 | 0.062 |
| grid removal (checkerboard) | 196 | 98.0 ± 1.0 | 96 | 100 | 100 | 96 | 4 / 0 | 0.12 |
| grid removal | 261 | 87.0 ± 2.4 | 94 | 96 | 80 | 78 | 26 / 0 | 3e-8 |
| ToMe merge | 131 | 98.0 ± 1.0 | 96 | 98 | 98 | 100 | 4 / 0 | 0.12 |
| ToMe merge | 196 | 34.5 ± 3.4 | 66 | 16 | 42 | 14 | 131 / 0 | 7e-40 |
| ToMe merge | 261 | 16.5 ± 2.6 | 38 | 2 | 26 | 0 | 167 / 0 | 1e-50 |
| K/V reuse, random | 131 | 19.5 ± 2.8 | 10 | 16 | 40 | 12 | 161 / 0 | 7e-49 |
| K/V reuse, random | 261 | 0 | 0 | 0 | 0 | 0 | 200 / 0 | 1e-60 |
| K/V reuse, top-9 refreshed | 383 | 0 | 0 | 0 | 0 | 0 | 200 / 0 | 1e-60 |
| K/V reuse, 9 random refreshed | 383 | 0 | 0 | 0 | 0 | 0 | 200 / 0 | 1e-60 |

Paired comparisons between strategies (same task and initial state; episodes where only one of the two succeeds; exact McNemar):

| Comparison | Only first succeeds | Only second succeeds | p |
|---|---|---|---|
| grid vs. random, 261 | 38 | 13 | 0.0006 |
| grid vs. random, 196 | 9 | 3 | 0.15 |
| grid vs. random, 131 | 4 | 4 | 1 |
| ToMe vs. grid, 196 | 2 | 129 | 6e-36 |
| ToMe vs. grid, 131 | 4 | 3 | 1 |
| K/V reuse vs. random removal, 131 | 1 | 157 | 9e-46 |

## Paired Action Deviation

Mean absolute difference between the reduced and the paired unreduced action chunk (normalized units, 16 × 7 values). For K/V reuse, the first call of each episode computes every token and is excluded.

| Condition | Tokens reduced | Mean | Median | 90th percentile |
|---|---|---|---|---|
| random removal | 131 / 196 / 261 | 0.039 / 0.058 / 0.096 | 0.032 / 0.048 / 0.075 | 0.062 / 0.092 / 0.172 |
| grid removal | 131 / 196 / 261 | 0.031 / 0.038 / 0.068 | 0.026 / 0.033 / 0.057 | 0.048 / 0.057 / 0.106 |
| ToMe merge | 131 / 196 / 261 | 0.048 / 0.132 / 0.148 | 0.040 / 0.106 / 0.123 | 0.078 / 0.259 / 0.281 |
| K/V reuse, random | 131 / 261 | 0.095 / 0.107 | 0.070 / 0.073 | 0.208 / 0.239 |
| K/V reuse, top-9 / 9 random refreshed | 383 | 0.192 / 0.210 | 0.182 / 0.203 | 0.266 / 0.281 |

K/V entry age when reused, in calls (12 control steps per call):

| Condition | Mean | Median | 90th percentile | Max |
|---|---|---|---|---|
| random, 131 | 1.5 | 1 | 3 | 14 |
| random, 261 | 2.8 | 2 | 6 | 29 |
| top-9 refreshed | 20.9 | 19 | 42 | 58 |
| 9 random refreshed | 15.9 | 13 | 32 | 58 |

- K/V-reuse deviation peaks within the first five calls and then falls (`figures/08_kv_reuse.png`). Failed episodes stall, and once the scene stops changing the stale entries match it again.
- The same dilution explains why the successful K/V-reuse episodes at 131 have a larger mean deviation (0.120) than the failed ones (0.095).
- `kv_topk` refreshes the robot arm in the agent view and the gripper fingers in the wrist view (refresh-frequency map in `figures/08_kv_reuse.png`).

## Open-Loop Dose Test: Reused K/V vs. Removed Tokens

Script: `kv_dose.py`; output: `runs/20261004_reduce_grid/kv_dose.json`.

- **States:** saved states of 20 no-reduction episodes (tasks 0–4 of every suite, trial 0) at calls 1, 3 and 6.
- **Canvas:** the K/V canvas comes from the observation of the previous call.
- **Arms:** the same nested random token sets are either reused from the canvas or removed.
- **Measure:** both arms are compared with the full inference on the same restored state, which isolates the mechanism from closed-loop drift.

| Tokens | Deviation, reused | Deviation, removed | Gripper sign flips, reused | Gripper sign flips, removed |
|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 0% | 0% |
| 13 | 0.019 | 0.008 | 0.8% | 0.3% |
| 33 | 0.042 | 0.012 | 2.5% | 0.1% |
| 65 | 0.078 | 0.022 | 5.3% | 0.4% |
| 131 | 0.134 | 0.035 | 8.3% | 1.0% |
| 196 | 0.184 | 0.053 | 10.8% | 2.2% |
| 261 | 0.208 | 0.078 | 11.4% | 2.1% |
| 383 | 0.221 | 0.291 | 11.7% | 36.7% |

Gripper sign flips are the share of the 12 executed steps whose binarized gripper command differs from the full inference.

- With no token reused or removed, both arms equal the full inference exactly.
- Reused K/V deviates 3–4 times as much as removal at every count up to 261, and the deviation grows monotonically with the count. It flips the gripper command 4–8 times as often.
- Only with 383 of 392 tokens affected is a stale image better than almost no image.
- The simulator restore is not bit-exact: the full inference on the restored state differs from the recorded action by a median of 0.018, and by more than 0.1 in 7 of 60 states. Both arms use the same restored state, so the comparison is unaffected.

## Conclusions

- **Fixed regular removal is at least as good as random removal.** It is better at 2/3 (87.0% vs. 74.5%, paired p = 0.0006), mostly in Goal (+18) and Long (+18). A regular pattern leaves every image region represented, while random removal can empty a region such as the gripper area.
- **ToMe at the input matches removal at 1/3 and fails beyond it** (34.5% at 1/2 against 98.0% for the checkerboard). A merged token is an average of VAE latents at an averaged, fractional position, which the model has never seen. The ToFu variant below rules out the shrunken norm.
- **Reusing stale keys and values for a random subset is far worse than removing the same tokens:** 19.5% vs. 97.5% at 1/3, and 0% at 2/3 and with 9 refreshed tokens.
  - The open-loop test confirms that this comes from the reuse itself, not from closed-loop drift.
  - Randomly chosen stale entries include the moving gripper and objects at old positions, which contradict the fresh tokens.
  - With only 9 tokens refreshed, attention selection does not help, because 383 stale entries dominate.
  - Choosing the reused tokens by attention reverses the picture (next section).

# Attention-Guided Pruning, EViT and Token Fusion

Run: `runs/20261005_attn_merge_grid/` (git-ignored). 8 conditions × 200 episodes; every one of the 1,600 episodes completed without a worker crash. Design and the per-call score: `PLAN.md` → Attention-Guided Pruning and Merging Variants.

Sanity checks (`sanity_check.py`):
- Removed tokens receive zero attention.
- Every kept token scores at least as high as every removed one.
- EViT feeds k + 1 image tokens, the last of size n.
- Each ToFu merged token's norm equals the largest norm in its group (relative difference ≤ 6e-4, bf16).

| Condition | Tokens reduced | All | Spatial | Object | Goal | Long | Lost / gained vs. none | p | Deviation, median (p90) |
|---|---|---|---|---|---|---|---|---|---|
| attention top-k removal | 131 | 98.5 ± 0.9 | 96 | 100 | 100 | 98 | 3 / 0 | 0.25 | 0.010 (0.018) |
| attention top-k removal | 196 | 98.0 ± 1.0 | 96 | 100 | 100 | 96 | 4 / 0 | 0.12 | 0.017 (0.033) |
| attention top-k removal | 261 | 98.5 ± 0.9 | 96 | 100 | 98 | 100 | 3 / 0 | 0.25 | 0.031 (0.055) |
| EViT fusion | 131 | 99.0 ± 0.7 | 96 | 100 | 100 | 100 | 2 / 0 | 0.5 | 0.009 (0.017) |
| EViT fusion | 196 | 98.5 ± 0.9 | 98 | 100 | 100 | 96 | 3 / 0 | 0.25 | 0.017 (0.031) |
| EViT fusion | 261 | 97.0 ± 1.2 | 96 | 100 | 96 | 96 | 6 / 0 | 0.031 | 0.031 (0.056) |
| ToFu MLERP merge | 196 | 32.0 ± 3.3 | 60 | 12 | 46 | 10 | 136 / 0 | 2e-41 | 0.103 (0.263) |
| ToFu MLERP merge | 261 | 18.0 ± 2.7 | 40 | 0 | 32 | 0 | 164 / 0 | 9e-50 | 0.124 (0.274) |

Paired comparisons between strategies (episodes where only one of the two succeeds; exact McNemar):

| Comparison | Only first succeeds | Only second succeeds | p |
|---|---|---|---|
| attention top-k vs. random removal, 261 | 51 | 3 | 3e-12 |
| attention top-k vs. grid removal, 261 | 24 | 1 | 1.5e-6 |
| attention top-k vs. grid removal, 196 | 4 | 4 | 1 |
| EViT vs. attention top-k, 261 | 1 | 4 | 0.38 |
| EViT vs. attention top-k, 196 | 2 | 1 | 1 |
| ToFu vs. ToMe, 196 | 20 | 25 | 0.55 |
| ToFu vs. ToMe, 261 | 11 | 8 | 0.65 |

## Conclusions

- **Which tokens are kept matters more than how many.**
  - Keeping the 131 tokens the action expert attends to most loses no measurable success: 98.5%, against 74.5% for random and 87.0% for grid removal at the same budget.
  - The paired action deviation at 2/3 removed (median 0.031) is below that of random removal at 1/3 (0.032).
  - At 1/2 the selection no longer matters: grid and attention both reach 98%.
- **The selection needs the full input.** The score comes from a full-input prefill and one action denoising step on the current observation. On H100 a 1-step `infer_action` (VAE, prefill, one action step) takes 38.4 ms against 155.8 ms for 10 steps (`research/latency/results.txt`), so scoring costs about a quarter of an inference. The reduced inference after it saves compute only in the remaining steps. Whether a cheaper score works (fewer layers, the previous call's score with periodic refresh) is open.
- **EViT's fused token adds nothing.** It receives little attention (0.0004 in the sanity state), and success equals top-k removal (paired p ≥ 0.38).
- **ToMe's failure is not the norm of the averaged tokens.** Norm-preserving MLERP gives the same success as ToMe (paired p = 0.55 and 0.65). The remaining candidates are the blended latent content and the averaged fractional positions; neither was tested separately.

## Open Questions

- Attention top-k with fewer kept tokens (for example 64, 32, 16), to find where it starts to fail.
- A cheaper score: an early-layer attention, or the previous call's score refreshed by a full pass every few calls.
- Measured speed-up when the removed tokens are dropped from the sequence instead of masked.
- ToMe with destination positions kept (no position averaging), to separate content blending from position averaging.

# K/V Reuse at Proportional Budgets

Run: `runs/20261005_kv_topk_ratio/` (git-ignored). Every one of the 800 episodes completed without a worker crash.

Conditions:
- `kv_topk` at 1/3, 1/2 and 2/3 reused: the tokens with the most action attention at the previous call are recomputed, and the rest reuse their K/V. This is the same rule as the 9-token condition, at the budgets of the other strategies.
- `kv_random` at 1/2, which completes the random control.

The previous call's score suffices here: reused tokens stay in attention, so their score keeps updating and the selection does not freeze.

| Tokens reused | Refresh by previous attention | Random | Lost / gained vs. none (attention) | Deviation, median (attention / random) | Reused-entry age in calls, mean (attention / random) |
|---|---|---|---|---|---|
| 131 (1/3) | 96.5 ± 1.3 | 19.5 ± 2.8 | 7 / 0, p = 0.016 | 0.015 / 0.070 | 7.1 / 1.5 |
| 196 (1/2) | 91.5 ± 2.0 | 0.5 ± 0.5 | 17 / 0, p = 2e-5 | 0.031 / 0.075 | 9.6 / 1.9 |
| 261 (2/3) | 74.0 ± 3.1 | 0 | 52 / 0, p = 4e-16 | 0.064 / 0.073 | 13.9 / 2.8 |
| 383 (9 refreshed) | 0 | 0 | 200 / 0 | 0.182 / 0.203 | 20.9 / 15.9 |

Per suite with attention refresh (Spatial / Object / Goal / Long):

| Tokens reused | Spatial | Object | Goal | Long |
|---|---|---|---|---|
| 1/3 | 96 | 100 | 96 | 94 |
| 1/2 | 90 | 100 | 94 | 82 |
| 2/3 | 56 | 94 | 88 | 58 |

Paired comparisons (only first succeeds / only second succeeds, exact McNemar):

| Comparison | Only first | Only second | p |
|---|---|---|---|
| attention vs. random reuse, 1/3 | 154 | 0 | 9e-47 |
| attention vs. random reuse, 1/2 | 182 | 0 | 3e-55 |
| attention vs. random reuse, 2/3 | 148 | 0 | 6e-45 |
| attention reuse vs. random removal, 1/3 | 3 | 5 | 0.73 |
| attention reuse vs. random removal, 1/2 | 8 | 15 | 0.21 |
| attention reuse vs. random removal, 2/3 | 34 | 35 | 1 |
| attention reuse vs. attention top-k removal, 2/3 | 2 | 51 | 3e-13 |

## Conclusions

- **Which tokens reuse their K/V decides whether reuse works.**
  - Reusing the 131 least-attended tokens keeps 96.5% success, against 19.5% for a random 131 (154 paired episodes against 0).
  - The reused entries are older (7.1 calls on average vs. 1.5), yet the deviation is a fifth of the random case. The reused tokens are mostly static background (reuse-frequency map in `figures/08_kv_reuse.png`), where a stale entry still matches the scene.
- **Attention-selected reuse matches random removal at every budget** (paired p ≥ 0.21). It needs no extra pass, because the score comes from the previous call.
- **It stays below attention-selected removal at 2/3** (74.0% vs. 98.5%, paired 2 against 51), and its losses at 2/3 concentrate in Spatial (56) and Long (58). The latter re-scores on the current observation, at about a quarter of an inference.
