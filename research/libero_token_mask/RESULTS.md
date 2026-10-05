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

Action-to-prefix attention, averaged over heads, action tokens, denoising steps and calls. Full per-layer values are in `figures/summary_osmesa.json`.

| Layers | Valid text | Proprio | Agent-view image | Wrist image | Action tokens | Top-20 of 392 image tokens |
|---|---|---|---|---|---|---|
| 0 | 0.39 | 0.01 | 0.24 | 0.14 | 0.23 | 0.11 |
| 1–4 | 0.57–0.82 | 0.04–0.07 | 0.03–0.17 | 0.01–0.11 | 0.02–0.18 | 0.17–0.29 |
| 8–14 | 0.12–0.25 | ≤ 0.01 | 0.19–0.24 | 0.44–0.62 | 0.02–0.07 | 0.47–0.57 |
| 16–22 | 0.57–0.67 | 0.01–0.02 | 0.01–0.10 | 0.03–0.25 | 0.07–0.28 | 0.38–0.68 |
| 23–24 | 0.35–0.54 | 0.01–0.02 | 0.01–0.02 | 0.01–0.03 | 0.42–0.59 | 0.28–0.41 |

- **Image attention** (`figures/image_attention_layers_osmesa.png`):
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
