# Test-Time Registers for ImageWAM on LIBERO: Results

Design and register settings: `PLAN.md`. Figures and every number below: `figures/` (`make_figures.py`), `figures/summary.json`.

## Register Neurons

`find_neurons.py`, 40 calibration states, output `runs/register_neurons.json`.
- **Where:** the text MLP of layer 4, the last double block, writes the massive update on the template tokens. Its norm is about 43,000; the residual stream entering that layer has norm about 2,800.
- **Direction:** 97% of the update's direction lies in four channels: 2769 (+), 364 (−), 2808 (−) and 131 (+).
- **Not sparse:** 23 neurons give 50% of the update along this direction, 134 give 90% and 1,057 give 99% (of 9,216). Jiang et al. found a handful of register neurons in ViTs; here the update is spread out.

## Open-Loop Screen

`screen_register.py`, output `runs/screen.json`. 60 restored states; each inference with registers uses the same observation and initial noise as the unmodified one.
- **Deviation:** mean absolute difference of the normalized 16 × 7 action chunk.
- **Attention:** image queries are the 392 reference-image tokens in the prefix pass. Action queries are taken at the first denoising step. Both are averaged over heads, queries and states.
- **Norm:** median residual norm entering the first single block.

| Config | Deviation, mean | Gripper flips | Image queries on template, layers 5–10 | Image queries on registers, layers 5–10 | Action queries on template | Action queries on registers | Template norm | Register norm |
|---|---|---|---|---|---|---|---|---|
| unmodified (repeat) | 0.0000 | 0% | 43.3% | – | 27.2% | – | 48,381 | – |
| 1 zero token, nothing moved | 0.0012 | 0.1% | 43.1% | 0.5% | 26.8% | 0.9% | 48,284 | 2,277 |
| 16 zero tokens, nothing moved | 0.0040 | 0.3% | 40.2% | 6.3% | 23.0% | 9.5% | 47,520 | 2,256 |
| direction removed, no registers | 0.0384 | 0.3% | 16.9% | – | 22.8% | – | 5,038 | – |
| direction → 1 register | 0.0328 | 0.1% | 7.2% | 29.0% | 18.8% | 8.6% | 5,015 | 218,319 |
| direction → 4 registers | 0.0289 | 0.1% | 4.1% | 43.0% | 15.8% | 15.8% | 5,017 | 219,145 |
| direction → 16 registers | 0.0244 | 0.1% | 2.0% | 56.2% | 12.0% | 25.5% | 4,953 | 220,356 |
| 50% neurons removed | 0.0040 | 0.1% | 40.6% | – | 27.1% | – | 29,830 | – |
| 50% neurons → 1 register | 0.0036 | 0.1% | 35.8% | 7.6% | 26.1% | 1.8% | 29,814 | 162,242 |
| 50% neurons → 4 registers | 0.0030 | 0.3% | 28.2% | 20.0% | 24.0% | 5.7% | 29,691 | 162,239 |
| 50% neurons → 16 registers | 0.0042 | 0.3% | 17.6% | 39.1% | 19.5% | 14.3% | 29,385 | 161,431 |
| 90% neurons removed | 0.0105 | 0.1% | 33.5% | – | 25.7% | – | 6,764 | – |
| 90% neurons → 1 register | 0.0096 | 0.1% | 27.4% | 10.9% | 24.3% | 2.6% | 6,799 | 369,411 |
| 90% neurons → 4 registers | 0.0080 | 0% | 20.1% | 25.1% | 21.7% | 7.7% | 6,890 | 367,706 |
| 90% neurons → 16 registers | 0.0066 | 0.1% | 11.6% | 43.3% | 17.0% | 17.5% | 7,165 | 365,959 |
| 99% neurons removed | 0.0169 | 0.3% | 21.6% | – | 23.7% | – | 2,700 | – |
| 99% neurons → 1 register | 0.0150 | 0.1% | 15.0% | 17.0% | 21.8% | 4.2% | 2,698 | 461,831 |
| 99% neurons → 4 registers | 0.0127 | 0.1% | 9.8% | 32.3% | 19.1% | 10.4% | 2,706 | 461,710 |
| 99% neurons → 16 registers | 0.0098 | 0% | 5.2% | 48.5% | 14.8% | 20.4% | 2,673 | 458,638 |

For reference, the paired deviation of random token removal is 0.032 at 1/3 (97.5% success) and 0.075 at 2/3 (74.5%), on the same metric (`research/libero_token_mask/RESULTS.md`).

**Where the image queries' attention goes, all layers:**

| Config | Image keys | Task tokens | Template | Registers |
|---|---|---|---|---|
| unmodified | 62.1% | 6.2% | 24.4% | – |
| direction removed, no registers | 78.1% | 2.3% | 11.5% | – |
| direction → 16 registers | 62.7% | 1.0% | 3.6% | see table above |
| 99% neurons → 16 registers | 62.7% | 1.8% | 5.1% | see table above |

The action queries' share on the task tokens drops from 12.9% to 5.8% (direction → 16 registers) and 7.5% (99% neurons → 16 registers).

## Closed Loop

`eval_register.py`, run `runs/20261005_register_grid/`.
- **Episodes:** 200 per config (4 suites × 10 tasks × 5 initial states), with a paired inference of the unmodified model at every call. All 600 episodes completed without a worker restart.
- **Hardware:** one `nnmc71` H100 NVL per 3 workers.
- **Baseline:** the unmodified episodes of the token-mask study.

| Config | All | Spatial | Object | Goal | Long | Lost / gained vs. unmodified | McNemar p | Paired deviation, median (p10–p90) |
|---|---|---|---|---|---|---|---|---|
| unmodified | 100.0% | 100 | 100 | 100 | 100 | – | – | – |
| 99% neurons → 16 registers | 99.0% ± 0.7 | 98 | 100 | 100 | 98 | 2 / 0 | 0.5 | 0.0098 (0.0074–0.0137) |
| direction → 16 registers | 96.0% ± 1.4 | 96 | 100 | 94 | 94 | 8 / 0 | 0.0078 | 0.024 (0.016–0.035) |
| direction removed, no registers | 95.0% ± 1.5 | 92 | 98 | 92 | 98 | 10 / 0 | 0.0020 | 0.037 (0.026–0.056) |

**Failed episodes:**
- **99% neurons:** Spatial task 4 and Long task 9, one each.
- **direction:**
  - Long task 9: three;
  - Goal tasks 0, 1 and 2: one each;
  - Spatial tasks 4 and 5: one each.
- **direction removed:**
  - Goal task 0: two; Goal tasks 2 and 9: one each;
  - Spatial task 4: two; Spatial tasks 1 and 5: one each;
  - Object task 3 and Long task 4: one each.

Spatial task 4 also failed under W4A4 quantization (`research/libero_quant/RESULTS.md`).

Simulated inference takes 0.234 s with 16 registers against 0.228 s for the paired unmodified inference. This +3% covers the 16 extra prefix tokens and the hook; it is not a kernel-level measurement.

## Conclusions

- **A training-free register takes over the sink.**
  - Setup: 16 registers, with the massive update of layer 4 moved onto them (by 99% of the neurons or by direction).
  - Attention: the image queries' attention on the template in layers 5–10 falls from 43% to 2–5%. The registers take 49–56%.
  - Image keys keep their share: 62.7% against 62.1%.
  - Norms: the registers become the high-norm tokens (220,000–460,000). The template tokens drop to the level of task tokens.
- **With the paper's neuron shift, inference is unchanged within noise.**
  - Success rate: 99.0%, 2 lost and 0 gained, p = 0.5.
  - Paired action deviation: median 0.0098. That is a third of the deviation from removing a random third of the image tokens.
- **The registers are what keeps inference intact.**
  - At the same removal of the massive activation, adding registers lowers the deviation at every setting that removes at least 90% of it:
    - direction: 0.038 → 0.024;
    - 99% neurons: 0.017 → 0.0098;
    - 90% neurons: 0.0105 → 0.0066.
  - At 50% the deviations are all below 0.005 and do not order: 0.0040 removed, 0.0030–0.0042 with registers.
  - From 90% on, more registers help (1 → 4 → 16). This fits restoring the sink's total attention: *N* identical register keys act like one key whose logit is higher by log *N*.
  - In closed loop, removing the direction without registers costs 10 episodes (p = 0.002). Moving it to registers costs 8 (p = 0.008).
- **No improvement over the unmodified model.** No config gains a single episode.
  - In ViTs, test-time registers clean artifact tokens in the image. Here the image tokens have no artifacts: the model already uses the fixed prompt template as its registers.
  - A test-time register therefore relocates the sink. At best it matches the unmodified model.
- **Side effect on the instruction tokens.**
  - The task tokens also carry part of the massive direction: their norm is 6,962, and 2,800–3,400 after the shift.
  - Attention on them falls: from 6.2% to 1–2% for image queries, and from 12.9% to 6–7.5% for action queries.
  - The direction-shift and removal failures include LIBERO-Goal tasks, where the instruction alone decides the task. The neuron shift keeps Goal at 100%, so the evidence does not attribute the failures to this drop.

## Open Questions

- **Quantization:** with the massive activations isolated in a few register tokens, those tokens could stay in high precision while the rest are quantized per tensor (PrefixQuant-style).
- **Text pruning:** once the sink sits in registers, the template tokens may become removable.
- **Other paths and models:** the video-generation path (target-frame queries) and the three G1 models were not tested.
