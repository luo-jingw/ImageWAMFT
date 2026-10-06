# Extreme Quantization of ImageWAM on LIBERO: Results

Design and quantization settings: `PLAN.md`. Figures and every number below: `figures/` (`make_figures.py`), `figures/summary.json`.

## Open-Loop Screen

`screen_quant.py`, output `runs/screen.json`.
- **States:** 60 restored states. Each quantized inference uses the same observation and initial noise as the bfloat16 one.
- **Deviation:** mean absolute difference of the normalized 16 × 7 action chunk.
- **Gripper flips:** share of the 12 executed steps whose binarized gripper command differs.

| Config | Deviation, mean | Median | Gripper flips |
|---|---|---|---|
| rotation only, no quantization | 0.0010 | 0.0010 | 0% |
| W8A8 RTN | 0.0057 | 0.0056 | 0.1% |
| W8A8 rotation | 0.0017 | 0.0017 | 0% |
| W4A16 rotation (weights only) | 0.0075 | 0.0073 | 0% |
| W16A4 rotation (activations only) | 0.0203 | 0.0198 | 0.3% |
| W4A4 RTN | 0.178 | 0.177 | 1.4% |
| W4A4 SmoothQuant | 0.056 | 0.054 | 0.4% |
| W4A4 rotation | 0.022 | 0.022 | 0% |
| W4A4 SmoothQuant + rotation | 0.015 | 0.015 | 0.1% |
| W3A3 rotation | 0.059 | 0.057 | 0.6% |
| W2A16 rotation (weights only) | 0.048 | 0.048 | 0.7% |
| W16A2 rotation (activations only) | 0.340 | 0.305 | 29% |
| W2A2 RTN | 0.409 | 0.406 | 52% |
| W2A2 rotation | 0.359 | 0.312 | 33% |
| W2A2 SmoothQuant + rotation | 0.388 | 0.406 | 60% |

For reference, the paired deviation of random token removal is 0.032 at 1/3 (97.5% success) and 0.075 at 2/3 (74.5%), on the same metric (`research/libero_token_mask/RESULTS.md`).

## Closed Loop

`eval_quant.py`, run `runs/20261005_quant_grid/`.
- **Episodes:** 200 per config (4 suites × 10 tasks × 5 initial states), with a paired bfloat16 inference at every call. Every one of the 400 episodes completed without a worker crash.
- **Baseline:** the bfloat16 episodes of the token-mask study.
- **Hardware:** W4A4 ran on one H100 NVL. W2A2 ran 135 episodes on the H100 and 65 on L40S (`nnmc70`).

| Config | All | Spatial | Object | Goal | Long | Lost / gained vs. bfloat16 | McNemar p | Paired deviation, median (p10–p90) |
|---|---|---|---|---|---|---|---|---|
| bfloat16 | 100.0% | 100 | 100 | 100 | 100 | – | – | – |
| W4A4 rotation | 98.0% ± 1.0 | 94 | 100 | 98 | 100 | 4 / 0 | 0.125 | 0.021 (0.018–0.026) |
| W2A2 rotation | 0% | 0 | 0 | 0 | 0 | 200 / 0 | 1e-60 | 0.198 (0.163–0.253) |

- The four W4A4 failures are one episode each in Spatial tasks 4, 5 and 9 and in Goal task 9.
- Simulated inference takes 0.35 s (W4A4) and 0.38 s (W2A2) against 0.24 s for bfloat16. This is the cost of simulation (rotation and quantize-dequantize in float32), not a speed measurement.

## Conclusions

- **W4A4 with rotation keeps LIBERO success:** 98.0% against 100%, not a significant difference (4 lost, 0 gained, p = 0.125). Its paired action deviation (median 0.021) is below that of removing a random third of the image tokens.
- **Rotation is what makes W4A4 work.**
  - The open-loop deviation of W4A4 is 0.178 with plain RTN, 0.056 with SmoothQuant and 0.022 with rotation.
  - SmoothQuant before rotation lowers it further to 0.015. This variant was not run in closed loop.
- **At W4A4 the activations carry most of the error:** 0.020 with activations alone against 0.0075 with weights alone.
- **W2A2 fails completely** (0 of 200), and the 2-bit activations cause it.
  - With 2-bit weights alone (W2A16) the deviation is 0.048, close to random removal of half the tokens.
  - With 2-bit activations alone (W16A2) the deviation is 0.34 and 29% of the gripper commands flip.
  - Three activation levels per token cannot carry the signal, and neither rotation nor SmoothQuant changes that.

## Open Questions

- Closed-loop W2A16 or W2A8: a no-training 2-bit-weight setting whose open-loop deviation is in the usable range.
- Closed-loop W4A4 with SmoothQuant plus rotation.
- Mixed precision: per-layer activation sensitivity, and lower precision for the action expert, which runs 10 times per inference, than for the video-expert prefix.
- 2-bit activations likely need training: quantization-aware training with learned step sizes and distillation from the bfloat16 model.
- Real kernels are hardware-specific. There are no 2-bit tensor cores. W4A4 runs as INT4 on Ada (L40S) and as NVFP4 on Blackwell (B300); H100 offers FP8/INT8 only.
