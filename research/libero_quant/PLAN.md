# Extreme Quantization of ImageWAM on LIBERO

## Question

How does low-bit weight and activation quantization of the ImageWAM-FLUX.2-4B transformer change the LIBERO success rate, from W4A4 down to W2A2?

## Mechanism Under Test

`fake_quant.py`: simulated (quantize-dequantize) quantization. It measures accuracy only, not speed; real W4A4 speed needs dedicated kernels (for example SVDQuant/Nunchaku for FLUX).

| Item | Setting |
|---|---|
| Quantized | the 140 linears inside the transformer blocks of both experts (4.31 B weights): double-block img/txt qkv, proj and MLP, single-block linear1 and linear2 |
| Kept in bfloat16 | embedders, modulation linears, final layers, attention score and value products, VAE, Qwen3 text encoder |
| Weights | symmetric per output channel; the clipping ratio of each row is searched over 1.0 → 0.3 for minimum squared error (RTN with clip search, no GPTQ) |
| Activations | symmetric per token, dynamic, clip 1.0 |
| Levels | b bits use -(2^(b-1) - 1) … 2^(b-1) - 1: 15 values at 4 bits, 3 at 2 bits |
| Rotation (QuaRot-style) | each linear's input is multiplied by a block-diagonal randomized Hadamard matrix (1024 × 1024 blocks, fixed signs), and the weight by the same matrix; exact in full precision |
| SmoothQuant | per-input-channel scales s = max\|X\|^α / max\|W\|^(1-α), α = 0.5, from calibration activations |

Full QuaRot also rotates the residual stream. That needs normalization layers that are invariant to rotation. FLUX.2 blocks use LayerNorm with per-channel adaLN modulation, so only the per-linear (online) rotation is applied.

## Grid

1. **Open-loop screen** (`screen_quant.py`), all configs in `fake_quant.CONFIGS`.
   - 60 restored simulator states: tasks 0–4 of every suite, calls 1, 3 and 6 of the no-reduction episodes of the token-mask study.
   - Measures: action deviation from the full-precision inference with the same noise, and the gripper sign-flip rate.
   - SmoothQuant calibration uses 40 disjoint states (tasks 5–9).
2. **Closed loop** (`eval_quant.py`): `rot_w4a4` and `rot_w2a2`, 200 episodes each (4 suites × 10 tasks × 5 initial states).
   - Settings as in the token-mask study: OSMesa, seed 42, 10 denoising steps, replanning every 12 steps.
   - At every call, a paired full-precision inference from the same observation and noise is recorded.
   - The full-precision success rate (100%, 200 episodes) comes from the token-mask study.

## Environment

`.venv-libero` on `nnmc71` GPU 0, the only GPU free of other users' jobs at the time.
