# Qwen-Image-2.1 Parity

Reference: diffusers git commit `578c9b2c6636ab2424a0e56186268b83623656b2`, run in `.venv-qwenimage`. Fixtures come from `ImageWAM/scripts/qwenimage21/dump_reference_fixtures.py` and are stored in `ImageWAM/data/qwenimage21_fixtures/`.

Fixture inputs are one G1 validation frame pair (288×256, RoboTwin tiling) and the drawer instruction wrapped in `DEFAULT_PROMPT`.

The text-to-image Qwen3-VL features use the last decoder layer before the final norm, with system tokens dropped. Token counts for drawer, biomanual, simple-pick and Stack-the-cubes are 59, 60, 40 and 57.

## Transformer (`QwenImage21VideoExpert`)

Check: `ImageWAM/scripts/qwenimage21/check_parity_dit.py` on one H100 with `.venv`.
- Layout: [59 text tokens | 288 reference tokens | 288 target tokens], native mask (token-causal text, bidirectional image blocks).
- Inputs: t = 0.6; the target latent is 0.4·x0 + 0.6·noise.
- Parameter count: 7,115,124,736, with strict key match against the released checkpoint.

| Quantity | Max abs diff | Max rel diff |
|---|---|---|
| `txt_in` output (fp32) | 1.10e-3 | 4.1e-7 |
| `img_in` output (fp32) | 0 | 0 |
| timestep embedding, target row (fp32) | 0 | 0 |
| block outputs 0, 1, 7, 15, 23, 31 (fp32) | ≤ 4.9e-3 | ≤ 6.3e-7 |
| target prediction (fp32) | 1.79e-5 | 2.2e-6 |
| target prediction, ours bf16 vs diffusers bf16 | 6.25e-2 | 7.6e-3 |
| target prediction, ours bf16 vs diffusers fp32 | 6.56e-2 | 8.0e-3 |
| target prediction, diffusers bf16 vs diffusers fp32 (reference spread) | 5.62e-2 | 6.8e-3 |

Relative diff is the max abs diff divided by the reference's max abs value.

## Zero-Shot Edit

`ImageWAM/data/qwenimage21_fixtures/edit_sample.png`: `QwenImage21Pipeline` at 288×256 (`output_resolution` 272, 40 steps) on the current frame with the drawer instruction. The output keeps the three-view tiling (head view on top, two wrist views below) and shows the right arm holding the purple cube toward the open drawer.
