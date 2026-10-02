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

## VAE (`QwenImage21VAE`)

Vendored from diffusers `AutoencoderKLQwenImage21` (single-image encode/decode path) into `ImageWAM/src/imagewam/models/backbones/qwenimage21_vae.py`.
- `encode` takes RGB in [-1, 1], appends an opaque alpha channel and returns the posterior mode normalized per channel with `latents_mean` / `latents_std`, shape [B, 64, H/16, W/16].
- `decode` inverts both steps and clamps the output to [-1, 1].
- Parameter count: 337,740,404 (encoder 78,675,680, decoder 259,044,052), with strict key match.

Check: `ImageWAM/scripts/qwenimage21/check_parity_vae.py`, fp32 on CPU.
- Vendored vs diffusers, both on CPU: latents and reconstructions are bitwise identical (`torch.equal`) for both fixture frames.
- Against the stored `vae.pt`: latents max abs 2.99e-2 / 1.18e-1 (current / target), reconstruction max abs 3.16e-3 / 4.44e-3. Diffusers on CPU shows exactly the same gap to the fixture.
- Cause of that gap: the fixture was dumped on GPU, where cuDNN convolutions use TF32 by default. Rounding the conv inputs and weights to TF32 on CPU reproduces a gap of the same size.
- The transformer fixture is unaffected: matmul TF32 is off by default, and the fp32 DiT parity above is at float-noise level.

Reconstruction PSNR on 16 G1 tiled frames (frames t and t+16 of 8 validation samples spread over all four datasets, 288×256, one H100, TF32 off):

| VAE dtype | Mean PSNR | Min | Max |
|---|---|---|---|
| fp32 | 44.58 dB | 42.99 dB | 45.76 dB |
| bf16 | 44.36 dB | 42.81 dB | 45.57 dB |

The normalized latents of these frames have mean −0.065 and std 1.12.

## Text Cache

`ImageWAM/scripts/qwenimage21/precompute_qwenimage21_text_embeds.py` (run in `.venv-qwenimage`) writes one file per prompt to `ImageWAM/data/g1_lgg100/qwenimage21_text_cache/`.
- File name: `<sha256(prompt)>.qwen3vl_qwenimage21_len128.pt`.
- Contents: `text_hidden_states` [128, 4096] bf16, zero-padded, and `text_attention_mask` [128] bool.
- The cached features equal `text_embeds.pt` exactly (max abs diff 0) for all four prompts. The padding rows are all zero.
- `RobotVideoDataset` with `qwen_text_cache_format: qwen3vl_qwenimage21` returns these tensors for all four instructions. Valid lengths are 59, 60, 40 and 57.
