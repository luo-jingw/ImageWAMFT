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

## Action DiT (`ActionDiTQwenImage21`)

`ImageWAM/src/imagewam/models/backbones/action_dit_qwenimage21.py`; initialization by `ImageWAM/scripts/qwenimage21/preprocess_action_dit_qwenimage21.py` → `ImageWAM/checkpoints/action_dit_qwenimage21_g1_init.pt` (bf16, 1.7 GB).
- Layout: 32 `QwenImage21Block`s with residual width 1024, attention width 4096 (32 heads × 128, shared with the video expert), SwiGLU 1024 → 3072. One shared modulation, and a fresh scale-only AdaLN head projecting to 16.
- Parameter count: 845,456,384.
- Copied unchanged (64 tensors): the per-head QK-norm weights.
- Interpolated from the video transformer (227 tensors). Alpha scaling sqrt(src_in / dst_in) = 2 is applied wherever the input axis shrinks from 4096 to 1024:
  - `to_q`/`to_k`/`to_v` (4096, 4096) → (4096, 1024);
  - `to_out.0` (4096, 4096) → (1024, 4096);
  - `img_mlp.proj`/`gate_layer` (12288, 4096) → (3072, 1024);
  - `img_mlp.out` (4096, 12288) → (1024, 3072);
  - `modulation.1` (16384, 4096) → (4096, 1024);
  - timestep MLP (4096, 256) → (1024, 256) and (4096, 4096) → (1024, 1024).
- Fresh: `action_encoder` (Linear 16 → 1024) and `head` (`norm_out.linear`, `proj_out`), as for the FLUX.2 action DiT.
- Check, bf16 on one H100: `prepare_qkv` returns q/k/v of shape [B, L, 4096]. A 32-block forward on random actions gives a finite [2, 16, 16] output.

## ImageWAM Stack Integration (Phase 12)

Check: `ImageWAM/scripts/qwenimage21/check_stack_qwenimage21.py` on one H100 with `.venv`. The model is built from the task config `g1_lgg100_qwenimage21_imagewam`, using validation samples and the FLUX.2 run's normalization statistics.
- **Attention mask** for the layout [2 text tokens + 1 padded | 2 reference | 2 target | 2 action] matches D2:
  - text and reference rows attend to text and reference;
  - target rows attend to text, reference and target;
  - action rows attend to text, reference and action;
  - the padded text column is masked everywhere.
- **One training batch** (2 samples, bf16, trainer freeze policy):
  - loss 2.207 (λ·video 0.030, λ·action 2.177), finite.
  - Every trainable tensor gets a non-zero gradient: video expert 297/297, action expert 295/295, proprio encoder 2/2.
  - The VAE has `requires_grad=False` and no gradients.
  - Peak memory is 30.9 GiB: bf16 weights and gradients, with whole-layer gradient checkpointing.
- **`infer_action`** (10 steps) returns a finite [16, 16] action.
- **Prefix cache:** for the same noisy action and timestep, the action prediction from the cached t = 0 prefix (`prefill_video_cache` + `forward_action_with_video_cache`) is compared with a full joint forward over [text | reference | noisy target | action]:
  - fp32: max abs diff 6.3e-6, against a prediction max of 1.9;
  - bf16: max abs diff 1.0e-1, against a prediction max of 2.0. This is rounding accumulated over 32 layers, since the prefix runs at a different sequence length.

Layout notes:
- The text segment is the cached 128-token Qwen3-VL features plus the proprio token packed after the valid text: 129 positions with right padding, the same packing as the FLUX.2 stack.
- The reference block therefore starts at rotary position 129 for every prompt, not directly after the valid text as in the native pipeline.
