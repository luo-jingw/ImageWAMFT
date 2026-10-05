# G1 Training Results

Results of the three ImageWAM runs on the four LGG100 datasets (`plan.md` Phases 6, 14 and 16). Configuration: `docs/g1_training_setup.md`. Backbone parity: `docs/qwenimage21_parity.md`. Serving: `docs/g1_policy_server.md`.

## Runs

| | FLUX.2-klein-base-4B | Qwen-Image-2.1 | FLUX.2-klein-base-9B |
|---|---|---|---|
| Run directory | `checkpoints/g1_lgg100_flux2_klein_4b_base_imagewam/` | `checkpoints/g1_lgg100_qwenimage21_imagewam/` | `checkpoints/g1_lgg100_flux2_klein_9b_base_imagewam/` |
| GPUs | `nnmc71` GPUs 6, 7 (2 × H100 NVL) | `nnmc72` GPUs 4, 5 (2 × H100 NVL) | `nnmc72` GPUs 0–3 (4 × H100 NVL) |
| Micro-batch × GPUs × accumulation | 4 × 2 × 8 = 64 | 4 × 2 × 8 = 64 | 4 × 4 × 4 = 64 |
| Steps | 27,950 (10 epochs) | 27,950 | 27,950 |
| Wall clock (log start to end, no restart) | 31.3 h (2026-10-01 21:25 – 10-03 04:46) | 51.8 h (10-01 22:40 – 10-04 02:25) | 28.1 h (10-04 02:40 – 10-05 06:46) |
| Speed, median | 0.237 steps/s | 0.150 steps/s | 0.278 steps/s |
| Peak GPU memory (wandb system stats, per GPU) | 60.9 GiB | 84.6 GiB | 86.1 GiB |
| Video expert / action expert parameters | 3.88 B / 0.64 B | 7.12 B / 0.85 B | 9.08 B / 0.95 B |
| Weights | `checkpoints/weights/step_{005000,…,025000,027950}.pt`, 9.0 GB each | 15.9 GB each | 38.9 GB each |

The 9B weight files are saved through the LoRA-merged path (`checkpoint_format: lora_merged`). Each video-expert block is stored under two names, `mixtures.video.transformer.*` and `mixtures.video.{double,single}_blocks.*`. The FLUX.2-4B files share these tensors and store them once. All weights load with 0 missing and 0 unexpected keys.

## Training-Time Validation

- **Setup:** every 500 steps on random samples of the four held-out episodes: 2 samples per evaluation, 4 for the 9B run (one per rank). Last evaluation at step 27,500.
- **Metrics:**
  - `action_l1` and `action_l2` on de-normalized actions;
  - PSNR and SSIM of the predicted frame t + 16 against the true frame (`psnr_rg`, `ssim_rg`);
  - PSNR of the VAE reconstruction against the true frame (`psnr_dg`).
- **Columns:** mean ± standard error of the last 10 evaluations (steps 23,000–27,500), with the final evaluation in parentheses.

| Metric | FLUX.2-4B | Qwen-Image-2.1 | FLUX.2-9B |
|---|---|---|---|
| `val_loss` | 0.174 ± 0.033 (0.021) | 0.093 ± 0.020 (0.011) | 0.176 ± 0.023 (0.131) |
| `action_l1` | 0.0186 ± 0.0022 (0.0132) | 0.0182 ± 0.0021 (0.0150) | 0.0233 ± 0.0029 (0.0135) |
| `action_l2` | 0.0016 ± 0.0005 (0.0004) | 0.0015 ± 0.0004 (0.0006) | 0.0055 ± 0.0034 (0.0004) |
| PSNR, predicted vs. true frame | 21.73 ± 0.60 (21.65) | 21.61 ± 0.43 (22.38) | 21.68 ± 0.57 (21.79) |
| SSIM, predicted vs. true frame | 0.843 ± 0.013 (0.871) | 0.833 ± 0.010 (0.879) | 0.831 ± 0.010 (0.849) |
| PSNR, VAE reconstruction | 43.27 ± 0.08 | 43.44 ± 0.10 | 43.32 ± 0.10 |

- Two to four samples per evaluation make single evaluations and best-step values noise-dominated. Best values over the run reach `action_l1` 0.0088 (4B, step 21,000), 0.0094 (Qwen, step 21,000) and 0.0135 (9B, step 23,500).
- The lower Qwen-Image-2.1 `val_loss` comes from its latent space (`docs/g1_training_setup.md` → Video Loss Across Backbones).
- Training loss over the last 1% of logged steps:

  | Run | Total | Action | Video |
  |---|---|---|---|
  | FLUX.2-4B | 0.084 | 0.0038 | 0.080 |
  | Qwen-Image-2.1 | 0.034 | 0.0042 | 0.030 |
  | FLUX.2-9B | 0.081 | 0.0031 | 0.078 |

## Held-Out Action Error

`deploy/g1_server/replay_check.py --stride 8` through the policy server.

- **Samples:** 316, every 8th frame of the four held-out episodes.
- **Preprocessing:** that of training (image and proprio difference 0, `docs/g1_policy_server.md`), from the raw 480 × 640 frames.
- **Inference:** final weights (step 27,950), 10 denoising steps, seed 42.
- **Metric:** mean absolute error against the recorded actions t..t+15, in raw units. Arm joints are in rad.

| Model | Arm joints | Grippers | drawer (62) | biomanual (46) | simple-pick (26) | Stack-the-cubes (182) |
|---|---|---|---|---|---|---|
| FLUX.2-4B | 0.0209 ± 0.0007 | 0.038 | 0.0356 | 0.0185 | 0.0154 | 0.0172 |
| Qwen-Image-2.1 | 0.0206 ± 0.0007 | 0.044 | 0.0332 | 0.0190 | 0.0155 | 0.0174 |
| FLUX.2-9B | 0.0209 ± 0.0007 | 0.039 | 0.0359 | 0.0182 | 0.0152 | 0.0174 |

- Per-dataset columns are arm-joint errors, with the sample count in parentheses.
- The three backbones give the same arm-joint error within one standard error (0.0007 rad).
- The drawer task has about twice the error of the other three in every model.
- The robot camera path (256 × 320 resize, JPEG quality 90) changes the predicted actions by 0.0010 on average.

## Inference Latency

One H100 NVL, batch 1, 10 denoising steps with the prefix K/V cache, from `docs/g1_policy_server.md`:

| Model | Model time | Request total |
|---|---|---|
| FLUX.2-4B | 156 ms | 170 ms |
| Qwen-Image-2.1 | 237 ms | 258 ms |
| FLUX.2-9B | 202 ms | 215 ms |

## Conclusions

- All three runs completed the full recipe (27,950 steps at effective batch 64) with finite losses and no restart.
- On the held-out episodes, the backbone does not change the offline action error: 0.0206–0.0209 rad for the arm joints. The training-time validation metrics agree within their noise.
- Offline action error does not measure closed-loop success. The physical-robot evaluation remains open.

## Published Weights

`https://huggingface.co/JingwuLuo/ImageWAM3backbones` (public; model card `README.md`, license index `LICENSE.md`). Per run directory, the layout follows `checkpoints/<run>/`:
- `checkpoints/weights/step_*.pt`: all six saved steps;
- `config.yaml` and `dataset_stats.json`;
- `text_cache/`: the four training task strings;
- the license of the base model: Apache-2.0 for FLUX.2-4B; FLUX Non-Commercial License v2.1 with `NOTICE` for FLUX.2-9B; Qwen Research License with `NOTICE` for Qwen-Image-2.1.

Training logs, videos, wandb files and optimizer states are not published. Verified on 2026-10-05: 43 files, 383 GB, with every file size equal to the local file.
