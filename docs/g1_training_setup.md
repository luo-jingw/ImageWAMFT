# G1 Training Setup

Verified configuration for training ImageWAM on the LGG100 G1 data. Data facts are in `docs/lgg100_dataset.md`; the paper recipe is in `docs/imagewam_paper_recipe.md`.

## Configuration Files

| Purpose | File |
|---|---|
| Data (both backbones) | `ImageWAM/configs/data/g1_lgg100.yaml` |
| FLUX.2-klein-base-9B task | `ImageWAM/configs/task/g1_lgg100_flux2_klein_9b_base_imagewam.yaml` |
| FLUX.2 launcher | `ImageWAM/scripts/flux2/run_train_flux2_klein_imagewam.sh` with `TASK_TYPE=g1` |
| Local paths | `ImageWAM/.env.local` (git-ignored; contents in `PROJECT.md`) |

## Data Pipeline

- Roots: `dataset/LGG100/{drawer,biomanual,simple-pick,Stack-the-cubes}`, trained jointly.
- Excluded episodes: `episode_index_filter: {mode: explicit_exclude}` with the 11 invalid episodes.
- Split: episode-level, 1%, seed 42. Training has 335 episodes and 178,864 frames; validation has 4 episodes and 2,539 frames.
- Normalization: z-score using global statistics of the training split, 16 action and 16 state dimensions. The file is written as `dataset_stats.json` in the run directory.
- Sample layout, from `RobotVideoDataset` with `concat_multi_camera: robotwin` and `robotwin_camera_layout: compact_288x256`:

| Key | Shape | Content |
|---|---|---|
| `video` | [3, 2, 288, 256] | Frames t and t+16, values in [-1, 1]. `cam_left_high` 192×256 on top; `cam_left_wrist` and `cam_right_wrist` 96×128 each, bottom left and bottom right |
| `action` | [16, 16] | 16 normalized 16-dim actions from t |
| `proprio` | [16, 16] | Normalized states; ImageWAM uses the first step |
| `text_hidden_states` | [128, 12288] bf16 | Cached Qwen3-8B features (layers 9, 18, 27) of the `DEFAULT_PROMPT`-wrapped task string |
| `text_attention_mask` | [128] bool | True for the prompt tokens |

- Training-time video augmentation follows the RoboTwin data config: color jitter, gamma, Gaussian noise, random resized crop at scale 0.95–1.0 and rotation ±5°, applied with p = 0.5.

## FLUX.2 Text Cache

`ImageWAM/data/g1_lgg100/flux2_qwen3_cache_9b/`, 4 files `<sha256(prompt)>.qwen3_flux2_len128.pt`. True token counts per prompt:

| Dataset | Tokens (of 128) |
|---|---|
| drawer | 63 |
| biomanual | 64 |
| simple-pick | 44 |
| Stack-the-cubes | 61 |

Nothing is truncated at `qwen_context_len` 128.

## Logging

- Training and validation metrics are written to the launcher's standard output by `Trainer`.
- `wandb` runs in offline mode with no workspace (`wandb.mode: offline`, `wandb.workspace: null`), writing under `<run_dir>/wandb/`. No project wandb identity is configured.

## Gradient Clipping

`ImageWAM/scripts/ds_configs/ds_zero{1,2}_config.json` set `"gradient_clipping": 1.0`. Under DeepSpeed, `Accelerator.clip_grad_norm_` (accelerate 1.12.0) only returns `get_global_grad_norm()`. Clipping is applied by DeepSpeed according to its config.

Measured on a ZeRO-2 engine built through `accelerate launch --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml` (2 GPUs):

| DeepSpeed config | `engine.gradient_clipping()` | `optimizer.clip_grad` |
|---|---|---|
| With `gradient_clipping: 1.0` | 1.0 | 1.0 |
| Upstream config without the key | 0.0 (no clipping) | 0.0 |

## Optimizer Parameter Groups (DeepSpeed ZeRO-2 Partition Limit)

DeepSpeed 0.18.5 ZeRO-2 flattens each optimizer param group and gives each rank a contiguous partition. When the per-rank partition exceeds 2^31 elements, the gradients come out non-finite from the first optimizer step. DeepSpeed then masks the norm to −1, and the logged `grad_norm` reads 1. Clipping does not engage, and every later loss is NaN.

`Trainer._split_param_groups` (`optimizer_max_group_numel` in `configs/train.yaml`) splits the trainable parameters into AdamW groups of at most N elements, all with identical hyperparameters. Both G1 task configs set `optimizer_max_group_numel: 1000000000`.

Experiment on FLUX.2-klein-base-4B G1, full fine-tuning of about 4.6B trainable parameters. Fixed: ZeRO-2, bf16, 4 per GPU, gradient accumulation 2, lr 1e-4, `gradient_clipping` 1.0, `overlap_comm` true, H100 NVL, `.venv`. Runs: `ImageWAM/runs/smoke_logs/exp008_{A,B,C}.log`.

| Cell | GPUs | Param groups | Per-rank partition (largest group) | Loss, steps 1–18 |
|---|---|---|---|---|
| A (`nnmc71`) | 2 | 1 | ≈2.3B elements (> 2^31) | step 1: 3.086; steps 2–20: NaN |
| B (`nnmc72`) | 4 | 1 | ≈1.15B elements (< 2^31) | finite, 3.71 → 1.96 |
| C (`nnmc71`) | 2 | 5 (largest 995,773,696) | ≈0.50B elements | finite, 2.95 → 2.51 (range 1.21–4.47) |

The same failure appeared earlier with FLUX.2-9B on 4 GPUs (≈2.5B per rank), while a 9B run with a LoRA video expert (about 1B trainable) on 2 GPUs stayed finite for 200 steps.

## FLUX.2-klein-base-4B Smoke Test

`nnmc71` GPUs 6–7 (H100 NVL), `.venv`, task `g1_lgg100_flux2_klein_4b_base_imagewam`. Settings: ZeRO-2, 4 per GPU, gradient accumulation 8 (effective batch 64), `optimizer_max_group_numel` 1e9, `expandable_segments`, 30 steps, validation at steps 15 and 30. Log: `ImageWAM/runs/smoke_logs/smoke_flux2_4b.log`.

| Measure | Value |
|---|---|
| Loss, steps 1–30 | all finite: 2.41 at step 1, 1.10–1.55 at steps 25–30 |
| Validation, step 15 (1 sample) | `val_loss` 2.361, PSNR 15.88, `action_l2` 0.294, `action_l1` 0.340 |
| Validation, step 30 (1 sample) | `val_loss` 3.151, PSNR 14.16, `action_l2` 0.463, `action_l1` 0.407 |
| Throughput | 0.18 optimizer steps/s, i.e. 5.5 s and 64 samples per step (11.6 samples/s). The trainer's "samples/s" field omits gradient accumulation. |
| Peak GPU memory | 62.0 GB per GPU |
| Checkpoint size | weights 8.5 GB per save; resumable state 60 GB |
| Projected full run | 27,950 steps × 5.5 s ≈ 43 h plus validation every 500 steps |

## FLUX.2-klein-base-4B Full Run (Phase 6)

- Launch: `launch/g1_flux2_4b_train.sh`, `tmux` session `imagewamft_flux4b`, `nnmc71` GPUs 6,7 (an NVLink pair), `.venv`.
- Output: `checkpoints/g1_lgg100_flux2_klein_4b_base_imagewam/`, with `train.log`, `config.yaml`, `dataset_stats.json`, `eval/`, offline `wandb/`, and `checkpoints/{weights,state}/`.
- Recipe: the smoke-test settings for 27,950 steps (10 epochs). Weights are saved every 5,000 steps; only the latest resumable state is kept.

| Measure (first 500 steps) | Value |
|---|---|
| Loss | all finite; 3.03 at step 10, 0.22 at step 500 (λ·action 0.080, λ·video 0.136) |
| Throughput | 0.26–0.27 optimizer steps/s; per micro-batch: forward 0.11 s, backward 0.36 s |
| Validation, step 500 (2 samples, one per rank) | `val_loss` 0.1645, PSNR 16.97, SSIM 0.756, `action_l2` 0.0601, `action_l1` 0.1215 |
| Projected wall clock | about 29 h plus validation |

## Qwen-Image-2.1 Smoke Test (Phase 13)

`nnmc72` GPUs 4–5 (H100 NVL, an NVLink pair), `.venv` through the `nnmc72` symlink, task `g1_lgg100_qwenimage21_imagewam`, launcher `ImageWAM/scripts/qwenimage21/run_train_qwenimage21_imagewam.sh`. The settings match the FLUX.2-4B smoke test:
- ZeRO-2 with the same DS JSON, 4 per GPU, gradient accumulation 8 (effective batch 64), `optimizer_max_group_numel` 1e9 (9 groups, largest 996,696,576);
- `expandable_segments`, 30 steps, validation at steps 15 and 30.

Model-specific settings: whole-layer gradient checkpointing, and normalization statistics from the FLUX.2 run. Logs: `ImageWAM/runs/smoke_logs/smoke_qwenimage21_{a,b_offload,c_probe,d_profile,e}.log`.

| Run | Change | Throughput | Notes |
|---|---|---|---|
| a | none | 0.08 steps/s; per micro-batch forward 0.18 s, backward 1.5 s | 30 finite steps; stopped before the step-30 save |
| b | ZeRO-2 CPU optimizer offload + `DeepSpeedCPUAdam` | 17–20 s per step | stopped after 6 steps; GPU memory 23.6 GB |
| c | allocator probe, `NCCL_DEBUG=INFO` | 0.08 steps/s | peak allocated 78.2 GiB, 0 allocation retries, 8.5 GiB free; NCCL over P2P/IPC |
| d | torch-profiler window on rank 0 | — | 3.8 s CPU vs 1.3 s GPU over 2 backward calls; 707k `view_as` in DeepSpeed's gradient hooks (ISSUE-009) |
| e | `deepspeed_cache_used_param_count` (ISSUE-009 fix) | **0.15 steps/s**; forward 0.18 s, backward 0.69 s | final configuration, below |

Run e results:

| Measure | Value |
|---|---|
| Loss | all finite: 2.44 at step 10 (λ·action 2.38, λ·video 0.062), 1.97 at step 20, 1.12 at step 30 |
| Validation, step 15 (2 samples) | `val_loss` 1.603, PSNR 16.98, SSIM 0.700, `action_l2` 0.345, `action_l1` 0.354 |
| Validation, step 30 (2 samples) | `val_loss` 3.036, PSNR 13.97, SSIM 0.613, `action_l2` 0.299, `action_l1` 0.352 |
| Peak GPU memory | 78.2 GiB allocated per GPU (run c) |
| Checkpoint | weights 15 GB per save, which reload strictly into a fresh model (`mot` and `proprio_encoder`); resumable state 106 GB |
| Projected full run | 27,950 steps / 0.15 steps/s ≈ 52 h plus validation |

Single-GPU reference without DeepSpeed (batch 4, bf16): forward 0.18 s, backward 0.47 s with whole-layer checkpointing (peak 32 GiB with weights and gradients). Without checkpointing the backward is 0.32 s, but the peak is 55.7 GiB, which leaves no room for the optimizer state on 94 GB.

## Qwen-Image-2.1 Full Run (Phase 14)

- Launch: `launch/g1_qwenimage21_train.sh`, `tmux` session `imagewamft_qwenimage21` on **`nnmc72`** (`ssh nnmc72`, then `tmux attach -t imagewamft_qwenimage21`), GPUs 4,5 (an NVLink pair), `.venv` through the `nnmc72` symlink. Started 2026-10-01 22:39.
- Output: `checkpoints/g1_lgg100_qwenimage21_imagewam/`. It holds `train.log`, `config.yaml`, `dataset_stats.json` (copied from the FLUX.2-4B run, D8), `eval/`, offline `wandb/` and `checkpoints/{weights,state}/`.
- Recipe: identical to the FLUX.2-4B run, 27,950 steps with weights every 5,000 steps.
- Resolved config vs. the FLUX.2-4B run (`diff` of the two `config.yaml`):
  - Backbone-bound entries: model target and weight paths, the action-DiT shape and initialization file, and the text cache directory and format.
  - `pretrained_norm_stats` points to the FLUX.2 run's statistics.
  - Settings that do not change the computed values: `gradient_checkpointing: true`, `deepspeed_cache_used_param_count: true` (ISSUE-009), and `optimizer_impl: torch_adamw`, which is the same AdamW the FLUX.2 run constructs.
- First steps: loss 3.02 at step 10 (FLUX.2-4B: 3.03) and 2.44 at step 20. Speed 0.15 steps/s, 86 GB per GPU.

| Measure (first 500 steps) | Qwen-Image-2.1 | FLUX.2-4B (same steps) |
|---|---|---|
| Loss | all finite; 3.02 at step 10, 0.128 at step 500 (λ·action 0.076, λ·video 0.051) | 3.03 at step 10, 0.216 at step 500 |
| Throughput | 0.15 steps/s; per micro-batch forward 0.18 s, backward 0.69 s | 0.27 steps/s |
| Validation, step 500 (2 samples, one per rank) | `val_loss` 0.1194, PSNR 16.24, SSIM 0.726, `action_l2` 0.0939, `action_l1` 0.1593 | `val_loss` 0.1645, PSNR 16.97, SSIM 0.756, `action_l2` 0.0601, `action_l1` 0.1215 |
| Projected wall clock | about 52 h plus validation | about 29 h |

Validation uses 2 random samples per evaluation, so single-step comparisons are noisy.

## Predicted-Frame Time Offset (2026-10-02)

Question: the validation images show predicted frames with correct object shapes and textures but slightly displaced arms and objects, as if shifted by a few frames. Is the prediction systematically early or late, and is the data misaligned?

Script: `ImageWAM/scripts/analyze_prediction_time_offset.py`, run on `nnmc71` GPU 1 against the saved weights while both runs continued training. Results are in `ImageWAM/runs/analysis/time_offset_*.json`.

**Data alignment** (4 validation episodes; lag of best correlation between a camera region's frame-to-frame change and the matching arm's joint speed; positive means the video is later than the state):

| Episode | head | left wrist | right wrist |
|---|---|---|---|
| drawer | +3 | +4 | +2 |
| biomanual | +2 | +3 | −1 |
| simple-pick | +4 | +4 | +3 |
| Stack-the-cubes | +3 | +3 | see JSON |

The video lags the recorded joint state by about 3 frames (100 ms), with the cameras at roughly the same lag as one another. Together with the action leading the state by 3 frames (`docs/lgg100_dataset.md`), the image shows a commanded action about 6 frames later. The lag is the same in training and validation, so it does not shift the validation comparison. It matters for deployment, where proprio and images must keep the recorded relative timing.

**Model offset**: 40 validation samples, 10 per episode. Each prediction of frame t+16 is compared with ground-truth frames t+16+k, k ∈ [−8, 8], by pixel MSE on the full tiling.
- `best k` is the per-sample k with the lowest error; negative means the predicted scene is behind the ground truth.
- The `copy current frame` baseline correctly lands at the boundary k = −8.

| Model, checkpoint | Sampling | Argmin of mean error curve | Median best k | PSNR at k = 0 | PSNR at best k | Seed disagreement in best k |
|---|---|---|---|---|---|---|
| Qwen-Image-2.1, step 5000 | 10 steps, seed 0 / 1 | −3 / −5 | −3.5 / −4.0 | 19.77 / 20.13 | 20.70 / 21.33 | 3.25 |
| Qwen-Image-2.1, step 5000 | 30 steps, seed 0 | −3 | −3.5 | 19.52 | 20.48 | — |
| FLUX.2-4B, step 5000 | 10 steps, seed 0 / 1 | −4 / −5 | −3.5 / −3.5 | 19.70 / 19.74 | 20.85 / 20.86 | 2.92 |
| FLUX.2-4B, step 5000 | 30 steps, seed 0 | −4 | −5.0 | 19.58 | 20.71 | — |
| FLUX.2-4B, step 10000 | 10 steps, seed 0 / 1 | −2 / −3 | −2.0 / −3.0 | 20.37 / 20.09 | 21.69 / 21.33 | 1.93 |
| Copy current frame | — | −8 (boundary) | −8 | 18.87 | 21.97 at k = −8 | — |

Conclusions:
- Both backbones predict a scene that is behind the ground truth, by about 3–4 frames at step 5000. The motion is under-predicted; the timeline is not misindexed.
- The eval pipeline compares frame t+16 with the prediction of frame t+16, and adjacent dataset indices are consecutive frames.
- More sampling steps (30 vs 10) do not change the offset, so it is not a sampler-discretization effect.
- For the same input, two seeds differ by about 3 frames in best k, so a single validation image carries a large stochastic timing spread.
- From FLUX.2 step 5000 to 10000 the offset shrinks to about 2–3 frames, the seed spread to about 2 frames, and PSNR at k = 0 rises by about 0.5 dB. The bias is decreasing with training.
- By design (D2) the image branch does not attend to the action tokens. It must infer the 16-frame progress from one frame, the joint state and the instruction, with no velocity input. A conservative, slower predicted progress is the expected early-training behavior.
