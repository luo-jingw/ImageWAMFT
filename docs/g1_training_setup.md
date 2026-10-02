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
