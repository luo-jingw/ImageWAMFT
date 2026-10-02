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
