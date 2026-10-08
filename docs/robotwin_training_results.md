# RoboTwin 2.0 Training Results

Runs of `plan.md` Phase 5. The setup and its verification against the released FLUX.2-4B model are in `docs/robotwin_training_setup.md`. The smoke-test measurements are in `plan.md` → Phase 4.

## Runs

| | FLUX.2-klein-base-9B | Qwen-Image-2.1 |
|---|---|---|
| Run directory | `nnmc71:checkpoints/robotwin_flux2_klein_9b_base_b384_imagewam/` | `nnmc73:/home/user1/workspace/jingwu/ImageWAMFT_nnmc73/checkpoints/robotwin_qwenimage21_imagewam/` |
| Launcher | `ZERO=1 GRAD_CLIP=1.0 GPUS=0,1,2,3,4,5,6,7 bash launch/robotwin_flux2_9b_train.sh` | `GRAD_CLIP=1.0 MICRO_BATCH=96 GPUS=0,1,2,3 bash launch/robotwin_qwenimage21_train.sh` |
| tmux session | `imagewamft_robotwin_9b_train` | `imagewamft_robotwin_qwen_train` |
| GPUs | `nnmc71` 8 × H100 NVL | `nnmc73` GPUs 0–3, 4 × B300 |
| Micro-batch × accumulation × GPUs | 4 × 12 × 8 = 384 | 96 × 1 × 4 = 384 |
| DeepSpeed | ZeRO-1, `gradient_clipping` 1.0 | ZeRO-2, `gradient_clipping` 1.0 |
| Optimizer steps | 69,900 (5 epochs), warmup 3,495 | 69,900 (5 epochs), warmup 3,495 |
| Started | 2026-10-08 17:32 EDT | 2026-10-08 16:59 EDT |
| Smoke-test speed | 9.8 s per step, about 7.9 days | 6.4 s per step, about 5.2 days |
| Weights | every 5,000 steps and at the end, 54 GB each | every 5,000 steps and at the end, 15.9 GB each |

## Training

Pending.

## RoboTwin Evaluation

Pending. Blocked for `nnmc72` (ISSUE-010; `plan.md` Phase 6).
