# RoboTwin 2.0 Training Results

Runs of `plan.md` Phase 5. The setup and its verification against the released FLUX.2-4B model are in `docs/robotwin_training_setup.md`. The smoke-test measurements are in `plan.md` → Phase 4.

## Runs

| | FLUX.2-klein-base-9B | Qwen-Image-2.1 |
|---|---|---|
| Run directory | `nnmc71:checkpoints/robotwin_flux2_klein_9b_base_b384_imagewam/` | `nnmc73:/home/user1/workspace/jingwu/ImageWAMFT_nnmc73/checkpoints/robotwin_qwenimage21_imagewam/` |
| Launcher | `ZERO=1 GRAD_CLIP=1.0 GPUS=0,1,2,3,4,5,6,7 bash launch/robotwin_flux2_9b_train.sh` | `GRAD_CLIP=1.0 MICRO_BATCH=48 GPUS=0,1,2,3,4,5,6,7 bash launch/robotwin_qwenimage21_train.sh` |
| tmux session | `imagewamft_robotwin_9b_train` | `imagewamft_robotwin_qwen_train` |
| GPUs | `nnmc71` 8 × H100 NVL | `nnmc73` 8 × B300 |
| Micro-batch × accumulation × GPUs | 4 × 12 × 8 = 384 | 48 × 1 × 8 = 384 |
| DeepSpeed | ZeRO-1, `gradient_clipping` 1.0 | ZeRO-2, `gradient_clipping` 1.0 |
| Optimizer steps | 69,900 (5 epochs), warmup 3,495 | 69,900 (5 epochs), warmup 3,495 |
| Started | 2026-10-08 17:32 EDT | 2026-10-08 20:45 EDT |
| Measured speed | 9.9 s per step (steps 620–1,120, one evaluation included), about 8.0 days | 4.2 s per step (steps 30–130), about 3.4 days |
| Peak GPU memory | 76.6 GB | 55.3 GB |
| Weights | every 5,000 steps and at the end, 54 GB each | every 5,000 steps and at the end, 15.9 GB each |

Run changes:
- Qwen-Image-2.1 first ran on `nnmc73` GPUs 0–3 with 96 × 1 × 4 from 2026-10-08 16:59 EDT at 6.47 s per step (steps 1,000–2,000). On the user's request it was stopped at step 2,000, before its first saved state, and restarted from step 0 on all 8 GPUs. The stopped run is kept as `robotwin_qwenimage21_imagewam.4gpu_stopped/` next to the run directory.
- 8 GPUs give 1.55× the 4-GPU speed. Forward plus backward halves (5.9 s to 3.0 s per step); the remaining time per step grows from 0.5 s to about 1.1 s. `nnmc73` has no active NVLink (PROJECT.md), so the gradient and parameter exchange runs over PCIe and across sockets.
- FLUX.2-9B moves to the 8 `nnmc73` B300 when Qwen-Image-2.1 finishes, resuming from its latest saved state (`plan.md` D11).

## Training

Pending.

## RoboTwin Evaluation

Pending. Blocked for `nnmc72` (ISSUE-010; `plan.md` Phase 6).
