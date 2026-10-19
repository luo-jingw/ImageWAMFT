# RoboTwin 2.0 Training Results

Training runs and RoboTwin evaluation of the FLUX.2-9B and Qwen-Image-2.1 models. The setup and its verification against the released FLUX.2-4B model are in `docs/robotwin_training_setup.md`. The smoke-test measurements are in the training plan, Phase 4 (`git show 8f02eb3:plan.md`).

## Runs

| | FLUX.2-klein-base-9B | Qwen-Image-2.1 |
|---|---|---|
| Run directory | `nnmc71:checkpoints/robotwin_flux2_klein_9b_base_b384_imagewam/` | `nnmc73:/home/user1/workspace/jingwu/ImageWAMFT_nnmc73/checkpoints/robotwin_qwenimage21_imagewam/` |
| Launcher | to step 42,150: `ZERO=1 GRAD_CLIP=1.0 GPUS=0,1,2,3,4,5,6,7 bash launch/robotwin_flux2_9b_train.sh`; from the step-40,000 state: `GRAD_CLIP=1.0 ZERO=2 GPUS=0,1,2,3 bash launch/robotwin_flux2_9b_train.sh optimizer_max_group_numel=1000000000` (first launch with `LOAD_UNIVERSAL=1 resume=<run>/checkpoints/state_universal/step_040000`, deleted after the step-45,000 state was checked) | `GRAD_CLIP=1.0 MICRO_BATCH=48 GPUS=0,1,2,3,4,5,6,7 bash launch/robotwin_qwenimage21_train.sh` |
| tmux session | `imagewamft_robotwin_9b_train` | `imagewamft_robotwin_qwen_train` |
| GPUs | `nnmc71` 8 × H100 NVL, then `nnmc71` GPUs 0–3 | `nnmc73` 8 × B300 |
| Micro-batch × accumulation × GPUs | 4 × 12 × 8 = 384, then 4 × 24 × 4 = 384 | 48 × 1 × 8 = 384 |
| DeepSpeed | ZeRO-1, then ZeRO-2 with 11 optimizer groups; `gradient_clipping` 1.0 | ZeRO-2, `gradient_clipping` 1.0 |
| Optimizer steps | 69,900 (5 epochs), warmup 3,495 | 69,900 (5 epochs), warmup 3,495 |
| Started | 2026-10-08 17:32 EDT | 2026-10-08 20:45 EDT |
| Measured speed | 9.9 s per step on 8 GPUs (steps 620–1,120, one evaluation included); 19.8 s per step on 4 GPUs (steps 42,500–48,460, evaluations included) | 4.2 s per step (steps 30–130), about 3.4 days |
| Peak GPU memory | 76.6 GB on 8 GPUs; 88.2 GB used per GPU on 4 GPUs (`nvidia-smi`) | 55.3 GB |
| Weights | every 5,000 steps and at the end, 54 GB each | every 5,000 steps and at the end, 15.9 GB each |

Run changes:
- Qwen-Image-2.1 first ran on `nnmc73` GPUs 0–3 with 96 × 1 × 4 from 2026-10-08 16:59 EDT at 6.47 s per step (steps 1,000–2,000). On the user's request it was stopped at step 2,000, before its first saved state, and restarted from step 0 on all 8 GPUs. The stopped run is kept as `robotwin_qwenimage21_imagewam.4gpu_stopped/` next to the run directory.
- 8 GPUs give 1.55× the 4-GPU speed. Forward plus backward halves (5.9 s to 3.0 s per step); the remaining time per step grows from 0.5 s to about 1.1 s. `nnmc73` has no active NVLink (PROJECT.md), so the gradient and parameter exchange runs over PCIe and across sockets.
- FLUX.2-9B moved from 8 to 4 `nnmc71` GPUs on 2026-10-13 (training plan D11, `git show 8f02eb3:plan.md`). The 8-GPU run was stopped at about step 42,150 (14:17 EDT) and resumed from the step-40,000 state on GPUs 0–3 at 14:21 EDT, so steps 40,001–42,150 were trained twice.
  - The 8-rank ZeRO-1 state was converted by `ImageWAM/scripts/reshard_training_state.py` in 2.8 min: 355 parameters with fp32 weights and both Adam moments, 140.6 GB, `batch_in_epoch` 144,480 → 288,960 (sample offset 4,623,360 in both runs).
  - The resumed run read the same samples as the first run. Over steps 40,001–42,150, training loss is 0.0790 resumed against 0.0806 originally (action 0.0029 against 0.0028). The 7 validations from step 40,500 to 43,500 average `action_l1` 0.0127 and PSNR 28.6 dB, against 0.0122 ± 0.0052 and 27.5 ± 1.9 dB for the first run's validations at steps 38,000–42,000.
  - The logged `train/grad_norm` is 2.0× the first run's over the same steps (ISSUE-012).
  - The first 4-rank state (step 45,000, saved 2026-10-14 17:54 EDT, 132 GB) was read back: 4 shards with 11 optimizer groups each, largest partition 249,237,952 elements, no non-finite value in the fp32 weights or Adam moments, Adam step 45,000, learning rate 3.155e-5 (the cosine schedule with its 1e-6 floor). The converted step-40,000 state was then deleted; the 8-rank step-40,000 state had been pruned by the step-45,000 save.
- Qwen-Image-2.1 was killed at step 67,110 on 2026-10-18 07:53 EDT by the owner of the `nnmc_ww` jobs that share `nnmc73` (ISSUE-013). Its last saved state is step 65,000.
  - The run is not continued. The step-65,000 weights are the final Qwen-Image-2.1 model; the learning rate at that step is 2.32e-6.
  - On 2026-10-19, `config.yaml`, `dataset_stats.json`, `train.log`, `wandb/`, `eval/` and `checkpoints/weights/step_065000.pt` were copied to `nnmc71:checkpoints/robotwin_qwenimage21_imagewam/`. The weight file has the same SHA-256 on both nodes (`b55fcd47…3a834d`).
  - The weight file holds 594 bfloat16 tensors, all finite, with `step` 65,000. It loads into the model built from `config.yaml` with 0 missing and 0 unexpected keys (CPU check).
- A move to the 8 `nnmc72` H100 was not possible: another project's eviction script kills this project's GPU processes there (ISSUE-010).
- FLUX.2-9B was stopped on 2026-10-19 at 16:33 EDT, at step 66,590, and is not continued. Its step-65,000 weights are the final FLUX.2-9B model, so both trained models end at step 65,000 (learning rate 2.32e-6). `nnmc71` GPUs 0–3 then went to the RoboTwin evaluation.

## Training

Pending.

## RoboTwin Evaluation

Pending (`plan.md`).

## Published Weights

Qwen-Image-2.1, step 65,000: `https://huggingface.co/JingwuLuo/imagewam-qwenimage2.1-robotwin` (public).
- The files are `checkpoints/weights/step_065000.pt`, `config.yaml`, `dataset_stats.json`, `LICENSE` and `NOTICE` (Qwen Research License), and the model card `README.md`.
- The layout follows the run directory, so `launch/robotwin_eval.sh` finds `dataset_stats.json` from the weight path.
- Training logs, optimizer states and text features are not published.
- Verified on 2026-10-19 at Hub commit `f47d3176`: 6 files. The weight file matches its local SHA-256, and the other files match by git blob hash.
- The local upload folder is `checkpoints/hf_release/imagewam-qwenimage2.1-robotwin/`, with the weight file hard-linked to the run copy.
