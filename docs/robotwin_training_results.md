# RoboTwin 2.0 Training Results

Training runs of the FLUX.2-9B and Qwen-Image-2.1 models. Their RoboTwin evaluation is in `docs/robotwin_evaluation_results.md`. The setup and its verification against the released FLUX.2-4B model are in `docs/robotwin_training_setup.md`. The smoke-test measurements are in the training plan, Phase 4 (`git show 8f02eb3:plan.md`).

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
| Measured speed | 9.9 s per step on 8 GPUs (steps 620–1,120, one evaluation included); 19.8 s per step on 4 GPUs (steps 42,500–48,460, evaluations included) | 4.2 s per step (steps 30–130) |
| Peak GPU memory | 76.6 GB on 8 GPUs; 88.2 GB used per GPU on 4 GPUs (`nvidia-smi`) | 55.3 GB |
| Weights | every 5,000 steps and at the end, 54 GB each | every 5,000 steps and at the end, 15.9 GB each |
| Final weights | step 65,000 | step 65,000 |
| Wall clock, first logged step to step 65,000 | 254.2 h (2026-10-08 17:37 – 10-19 07:47 EDT), including steps 40,001–42,150 trained twice | 206.2 h (2026-10-08 20:51 – 10-17 11:05 EDT): 4.07 s per step up to step 45,000 (50.9 h), 8.38 s over steps 45,001–50,000 and 33.3–36.3 s from step 50,001 on, while the `nnmc73` GPUs were shared (ISSUE-011) |

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

- **Source:** the wandb offline records `<run dir>/wandb/offline-run-*/run-*.wandb`, read with `read_history` of `ImageWAM/scripts/wandb_offline_to_tensorboard.py`. Training scalars are logged every 10 steps.
- **FLUX.2-9B history:** the first run (`231w4w9t`) up to step 40,000 and the resumed run (`tqq75pvs`) from step 40,001. Steps 40,001–42,150, trained twice, come from the resumed run.
- **Validation:** every 500 steps, one random sample per rank from the 1% validation split (`val_set_proportion` 0.01): 8 samples per evaluation, 4 for FLUX.2-9B after its move to 4 GPUs. Metrics as in `docs/g1_training_results.md` → Training-Time Validation (`psnr_rg`: predicted frame t + 16 against the true frame).
- The logged FLUX.2-9B `grad_norm` doubles at step 40,000 (ISSUE-012), so it is not tabulated.

Training loss, mean over each 5,000-step bin (500 logged steps), and the learning rate at the end of the bin:

| Steps | FLUX.2-9B total | action | video | Qwen-Image-2.1 total | action | video | Learning rate |
|---|---|---|---|---|---|---|---|
| 1–5,000 | 0.2057 | 0.0908 | 0.1149 | 0.1505 | 0.0995 | 0.0511 | 9.99e-5 |
| 5,001–10,000 | 0.1068 | 0.0106 | 0.0962 | 0.0522 | 0.0110 | 0.0412 | 9.77e-5 |
| 10,001–15,000 | 0.0993 | 0.0072 | 0.0922 | 0.0462 | 0.0076 | 0.0386 | 9.28e-5 |
| 15,001–20,000 | 0.0949 | 0.0060 | 0.0889 | 0.0431 | 0.0059 | 0.0372 | 8.57e-5 |
| 20,001–25,000 | 0.0904 | 0.0048 | 0.0856 | 0.0410 | 0.0050 | 0.0360 | 7.65e-5 |
| 25,001–30,000 | 0.0887 | 0.0039 | 0.0848 | 0.0393 | 0.0043 | 0.0350 | 6.59e-5 |
| 30,001–35,000 | 0.0855 | 0.0035 | 0.0820 | 0.0376 | 0.0037 | 0.0340 | 5.45e-5 |
| 35,001–40,000 | 0.0839 | 0.0031 | 0.0809 | 0.0364 | 0.0032 | 0.0332 | 4.28e-5 |
| 40,001–45,000 | 0.0799 | 0.0027 | 0.0772 | 0.0350 | 0.0027 | 0.0323 | 3.16e-5 |
| 45,001–50,000 | 0.0810 | 0.0022 | 0.0788 | 0.0341 | 0.0024 | 0.0317 | 2.14e-5 |
| 50,001–55,000 | 0.0792 | 0.0020 | 0.0772 | 0.0333 | 0.0021 | 0.0312 | 1.28e-5 |
| 55,001–60,000 | 0.0790 | 0.0016 | 0.0773 | 0.0324 | 0.0018 | 0.0306 | 6.33e-6 |
| 60,001–65,000 | 0.0757 | 0.0016 | 0.0741 | 0.0322 | 0.0016 | 0.0305 | 2.32e-6 |

- From step 5,001 on, the action losses of the two backbones differ by at most 0.00044 per bin; FLUX.2-9B is lower in 11 of the 12 bins. Over steps 60,001–65,000 they are 0.00157 and 0.00164. The video loss is not comparable across backbones: Qwen-Image-2.1 has a different latent space (`docs/g1_training_setup.md` → Video Loss Across Backbones).
- After step 65,000, up to their stops, the losses stay at the last bin's level: FLUX.2-9B 0.0764 / 0.0016 / 0.0748 (steps 65,001–66,580), Qwen-Image-2.1 0.0318 / 0.0015 / 0.0303 (steps 65,001–67,100).

Validation, mean over the 10 evaluations of each 5,000-step bin:

| Steps | FLUX.2-9B `action_l1` | FLUX.2-9B PSNR (dB) | Qwen-Image-2.1 `action_l1` | Qwen-Image-2.1 PSNR (dB) |
|---|---|---|---|---|
| 1–5,000 | 0.0546 | 22.64 | 0.0569 | 22.78 |
| 5,001–10,000 | 0.0274 | 24.90 | 0.0273 | 24.44 |
| 10,001–15,000 | 0.0208 | 25.58 | 0.0185 | 25.41 |
| 15,001–20,000 | 0.0168 | 25.30 | 0.0168 | 25.17 |
| 20,001–25,000 | 0.0170 | 26.09 | 0.0181 | 25.70 |
| 25,001–30,000 | 0.0149 | 27.09 | 0.0148 | 26.61 |
| 30,001–35,000 | 0.0157 | 26.06 | 0.0154 | 25.77 |
| 35,001–40,000 | 0.0139 | 27.87 | 0.0126 | 28.23 |
| 40,001–45,000 | 0.0121 | 28.19 | 0.0129 | 27.59 |
| 45,001–50,000 | 0.0107 | 26.77 | 0.0093 | 27.39 |
| 50,001–55,000 | 0.0094 | 28.50 | 0.0102 | 27.90 |
| 55,001–60,000 | 0.0113 | 28.73 | 0.0115 | 28.67 |
| 60,001–65,000 | 0.0083 | 28.72 | 0.0078 | 29.45 |

Validation at the final weights: mean ± standard error of the 20 evaluations at steps 55,500–65,000, with the step-65,000 evaluation in parentheses.

| Metric | FLUX.2-9B | Qwen-Image-2.1 |
|---|---|---|
| `val_loss` | 0.109 ± 0.010 (0.152) | 0.045 ± 0.004 (0.040) |
| `action_l1` | 0.0098 ± 0.0011 (0.0043) | 0.0096 ± 0.0009 (0.0058) |
| `action_l2` | 0.0012 ± 0.0003 (0.0001) | 0.0012 ± 0.0003 (0.0002) |
| PSNR, predicted vs. true frame | 28.73 ± 0.83 (32.39) | 29.06 ± 0.52 (30.36) |
| SSIM, predicted vs. true frame | 0.915 ± 0.007 (0.936) | 0.922 ± 0.005 (0.929) |
| PSNR, VAE reconstruction | 43.45 ± 0.20 | 43.04 ± 0.13 |
| Training loss, steps 64,001–65,000 (total / action / video) | 0.0737 / 0.00145 / 0.0723 | 0.0318 / 0.00164 / 0.0302 |

- The two models agree on every action and predicted-frame metric within one standard error. With 4–8 random samples per evaluation, single evaluations, including the one at step 65,000, are noise-dominated.

## Published Weights

Qwen-Image-2.1, step 65,000: `https://huggingface.co/JingwuLuo/imagewam-qwenimage2.1-robotwin` (public).
- The files are `checkpoints/weights/step_065000.pt`, `config.yaml`, `dataset_stats.json`, `LICENSE` and `NOTICE` (Qwen Research License), and the model card `README.md`.
- The layout follows the run directory, so `launch/robotwin_eval.sh` finds `dataset_stats.json` from the weight path.
- Training logs, optimizer states and text features are not published.
- Verified on 2026-10-19 at Hub commit `f47d3176`: 6 files. The weight file matches its local SHA-256, and the other files match by git blob hash.
- The local upload folder is `checkpoints/hf_release/imagewam-qwenimage2.1-robotwin/`, with the weight file hard-linked to the run copy.
