# Plan

Plan Status: approved

Approval: the user's instructions of 2026-10-08 fix the scope and resources.
- Train FLUX.2-klein-base-9B and Qwen-Image-2.1 on RoboTwin 2.0, aligned with the ImageWAM paper and repository.
- Use the released FLUX.2-4B RoboTwin model instead of retraining 4B.
- Keep the full recipe, with no reduction.
- Qwen-Image-2.1 trains on `nnmc73` with 4 B300; FLUX.2-9B on `nnmc71` with 8 H100. The 8-GPU use of `nnmc71` is a user-granted exception to the 4-per-node policy.
- The user asked for a full re-check of the training settings, prompt diversity included, without relying on earlier records. The result is `docs/robotwin_training_setup.md`.

The previous plan (G1 LGG100 training and policy server, Phases 1–24, completed) is in git history.

# Problem

## Current

- Data and reference, verified in `docs/robotwin_training_setup.md`:
  - The dataset is extracted at `dataset/robotwin2.0/robotwin2.0/` on `nnmc71`: 27,500 episodes, 6,075,103 frames, 921,032 task strings. `nonidle_ranges.json` is generated.
  - The released FLUX.2-4B model is at `checkpoints/imagewam_release/robotwin/flux2_klein_4b/`. It was trained with a global batch of 384 for 69,900 steps (5 epochs).
- Training code: the repository has RoboTwin FLUX.2 configs. Gaps:
  - The FLUX.2-9B config uses a global batch of 64 (4 × 2 per GPU on 8 GPUs), not the released 384.
  - There is no Qwen-Image-2.1 RoboTwin config or launcher. The Qwen-Image-2.1 stack (`qwenimage21_*.py`) was built and verified for G1: 16-dim, the same 288×256 compact camera layout, and a 128-slot Qwen3-VL text cache.
  - No text cache exists. Sizes: FLUX.2-9B 2.9 TB, Qwen-Image-2.1 0.97 TB. `nnmc71` has 2.7 TB free and `nnmc73` 22 TB.
  - `ImageWAM/scripts/qwenimage21/precompute_qwenimage21_text_embeds.py` encodes one prompt at a time from a prompts JSON. It was written for 4 G1 prompts.
- Evaluation: no RoboTwin simulator environment exists. The upstream installer writes into the training `.venv`.
- Environments: `.venv` (H100, `nnmc71`/`nnmc72`), `.venv-nnmc73` (B300, `nnmc73` only), `.venv-qwenimage` (Qwen3-VL text encoder, transformers 5.x).

## Problem

Neither FLUX.2-9B nor Qwen-Image-2.1 can be trained on RoboTwin with the released recipe, and no model can be evaluated on RoboTwin here.

## Goal

1. **FLUX.2-klein-base-9B RoboTwin model.** Settings match the released 4B model's `config.yaml` except for backbone-bound parts and the per-GPU split of the global batch:
   - Data: all 27,500 episodes; non-idle filter; 1% validation split; per-frame task strings with `DEFAULT_PROMPT`; Qwen3-8B text features in 128 slots.
   - Global batch 384 = 8 H100 × 4 per GPU × gradient accumulation 12. 5 epochs, about 69.9k optimizer steps.
   - bf16; AdamW (0.9, 0.95); lr 1e-4; weight decay 0.01; 5% warmup then cosine; clipping 1.0; DeepSpeed ZeRO-2.
   - Video augmentation, camera layout, horizons, normalization statistics and validation cadence as released.
2. **Qwen-Image-2.1 RoboTwin model:** the same data, split, recipe, length and global batch of 384 on 4 B300 on `nnmc73`. Only the backbone-bound components differ, as in the G1 work: editing DiT, text features (Qwen3-VL, 128 slots), VAE, and action-DiT depth.
3. **RoboTwin evaluation** of the released 4B model and both trained models under the paper protocol:
   - 50 tasks, clean and randomized, seed 42, unseen instructions;
   - 10 denoising steps, action horizon 16, replan 16, 100 trials per task.
   - The released 4B is evaluated first. Its per-task success is compared with paper Table 12 to validate the evaluation pipeline.
4. **Outputs:**
   - Weights every 5,000 steps (all kept) and the latest resumable state, under `checkpoints/robotwin_<model>/`. `nnmc73` runs write to `nnmc73` local disk; the path is recorded in PROJECT.md.
   - Offline validation metrics; measured step time and memory.
   - `docs/robotwin_training_results.md` with training and evaluation results.

Out of scope: retraining FLUX.2-4B, reduced recipes, G1.

Constraint check against this goal:
- 4 GPUs per node: `nnmc71` 8 GPUs is the user's explicit exception. Other phases use idle GPUs within policy.
- Raw data read-only: the dataset is only read. Copies on other nodes are byte-identical.
- wandb offline (`PROJECT.md`): kept.

# Structure

## Modules

- **Training configs** (`ImageWAM/configs/task/`): one FLUX.2-9B task and one Qwen-Image-2.1 task. Each states the release settings and the global-batch split explicitly.
- **Launchers** (`launch/`): one per model, binding node, GPUs, environment, data and cache paths, and run directory. Each runs in its own tmux session.
- **Text caches:** FLUX.2-9B Qwen3-8B features (`scripts/flux2/precompute_flux2_qwen3_embeds.py`, unchanged format) and Qwen-Image-2.1 Qwen3-VL features (existing script, sharded over processes).
- **Data placement:** dataset on `nnmc71` (FLUX.2-9B) and a byte-identical copy on `nnmc73` local disk (Qwen-Image-2.1).
- **Evaluation:** a separate `.venv-robotwin` environment, RoboTwin assets, and a launcher wrapping `scripts/flux2/run_eval_flux2_robotwin.sh` (FLUX.2) and an equivalent path for Qwen-Image-2.1.
- **Results:** `docs/robotwin_training_results.md`.

## Decisions

- **D1 Recipe source.** The released 4B `config.yaml` and `model.pt` (step 69,900) are the reference. They take precedence over the paper text (30k steps) and over the repository's FLUX.2-9B RoboTwin config (global batch 64).
- **D2 Global batch 384** for both models. The per-GPU micro-batch and accumulation are set per hardware so that micro × accumulation × GPUs = 384, and epochs stay at 5.
  - FLUX.2-9B: 4 × 12 × 8.
  - Qwen-Image-2.1: the largest micro-batch that fits a B300, with accumulation making 96 per GPU. Fixed in Phase 4.
- **D3 Text cache format unchanged:** padded 128 slots, one file per prompt, as the training loader reads it.
- **D4 Cache placement:**
  - Qwen-Image-2.1 cache on `nnmc73` local disk.
  - FLUX.2-9B cache (2.9 TB) on `nnmc73` disk, read by `nnmc71` over NFS (`/shared/user73/...`), if Phase 2 measures enough NFS read throughput for the training rate. Otherwise on `nnmc72` disk over NFS.
- **D5 Dataset copy** for `nnmc73`: rsync of `robotwin2.0/` plus `nonidle_ranges.json`, verified by file count and checksums of meta files.
- **D6 Evaluation environment:** `.venv-robotwin` built separately; the evaluation reads the policy from the training env through the repository's evaluation manager. Evaluation GPUs are idle `nnmc72` GPUs (at most 4).
- **D7 Logging:** wandb offline under the run directory, plus the launcher log.
- **D8 Data-loader workers:** the released 24 per GPU, lowered if CPU or RAM saturates. This does not change sampling or math.

## Responsibilities

- Configs and launchers own all training settings. Nothing is set by hand at launch.
- Cache scripts own text features. The training loader only reads them.
- The evaluation launcher owns evaluation settings (D6, Goal 3).

## State Ownership

| State | Owner | Location |
|---|---|---|
| Dataset, non-idle ranges | data placement | `dataset/robotwin2.0/robotwin2.0/` (`nnmc71`); `nnmc73` copy |
| FLUX.2-9B text cache | FLUX.2 cache script | D4 |
| Qwen-Image-2.1 text cache | Qwen-Image-2.1 cache script | `nnmc73` local disk |
| Action-DiT inits | preprocess scripts | `checkpoints/` (project) |
| Runs | trainer | `checkpoints/robotwin_flux2_klein_9b_base_imagewam/` (`nnmc71`); `nnmc73` local run dir for Qwen-Image-2.1 |
| Evaluation results | evaluation manager | `ImageWAM/evaluate_results/robotwin/...` |

# Interface

## Interfaces

- **`launch/robotwin_flux2_9b_train.sh`** (env: `GPUS`, optional `SMOKE_STEPS`). Runs `ImageWAM/scripts/train_zero2.sh 8 task=robotwin_flux2_klein_9b_base_b384_imagewam ...` with the data, cache, FLUX.2 weight and action-init overrides.
- **`launch/robotwin_qwenimage21_train.sh`** (env: `GPUS`, `MICRO_BATCH`, `GRAD_ACCUM`, optional `SMOKE_STEPS`). Runs on `nnmc73` with `.venv-nnmc73`, task `robotwin_qwenimage21_imagewam`.
- **Cache commands:**
  - FLUX.2-9B: `precompute_flux2_qwen3_embeds.py` with the FLUX.2-9B RoboTwin task overrides and `qwen_text_cache_dir` per D4.
  - Qwen-Image-2.1: `precompute_qwenimage21_text_embeds.py --prompts-json <shard> --cache-dir <dir>`, one process per GPU.
- **`launch/robotwin_eval.sh`** (env: `MODEL`, `CKPT`, `EPISODES`, `PHASES`, `GPUS`): evaluation per Goal 3.

## Inputs

Dataset, `nonidle_ranges.json`, dataset `dataset_stats.json`, base weights in `BaseModel/`, Qwen3-8B and the Qwen-Image-2.1 text encoder.

## Outputs

Text caches, action-DiT inits, run directories (Goal 4), evaluation results, and the docs.

## State Changes

Caches written once (the scripts skip existing files); runs append checkpoints; evaluation writes per-task result files.

# Flow

1. Write configs and launchers. Place the data and measure NFS read throughput (fixes D4).
2. Build both text caches (FLUX.2-9B on `nnmc71` GPUs before its training; Qwen-Image-2.1 on idle H100s with `.venv-qwenimage`, written to `nnmc73`).
3. Build the action-DiT inits (action dim 14). Smoke-test each model:
   - 30 steps; finite loss; step time, memory and loader throughput;
   - a resolved-config diff against the released 4B config.
4. Launch both full runs in tmux.
5. In parallel with training, build `.venv-robotwin` and assets. Evaluate the released 4B, first on a subset compared with Table 12, then in full.
6. Evaluate both trained models when their runs finish. Write the results.

# Code Mapping

| Task | Files |
|---|---|
| FLUX.2-9B task config | `ImageWAM/configs/task/robotwin_flux2_klein_9b_base_b384_imagewam.yaml` |
| Qwen-Image-2.1 task config | `ImageWAM/configs/task/robotwin_qwenimage21_imagewam.yaml` (data `robotwin_omnigen2`, model `imagewam_qwenimage21`) |
| Launchers | `launch/robotwin_flux2_9b_train.sh`, `launch/robotwin_qwenimage21_train.sh`, `launch/robotwin_eval.sh` |
| Qwen-Image-2.1 prompt shards | `launch/robotwin_qwen_prompts.py` (writes prompt JSON shards from `meta/tasks.jsonl` with `DEFAULT_PROMPT`) |
| Evaluation environment | `PROJECT.md` (recipe), `.venv-robotwin/` (ignored) |
| Facts and results | `docs/robotwin_training_setup.md`, `docs/robotwin_training_results.md`, `PROJECT.md` |

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Verify the training setup against first-hand sources and record it (`docs/robotwin_training_setup.md`). Download the dataset and the released model; generate the non-idle ranges.

### Files

`docs/robotwin_training_setup.md`, `plan.md`.

### Structures

None.

### Affected Modules

Docs.

### Dependencies

None.

### Observation

Config diff, dataset prompt statistics, checkpoint step, token lengths, as recorded in the doc.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Configs and launchers for both models; dataset copy on `nnmc73`; NFS read throughput from `nnmc71` to `nnmc73` disk (fixes D4).

### Files

`ImageWAM/configs/task/robotwin_flux2_klein_9b_base_b384_imagewam.yaml`, `ImageWAM/configs/task/robotwin_qwenimage21_imagewam.yaml`, `launch/robotwin_flux2_9b_train.sh`, `launch/robotwin_qwenimage21_train.sh`, `PROJECT.md`.

### Structures

The task configs and launchers.

### Affected Modules

Training configs, launchers, data placement.

### Dependencies

Phase 1.

### Observation

- Resolved-config diff of each task against the released 4B config shows only backbone-bound keys and the batch split.
- Dataset copy: file counts and meta checksums match.
- Measured NFS read MB/s on 3 MB files against the planned training read rate.

Result (2026-10-08):
- The FLUX.2-9B task differs from the released config only in backbone-bound keys (variant, action-DiT depth and heads, unused LoRA settings), the batch split 4 × 12 (released 12 × 4, both global 384), `save_every` and offline wandb.
- The Qwen-Image-2.1 task differs in the text-cache format, the batch split, `save_every` and offline wandb; its model section is the Qwen-Image-2.1 stack.
- Copy on `nnmc73` (`/home/user1/workspace/jingwu/ImageWAMFT_nnmc73/dataset/robotwin2.0/`): 27,500 parquet and 82,500 video files, 79,938,101,241 bytes, meta and non-idle checksums equal.
- NFS reads of 3.15 MB files from `nnmc71` to `nnmc73` disk: 228 MB/s with 8 threads and 962 MB/s with 32. FLUX.2-9B training needs about 110 MB/s, so D4 places its cache on `nnmc73`.
- Found during configuration: upstream DeepSpeed configs set no `gradient_clipping`, and under DeepSpeed `accelerator.clip_grad_norm_` only reports the norm. The released model was therefore most likely trained without clipping, while the paper (Table 8) states 1.0 and this project's configs enable 1.0 (commit `34bf80d`). Both launchers require `GRAD_CLIP=none|1.0`; `none` uses `scripts/ds_configs/ds_zero2_upstream_config.json`, the upstream file. The choice is the user's.

## Phase 3

Phase Status: active

Round: 3

### Goal

Build the FLUX.2-9B (Qwen3-8B) and Qwen-Image-2.1 (Qwen3-VL) text caches for all 921,032 prompts.

### Files

`launch/robotwin_qwen_prompts.py`; caches outside git.

### Structures

Prompt shards.

### Affected Modules

Text caches.

### Dependencies

Phase 2 (placement).

### Observation

- File counts equal 921,032 per cache.
- For sampled prompts, cached features equal a fresh encoding and the masks have the expected lengths.

## Phase 4

Phase Status: pending

Round: 4

### Goal

Action-DiT inits (action dim 14) and 30-step smoke tests of both models on their target hardware; fix the Qwen-Image-2.1 micro-batch (D2).

### Files

Launchers (smoke mode); inits in `checkpoints/`.

### Structures

None.

### Affected Modules

Training.

### Dependencies

Phase 3.

### Observation

Finite loss, step time, peak memory, loader throughput, projected wall clock, resolved-config diff.

## Phase 5

Phase Status: pending

Round: 5

### Goal

Launch both full runs in tmux and monitor them to completion.

### Files

Run directories; `docs/robotwin_training_results.md` (training part).

### Structures

None.

### Affected Modules

Training.

### Dependencies

Phase 4.

### Observation

Loss curves, validation metrics, step rate, no restarts or NaN; final step ≈ 69.9k.

## Phase 6

Phase Status: pending

Round: 5

### Goal

`.venv-robotwin`, RoboTwin assets and `launch/robotwin_eval.sh`. Evaluate the released FLUX.2-4B: a subset first, compared with Table 12, then the full protocol.

### Files

`launch/robotwin_eval.sh`, `PROJECT.md`, `docs/robotwin_training_results.md` (evaluation part).

### Structures

Evaluation launcher.

### Affected Modules

Evaluation.

### Dependencies

Phase 2 (data paths only).

### Observation

Per-task success against Table 12; episode time; GPU use.

## Phase 7

Phase Status: pending

Round: 6

### Goal

Evaluate the trained FLUX.2-9B and Qwen-Image-2.1 models under the same protocol and write the comparison.

### Files

`docs/robotwin_training_results.md`.

### Structures

None.

### Affected Modules

Evaluation, docs.

### Dependencies

Phases 5 and 6.

### Observation

Per-task and average success, clean and randomized, for all three models.
