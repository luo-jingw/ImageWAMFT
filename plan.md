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
   - bf16; AdamW (0.9, 0.95); lr 1e-4; weight decay 0.01; 5% warmup then cosine; gradient clipping 1.0 (D9); DeepSpeed ZeRO-1 (D10).
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
  - Qwen-Image-2.1: the largest micro-batch that fits a B300, with accumulation making 96 per GPU. Phase 4 fixed it at 96 × 1 × 4 (peak 92.8 GB of 275 GB).
- **D3 Text cache format unchanged:** padded 128 slots, one file per prompt, as the training loader reads it.
- **D4 Cache placement:**
  - Qwen-Image-2.1 cache on `nnmc73` local disk.
  - FLUX.2-9B cache (2.9 TB) on `nnmc73` disk, read by `nnmc71` over NFS (`/shared/user73/...`), if Phase 2 measures enough NFS read throughput for the training rate. Otherwise on `nnmc72` disk over NFS.
- **D5 Dataset copy** for `nnmc73`: rsync of `robotwin2.0/` plus `nonidle_ranges.json`, verified by file count and checksums of meta files.
- **D6 Evaluation environment:** `.venv-robotwin` built separately; the evaluation reads the policy from the training env through the repository's evaluation manager. Evaluation GPUs are idle `nnmc72` GPUs (at most 4).
- **D7 Logging:** wandb offline under the run directory, plus the launcher log.
- **D8 Data-loader workers:** the released 24 per GPU, lowered if CPU or RAM saturates. This does not change sampling or math.
- **D10 FLUX.2-9B uses DeepSpeed ZeRO-1** (Phase 4 measurement, 2026-10-08):
  - ZeRO-1 is the upstream launcher's default (`ZERO_STAGE:-1`), so the released 4B was trained with it.
  - ZeRO-2 reduce-scatters the gradients at each of the 12 micro-steps. ZeRO-1 reduces them once per optimizer step.
  - On `nnmc71`, NVLink joins only GPU pairs, and GPUs 0–3 reach 4–7 across sockets.
  - The optimizer math is the same in both stages: AdamW on fp32 partitions, clipping on the global norm.
  - `launch/robotwin_flux2_9b_train.sh ZERO=1` uses `scripts/ds_configs/ds_zero1_config.json`, which sets `gradient_clipping` 1.0.
  - Qwen-Image-2.1 stays on ZeRO-2. It accumulates once per step, so ZeRO-1 would save no communication.
- **D9 Gradient clipping 1.0** (user decision, 2026-10-08): follow the paper (Table 8) with `GRAD_CLIP=1.0` (`scripts/ds_configs/ds_zero2_config.json`). The released 4B was most likely trained without effective clipping (upstream DeepSpeed configs set none), so this is the one recipe difference to the released model, stated with every comparison.

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

- **`launch/robotwin_flux2_9b_train.sh`** (env: `GPUS`, `GRAD_CLIP`, `ZERO` (1 or 2), optional `SMOKE_STEPS`, `MODE=cache`). Runs `ImageWAM/scripts/train_zero{1,2}.sh 8 task=robotwin_flux2_klein_9b_base_b384_imagewam ...` with the data, cache, FLUX.2 weight and action-init overrides.
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
| Evaluation environment | `node_env/build_venv_robotwin.sh`, `PROJECT.md` (recipe), `.venv-robotwin/` and RoboTwin assets (ignored) |
| Qwen-Image-2.1 evaluation text features | `ImageWAM/scripts/qwenimage21/qwen_text_server.py` (`.venv-qwenimage`, Qwen3-VL loaded once); `ImageWAM/experiments/robotwin/imagewam_policy/deploy_policy.py` (`qwen_text_server` client for the `qwenimage21` stack); `ImageWAM/experiments/robotwin/eval_robotwin_single.py` and `ImageWAM/configs/sim_robotwin.yaml` (`EVALUATION.qwen_text_server`) |
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
- Found during configuration: upstream DeepSpeed configs set no `gradient_clipping`, and under DeepSpeed `accelerator.clip_grad_norm_` only reports the norm. The released model was therefore most likely trained without clipping, while the paper (Table 8) states 1.0 and this project's configs enable 1.0 (commit `34bf80d`). Both launchers require `GRAD_CLIP=none|1.0`; `none` uses `scripts/ds_configs/ds_zero2_upstream_config.json`, the upstream file. The user chose 1.0 (D9).

## Phase 3

Phase Status: completed

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

Result (2026-10-08, details in `docs/robotwin_training_setup.md` → Storage):
- FLUX.2-9B cache (`nnmc71` 8 GPUs, written to `nnmc73`):
  - 921,032 files, 0 missing, 0 extra.
  - 24 sampled entries against a batch-of-one fresh encoding: identical masks, relative difference 0.07–0.09% (bf16 batch-shape noise).
- Qwen-Image-2.1 cache: 921,032 files, 0 missing, 0 extra.
  - The `nnmc72` run was killed every 30 minutes (ISSUE-010); 315,231 prompts came from `nnmc72` H100s and the rest from `nnmc73` B300s.
  - B300-written entries equal a fresh encoding bit for bit. H100-written entries differ by 2.1–2.4% (cross-hardware).

## Phase 4

Phase Status: completed

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

Result (2026-10-08):
- Action-DiT inits: `ImageWAM/checkpoints/action_dit_{flux2_9b,qwenimage21}_robotwin_init.pt`. They equal the G1 inits tensor for tensor; the layers that depend on the action dimension are created at model build.
- Smoke tests, 30 steps each, `GRAD_CLIP=1.0`. Logs are in `checkpoints/robotwin_logs/` and on `nnmc73` in `checkpoints/qwen_smoke_mb96.log`.

| Model | Hardware | Micro × accumulation × GPUs | ZeRO | Loss at steps 10 / 20 / 30 | s per optimizer step | Peak GPU memory | Full run (69,900 steps) |
|---|---|---|---|---|---|---|---|
| Qwen-Image-2.1 | `nnmc73` 4 × B300 | 96 × 1 × 4 | 2 | 1.72 / 0.98 / 0.91 | 6.4–6.5 | 92.8 GB | about 5.2 days |
| FLUX.2-9B | `nnmc71` 8 × H100 NVL | 4 × 12 × 8 | 2 | 2.04 / 1.06 / 0.69 | 24.5–25.1 | 67.3 GB | about 20 days |
| FLUX.2-9B | `nnmc71` 8 × H100 NVL | 4 × 12 × 8 | 1 | 2.13 / 1.14 / 0.76 | 9.7–9.9 | 76.6 GB | about 7.9 days |

- The data loader is not a bottleneck: data wait was 0.001–0.007 s per micro-step.
- Validation at step 30 and the final save succeeded in every run.
- Resolved-config diff of the Qwen-Image-2.1 run against the released config: only module names, paths, the text-cache format, backbone-bound model keys, the batch split (96 × 1 against 12 × 4, both 384), project keys, wandb and the smoke-only keys differ.
- The full runs use Qwen-Image-2.1 96 × 1 × 4 with ZeRO-2 (D2) and FLUX.2-9B with ZeRO-1 (D10).
- 9B weight files are 54 GB each (`lora_merged` format: 18.76 B stored bf16 values, with the video expert under two names).

## Phase 5

Phase Status: active

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

Phase Status: blocked

Round: 5

### Goal

`.venv-robotwin`, RoboTwin assets and `launch/robotwin_eval.sh`. Evaluate the released FLUX.2-4B: a subset first, compared with Table 12, then the full protocol.

### Files

`node_env/build_venv_robotwin.sh`, `launch/robotwin_eval.sh`, the Qwen-Image-2.1 evaluation text-feature files (Code Mapping), the RoboTwin policy symlink, `PROJECT.md`, `docs/robotwin_training_results.md` (evaluation part).

### Structures

Evaluation launcher.

### Affected Modules

Evaluation.

### Dependencies

Phase 2 (data paths only).

### Observation

Per-task success against Table 12; episode time; GPU use.

Progress (2026-10-08):
- `.venv-robotwin`, the assets and `launch/robotwin_eval.sh` are done. A released FLUX.2-4B check (`beat_block_hammer`, clean, 2 episodes) succeeded on both episodes.
- The Qwen-Image-2.1 evaluation text path exists (Code Mapping); its client was tested against the server on CPU.
- Blocked: since 2026-10-08, GPU processes on `nnmc72`, the D6 evaluation host, are killed every 30 minutes (ISSUE-010). The full evaluation needs about 9 h. It can resume once `nnmc72` is usable or the user names another GPU host.

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
