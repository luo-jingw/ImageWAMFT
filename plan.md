# Plan

Plan Status: proposed

The previous plan (RoboTwin 2.0 training of FLUX.2-9B and Qwen-Image-2.1, Phases 1–7) is in git history. Its training phase ends with Phase 1 below; its evaluation phases are replaced by this plan.

# Problem

## Current

- Models to evaluate:

  | Model | Weights | Steps | Text features at evaluation |
  |---|---|---|---|
  | Released FLUX.2-4B | `checkpoints/imagewam_release/robotwin/flux2_klein_4b/model.pt` | 69,900 | Qwen3-4B, in the worker |
  | FLUX.2-9B | `checkpoints/robotwin_flux2_klein_9b_base_b384_imagewam/checkpoints/weights/step_069900.pt` (54 GB, LoRA-merged format) | 69,900 | Qwen3-8B, in the worker |
  | Qwen-Image-2.1 | `checkpoints/robotwin_qwenimage21_imagewam/checkpoints/weights/step_065000.pt` (15.9 GB) | 65,000 of 69,900 | Qwen3-VL, from `qwen_text_server.py` |

  - FLUX.2-9B runs on `nnmc71` GPUs 0–3: step 66,190 at 2026-10-19 14:20 EDT, 19.7 s per step, end about 2026-10-20 10:40 EDT.
  - Qwen-Image-2.1 stopped at step 67,110 (ISSUE-013). Its step-65,000 weights are the final model. They were copied from `nnmc73` to `nnmc71` with `config.yaml`, `dataset_stats.json`, `train.log`, `wandb/` and `eval/`; the weight file's SHA-256 is equal on both sides.
- Evaluation pipeline (`PROJECT.md` → RoboTwin 2.0 evaluation environment; `docs/robotwin_training_setup.md` → Evaluation Protocol):
  - `.venv-robotwin`, the RoboTwin assets and `launch/robotwin_eval.sh` exist on `nnmc71`.
  - It ran once: the released FLUX.2-4B on `beat_block_hammer`, clean, 2 episodes, both successful. A successful 105-action episode took about 10 s including the expert check.
  - The FLUX.2-9B and Qwen-Image-2.1 paths have not run in the simulator. The Qwen-Image-2.1 text-feature client was tested against the server on CPU only.
- Manager (`ImageWAM/experiments/robotwin/run_robotwin_manager.py`):
  - One worker process per (task, phase). A failed worker terminates all running workers and ends the run; there is no resume.
  - The clean and randomized phases of a task share the task directory and its `episode_end_states.jsonl`. A record's phase is in its video file name (`randomized-true|false`).
  - Results go to `ImageWAM/evaluate_results/robotwin/<checkpoint file stem>/<OUTPUT_TAG>/`.
- GPUs: only `nnmc71` is usable. `nnmc72` has the eviction script (ISSUE-010), and `nnmc73`–`nnmc76` the kill order (ISSUE-013). On `nnmc71`, GPUs 4–7 hold another user's job, and GPUs 0–3 are free once the FLUX.2-9B run ends.
- Cost reference:
  - Step limits per task are 400–1,700 actions (mean 620; `third_party/RoboTwin/task_config/_eval_step_limit.yml`). A failed episode runs to its limit.
  - Action-chunk latency on one H100 (G1 models, batch 1, 10 denoising steps): FLUX.2-4B 170 ms, FLUX.2-9B 215 ms, Qwen-Image-2.1 258 ms (`docs/g1_training_results.md`).
  - Estimate, to be measured in Phase 5: 6–12 h per model on 4 H100, so about 1–1.5 days for the three models.

## Problem

No model trained here has a RoboTwin success rate. The evaluation pipeline has not been checked against the paper. The full protocol is 30,000 episodes on 4 GPUs, and a single worker failure currently discards the whole run.

## Goal

1. **Pipeline check:** the released FLUX.2-4B under the paper protocol. Its per-task and mean success are compared with paper Table 12 (93.20% clean, 93.56% randomized).
2. **Trained models:** FLUX.2-9B at step 69,900 and Qwen-Image-2.1 at step 65,000 under the same protocol.
3. **Results** in `docs/robotwin_training_results.md`:
   - per-task clean and randomized success of the three models, and their means with standard errors;
   - paired differences between models on identical seeds;
   - episode time and GPU use;
   - the training section of both runs.

Protocol, fixed for all three models:
- 50 tasks; clean (`demo_clean`) and randomized (`demo_randomized`); 100 episodes per task and setting;
- seed 42: the expert-checked seeds from 4,300,000; unseen instructions;
- 10 denoising steps, action horizon 16, replan 16, `compact_288x256`, text CFG scale 1.0.

Out of scope: retraining or continuing any run, real-robot evaluation, and ISSUE-012.

Constraint check against this goal:
- GPU policy: 4 GPUs on `nnmc71` are within it. More GPUs need a grant from the user (E2).
- Qwen-Image-2.1 has 4,900 fewer steps. At step 65,000 the learning rate is 2.32e-6 (peak 1e-4, floor 1e-6). The step difference is stated with every comparison, as is gradient clipping 1.0 against the released model (previous plan D9).
- Text slots: the FLUX.2 workers encode instructions into 512 slots, while training used 128. The extra slots are masked padding, and the proprio token is placed right after the valid text tokens (`ImageWAM/src/imagewam/models/backbones/imagewam.py`, `pack_proprio_after_text`). The conditioning is therefore the same apart from numerics. Qwen-Image-2.1 is evaluated with 128 slots.
- The Qwen-Image-2.1 evaluation features are encoded on an H100 (`.venv-qwenimage`). 66% of its training features were encoded on B300; the two differ by 2.2–2.6% relative L2 (`PROJECT.md`).
- wandb is not used by evaluation. Raw data is only read.

# Structure

## Modules

- **Launcher** `launch/robotwin_eval.sh`: binds a model name to its checkpoint, statistics, base weights and text path, and sets the protocol.
- **Manager** `run_robotwin_manager.py`: schedules (task, phase) workers on GPUs, collects their success rates, and writes the summary. New: resume.
- **Worker:** `eval_robotwin_single.py` → RoboTwin `script/eval_policy.py` → `imagewam_policy/deploy_policy.py`. Unchanged.
- **Qwen-Image-2.1 text server** `scripts/qwenimage21/qwen_text_server.py`. Unchanged.
- **Paper reference** `launch/robotwin_table12_flux2_4b.csv`: the FLUX.2-4B columns of paper Table 12.
- **Comparison** `launch/robotwin_eval_compare.py`:
  - reads result directories and the reference;
  - checks that seeds, instructions and episode counts match across models;
  - writes the result tables.
- **Results** `docs/robotwin_training_results.md`.

## Decisions

- **E1 Checkpoints:** the three files in Current. Statistics come from each run directory's `dataset_stats.json`.
- **E2 Host:** `nnmc71` GPUs 0–3 once the FLUX.2-9B run has ended. GPUs 4–7, if they become free, are used only with the user's grant.
- **E3 Order:** released 4B, then FLUX.2-9B, then Qwen-Image-2.1, each on all 4 GPUs.
  - The 4B run is the gate. Its finished tasks are compared with Table 12 while it runs.
  - A systematic deviation stops the queue and is recorded in `issues.md`. Example: most tasks below their Table 12 value by more than 3 binomial standard errors.
- **E4 Resume:** `MULTIRUN.resume=true`, rerun with the same `OUTPUT_TAG`.
  - A task with result files for all selected phases is read from those files and not rerun.
  - Any other existing task directory is renamed to `<task>.incomplete_<k>`, and the task reruns both phases from its first episode.
  - The rerun gives the same seeds and instructions: each (task, phase) is a fresh process with seed 42, and the expert check that selects the seeds does not depend on the policy.
- **E5 Output names:** an explicit `OUTPUT_TAG` per model: `flux2_4b_release_full`, `flux2_9b_step069900_full`, `qwenimage21_step065000_full`.
- **E6 Workers per GPU (`MAX_TASKS_PER_GPU`):** fixed per model in Phase 5 from measured values:
  - GPU memory per worker, within 94 GB; GPU 0 also holds the Qwen-Image-2.1 text server;
  - host memory while a worker loads. Each FLUX.2-9B worker reads the 54 GB weight file into host memory.
  - If simultaneous loads exceed free host memory, the manager starts workers at a fixed interval (`MULTIRUN.launch_interval_sec`).
- **E7 Paired comparison:**
  - Episodes are paired by (task, phase, seed).
  - The comparison reports each model's mean with its binomial standard error over the 5,000 episodes per setting, and the paired difference between models with its standard error.
  - If instructions differ between models for the same seed, that is reported. The Python `random` state that draws them is not reseeded per episode.
- **E8 Matched-step FLUX.2-9B (step 65,000):** not planned. It costs one more full evaluation and isolates the effect of the last 4,900 steps.

## Responsibilities

- The launcher owns the protocol settings; nothing is set by hand at launch apart from `MODEL`, `CKPT`, `GPUS`, `OUTPUT_TAG` and `MAX_TASKS_PER_GPU`.
- The manager owns scheduling and resume.
- The comparison script owns every number in the results tables.

## State Ownership

| State | Owner | Location |
|---|---|---|
| Evaluation results | manager and workers | `ImageWAM/evaluate_results/robotwin/<checkpoint stem>/<OUTPUT_TAG>/` |
| Paper reference values | reference file | `launch/robotwin_table12_flux2_4b.csv` |
| Result tables | comparison script | its output, copied into `docs/robotwin_training_results.md` |
| Qwen-Image-2.1 evaluation text features | text server | `checkpoints/robotwin_eval_text_cache/qwenimage21/` |

# Interface

## Interfaces

- **`launch/robotwin_eval.sh`** (env `MODEL`, `CKPT`, `GPUS`, `OUTPUT_TAG`, `MAX_TASKS_PER_GPU`, `EPISODES`, `PHASES`, `TASKS`; Hydra overrides as arguments). Resume is the override `MULTIRUN.resume=true`.
- **Manager:** `MULTIRUN.resume: bool` (default `false`, `configs/sim_robotwin.yaml`).
  - When true, each task is either `done` (its rates are read) or `pending`. A pending task's existing directory is renamed first.
  - With E6, `MULTIRUN.launch_interval_sec: float` (default 0).
- **`launch/robotwin_eval_compare.py --result <name>=<run dir> ... --reference launch/robotwin_table12_flux2_4b.csv --output <file.md>`:**
  - Input: `summary.json` and `<task>/episode_end_states.jsonl` of each run directory.
  - Typed records: `EpisodeRecord` (task, phase, seed, success, instruction, actions taken) and `TaskResult` (task, phase, success rate, episodes).
  - Output:
    - a per-task table (reference, then each model, clean and randomized) with means and standard errors;
    - a paired-difference table;
    - printed checks: episodes per (task, phase), seed-list and instruction equality across models.
- **`launch/robotwin_table12_flux2_4b.csv`:** columns `task,clean,random`, success in percent, using RoboTwin task names.

## Inputs

The three weight files with their `dataset_stats.json`, the base weights in `BaseModel/`, `.venv-robotwin`, `.venv-qwenimage`, the RoboTwin assets, and paper Table 12 (`papers/ImageWAM_2606.19531.txt`).

## Outputs

Per model: an evaluation directory with `summary.json`, `summary.csv`, `manager.log`, per-task logs, records and videos. Overall: the comparison tables and the docs.

## State Changes

- Evaluation directories are append-only within a run.
- Resume renames incomplete task directories and deletes nothing.
- The text server adds one feature file per new instruction.

# Flow

1. Before GPUs are free (CPU only):
   - write the Table 12 reference and the comparison script;
   - add resume to the manager;
   - compose each model's evaluation config and diff its model and processor sections against the training config (4B: the released `config.yaml`).
2. After the FLUX.2-9B run ends (about 2026-10-20 10:40 EDT):
   - check the final weights;
   - write the training section.
3. Smoke on GPUs 0–3: each model on `beat_block_hammer`, clean and randomized, 5 episodes each.
   - Measured: load time, GPU and host memory per worker, episode time, checkpoint key mismatches.
   - Then the resume check, and fix `MAX_TASKS_PER_GPU` per model.
4. Released 4B, full protocol; compare with Table 12 (gate, E3).
5. FLUX.2-9B, full protocol; then Qwen-Image-2.1, full protocol.
6. Run the comparison script; write the evaluation section.

# Code Mapping

| Task | Files |
|---|---|
| Paper reference | `launch/robotwin_table12_flux2_4b.csv` |
| Comparison | `launch/robotwin_eval_compare.py` |
| Resume (and launch interval, if E6 needs it) | `ImageWAM/experiments/robotwin/run_robotwin_manager.py`, `ImageWAM/configs/sim_robotwin.yaml` |
| Workers per GPU, resume usage | `launch/robotwin_eval.sh` |
| Evaluation config check, protocol facts | `docs/robotwin_training_setup.md` (Evaluation Protocol) |
| Training and evaluation results | `docs/robotwin_training_results.md` |
| Run state, open problems | `PROJECT.md`, `issues.md` |

# Implementation

## Phase 1

Phase Status: pending

Round: 1

### Goal

Close the training runs: the FLUX.2-9B run reaches step 69,900, and both runs' training results are recorded.

### Files

`docs/robotwin_training_results.md` (Runs, Training), `issues.md` (ISSUE-013 resolution), `PROJECT.md` (run locations).

### Structures

None.

### Affected Modules

Results.

### Dependencies

None. It waits for the run to end.

### Observation

- `step_069900.pt` exists, with `step` 69,900 and only finite values.
- Loss and validation curves of both runs, with duplicate steps removed across the two FLUX.2-9B wandb runs (the newer run for steps 40,001–42,150).
- Final validation metrics of both runs, and wall-clock time.

## Phase 2

Phase Status: pending

Round: 1

### Goal

The Table 12 reference file and the comparison script.

### Files

`launch/robotwin_table12_flux2_4b.csv`, `launch/robotwin_eval_compare.py`.

### Structures

`EpisodeRecord`, `TaskResult`.

### Affected Modules

Paper reference, comparison.

### Dependencies

None.

### Observation

- The reference has 50 tasks whose names match `_eval_step_limit.yml`, with means 93.20 and 93.56.
- On the existing check run (`ImageWAM/evaluate_results/robotwin/model/smoke_2ep/`), the script reports 2 clean episodes, seeds 4,300,000 and 4,300,001, and success 1.0.

## Phase 3

Phase Status: pending

Round: 1

### Goal

Manager resume (E4).

### Files

`ImageWAM/experiments/robotwin/run_robotwin_manager.py`, `ImageWAM/configs/sim_robotwin.yaml`.

### Structures

`MULTIRUN.resume`.

### Affected Modules

Manager.

### Dependencies

None.

### Observation

In Phase 5, on a 2-task run:
- a resumed run reads both tasks and launches none;
- after one result file is removed, that task's directory is renamed and the task reruns with the same seeds and instructions as before.

## Phase 4

Phase Status: pending

Round: 1

### Goal

Check each model's evaluation config against its training config.

### Files

`docs/robotwin_training_setup.md` (Evaluation Protocol).

### Structures

None.

### Affected Modules

Launcher (read only).

### Dependencies

None.

### Observation

- For each model, the model and processor sections of the composed worker config are diffed against the training `config.yaml`. 4B uses the released config; the FLUX.2 workers compose the `_clean_` task configs.
- Every difference is listed with its reason. Expected differences: paths, `load_text_encoder`, text slots (512 against 128), and the action-expert init path.
- An unexplained difference in a key that changes the network or the preprocessing stops the plan.

## Phase 5

Phase Status: pending

Round: 2

### Goal

GPU smoke of the three models; resume check (Phase 3); `MAX_TASKS_PER_GPU` per model (E6).

### Files

`launch/robotwin_eval.sh` (per-model `DEFAULT_TASKS_PER_GPU`); the manager launch interval only if E6 requires it.

### Structures

None.

### Affected Modules

Launcher, manager.

### Dependencies

Phases 1 (free GPUs and the final FLUX.2-9B weights), 3 and 4.

### Observation

Per model:
- checkpoint missing and unexpected keys (expected 0 and 0);
- load time; peak GPU memory per worker; peak host memory while loading;
- episode time for successful and failed episodes;
- success on the 10 smoke episodes.

Also the projected wall-clock time of each full run, and the resume behavior of Phase 3.

## Phase 6

Phase Status: pending

Round: 3

### Goal

Released FLUX.2-4B under the full protocol, compared with Table 12 (gate, E3).

### Files

`docs/robotwin_training_results.md` (RoboTwin Evaluation).

### Structures

None.

### Affected Modules

Evaluation.

### Dependencies

Phases 2 and 5.

### Observation

- Per-task and mean success, clean and randomized, against Table 12, with the per-task difference in binomial standard errors.
- Wall-clock time; failed or resumed workers.

## Phase 7

Phase Status: pending

Round: 4

### Goal

FLUX.2-9B (step 69,900), then Qwen-Image-2.1 (step 65,000), under the full protocol.

### Files

None. The outputs are evaluation directories.

### Structures

None.

### Affected Modules

Evaluation.

### Dependencies

Phase 6 (gate passed).

### Observation

- 100 episodes per (task, phase) for both models;
- seed lists equal to the 4B run;
- wall-clock time; failed or resumed workers.

## Phase 8

Phase Status: pending

Round: 5

### Goal

Comparison of the three models and the evaluation write-up.

### Files

`docs/robotwin_training_results.md` (RoboTwin Evaluation).

### Structures

None.

### Affected Modules

Results.

### Dependencies

Phases 2 and 7.

### Observation

- Per-task and mean success of the three models, with standard errors.
- Paired differences: FLUX.2-9B − 4B, Qwen-Image-2.1 − 4B, and Qwen-Image-2.1 − FLUX.2-9B.
- The step and clipping differences stated beside each comparison.
