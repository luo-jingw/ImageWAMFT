# Plan

Plan Status: approved

Approval: the user approved this plan on 2026-10-19. The evaluation runs on one node with 4 GPUs (E2). The FLUX.2-9B run was stopped the same day, so both trained models are evaluated at step 65,000 (E1), and Qwen-Image-2.1 is evaluated first (E3).

The previous plan (RoboTwin 2.0 training of FLUX.2-9B and Qwen-Image-2.1, Phases 1–7) is in git history. Its training phase ends with Phase 6 below; its evaluation phases are replaced by this plan.

# Problem

## Current

- Models to evaluate:

  | Model | Weights | Steps | Text features at evaluation |
  |---|---|---|---|
  | Released FLUX.2-4B | `checkpoints/imagewam_release/robotwin/flux2_klein_4b/model.pt` | 69,900 | Qwen3-4B, in the worker |
  | FLUX.2-9B | `checkpoints/robotwin_flux2_klein_9b_base_b384_imagewam/checkpoints/weights/step_065000.pt` (54 GB, LoRA-merged format) | 65,000 of 69,900 | Qwen3-8B, in the worker |
  | Qwen-Image-2.1 | `checkpoints/robotwin_qwenimage21_imagewam/checkpoints/weights/step_065000.pt` (15.9 GB) | 65,000 of 69,900 | Qwen3-VL, from `qwen_text_server.py` |

  - The FLUX.2-9B run was stopped on 2026-10-19 at 16:33 EDT, at step 66,590; its last saved weights are those of step 65,000.
  - Qwen-Image-2.1 stopped at step 67,110 (ISSUE-013). Its step-65,000 weights are the final model. They were copied from `nnmc73` to `nnmc71` with `config.yaml`, `dataset_stats.json`, `train.log`, `wandb/` and `eval/`; the weight file's SHA-256 is equal on both sides.
- Evaluation pipeline (`PROJECT.md` → RoboTwin 2.0 evaluation environment; `docs/robotwin_training_setup.md` → Evaluation Protocol):
  - `.venv-robotwin`, the RoboTwin assets and `launch/robotwin_eval.sh` exist on `nnmc71`.
  - All three models ran `beat_block_hammer`, clean and randomized, 5 episodes each, on one L40S each (2026-10-19). All 30 episodes succeeded, and every checkpoint loaded with 0 missing and 0 unexpected keys.
- Manager (`ImageWAM/experiments/robotwin/run_robotwin_manager.py`):
  - One worker process per (task, phase). A failed worker terminates all running workers and ends the run.
  - The clean and randomized phases of a task share the task directory and its `episode_end_states.jsonl`. A record's phase is in its video file name (`randomized-true|false`).
  - Results go to `ImageWAM/evaluate_results/robotwin/<checkpoint file stem>/<OUTPUT_TAG>/`.
- GPUs:
  - `nnmc71` GPUs 0–3 (H100 NVL, 94 GB) are free since the FLUX.2-9B run stopped. GPUs 4–7 are not used by this project.
  - `nnmc72` has the eviction script (ISSUE-010), and `nnmc73`–`nnmc76` the kill order (ISSUE-013).
- Memory per worker, measured on L40S (46 GB):
  - FLUX.2-4B up to 24.5 GB;
  - FLUX.2-9B up to 43.6 GB over 10 episodes, then a lost Vulkan device in the randomized scenes of `pick_diverse_bottles`;
  - Qwen-Image-2.1 up to about 24 GB, plus about 17 GB for the text server on the first GPU. A server and a worker on one L40S ran out of Vulkan memory in randomized scenes.
- Cost reference:
  - Step limits per task are 400–1,700 actions (mean 620; `third_party/RoboTwin/task_config/_eval_step_limit.yml`). A failed episode runs to its limit.
  - On L40S, a successful `beat_block_hammer` episode took 10–19 s and the 4B worker ran 20 randomized `pick_diverse_bottles` episodes in 10.5 min including loading.
  - Estimate, to be measured in Phase 5: about 70 worker-hours per model (10,000 episodes at about 25 s each).

## Problem

No model trained here has a RoboTwin success rate, and the evaluation pipeline has not been checked against the paper. The full protocol is 30,000 episodes on 4 GPUs, so a run must survive a single worker failure.

## Goal

1. **Pipeline check:** the released FLUX.2-4B under the paper protocol. Its per-task and mean success are compared with paper Table 12 (93.20% clean, 93.56% randomized).
2. **Trained models:** FLUX.2-9B and Qwen-Image-2.1, both at step 65,000, under the same protocol.
3. **Results:**
   - in `docs/robotwin_evaluation_results.md`: per-task clean and randomized success of the three models, their means with standard errors, differences between models, episode time and GPU use;
   - in `docs/robotwin_training_results.md`: the training section of both runs.

Protocol, fixed for all three models:
- 50 tasks; clean (`demo_clean`) and randomized (`demo_randomized`); 100 episodes per task and setting;
- seed 42: the expert-checked seeds from 4,300,000; unseen instructions;
- 10 denoising steps, action horizon 16, replan 16, `compact_288x256`, text CFG scale 1.0.

Out of scope: retraining or continuing any run, real-robot evaluation, and ISSUE-012.

Constraint check against this goal:
- GPU policy: 4 GPUs of one node, within the 4-per-node limit (E2).
- Both trained models stop at step 65,000, 4,900 steps before the end of the schedule, where the learning rate is 2.32e-6 (peak 1e-4, floor 1e-6). The released 4B ran all 69,900 steps. This difference is stated with every comparison against the 4B, as is gradient clipping 1.0 (previous plan D9).
- Text slots: the FLUX.2 workers encode instructions into 512 slots, while training used 128. The extra slots are masked padding, and the proprio token is placed right after the valid text tokens (`ImageWAM/src/imagewam/models/backbones/imagewam.py`, `pack_proprio_after_text`). The conditioning is therefore the same apart from numerics. Qwen-Image-2.1 is evaluated with 128 slots.
- The Qwen-Image-2.1 evaluation features are encoded on an H100 (`.venv-qwenimage`). 66% of its training features were encoded on B300; the two differ by 2.2–2.6% relative L2 (`PROJECT.md`).
- wandb is not used by evaluation. Raw data is only read.

# Structure

## Modules

- **Launcher** `launch/robotwin_eval.sh`: binds a model name to its checkpoint, statistics, base weights and text path, and sets the protocol. It runs on `nnmc71` and refuses any GPU that holds a process.
- **Manager** `run_robotwin_manager.py`: one per model run. It schedules (task, phase) workers on the 4 GPUs, collects their success rates, and writes the summary. New: resume.
- **Worker:** `eval_robotwin_single.py` → RoboTwin `script/eval_policy.py` → `imagewam_policy/deploy_policy.py`. Unchanged.
- **Qwen-Image-2.1 text server** `scripts/qwenimage21/qwen_text_server.py`, started by the launcher on the first GPU. It loads only the processor and the text encoder.
- **Paper reference** `launch/robotwin_table12_flux2_4b.csv`: the FLUX.2-4B columns of paper Table 12.
- **Comparison** `launch/robotwin_eval_compare.py`:
  - reads each model's result directory and the reference;
  - checks episode counts, and per pair of models the shared seeds and instructions;
  - writes the result tables.
- **Results:** `docs/robotwin_evaluation_results.md` (evaluation), `docs/robotwin_training_results.md` (training).

## Decisions

- **E1 Checkpoints:** the three files in Current. Statistics come from each run directory's `dataset_stats.json`.
- **E2 Host:** `nnmc71` GPUs 0–3. One node runs all three models.
  - Its H100s hold every model, FLUX.2-9B included, which does not run reliably on a 46 GB L40S.
  - The evaluation environment was built there and needs no path changes.
  - Idleness is checked by the launcher right before each launch.
- **E3 Order:** Qwen-Image-2.1, then the released 4B, then FLUX.2-9B, each on all 4 GPUs.
  - The 4B run checks the pipeline against Table 12. A systematic deviation (example: most tasks below their Table 12 value by more than 3 binomial standard errors) is recorded in `issues.md` and stops the FLUX.2-9B run, and the Qwen-Image-2.1 result is not used until it is explained.
- **E4 Resume:** `MULTIRUN.resume=true`, rerun with the same `OUTPUT_TAG`.
  - A task with result files for all selected phases is read from those files and not rerun.
  - Any other existing task directory is renamed to `<task>.incomplete_<k>`, and the task reruns both phases from its first episode.
  - The rerun uses the same protocol and seed range. It does not reproduce the instructions or the kept seeds exactly: Python `random` is never seeded in the evaluation process (E7).
- **E5 Output names:** `OUTPUT_TAG` = `qwenimage21_step065000_full`, `flux2_4b_release_full`, `flux2_9b_step065000_full`.
- **E6 Workers per GPU (`MAX_TASKS_PER_GPU`):** fixed per model in Phase 5 from measured values:
  - GPU memory per worker within 94 GB; the first GPU also holds the Qwen-Image-2.1 text server (about 17 GB);
  - host memory while workers load. Each FLUX.2-9B worker reads the 54 GB weight file into host memory.
  - If simultaneous loads exceed free host memory, the manager starts workers at a fixed interval (`MULTIRUN.launch_interval_sec`).
- **E7 Model comparison:**
  - The evaluation process never seeds Python `random`. It draws the instructions and is used by cuRobo's graph planner. In the smoke runs the three models kept different seed lists for the same task, and almost no shared seed had the same instruction. This is the upstream protocol, under which the paper's numbers were produced, so it is kept.
  - The main comparison is the difference of means, with the binomial standard error over the 5,000 episodes per setting and model.
  - A paired difference over the (task, phase, seed) triples both models ran is reported beside it.

## Responsibilities

- The launcher owns the protocol settings and the GPU check. At launch only `MODEL`, `CKPT`, `GPUS`, `OUTPUT_TAG` and `MAX_TASKS_PER_GPU` are set.
- The manager owns scheduling and resume.
- The comparison script owns every number in the results tables.

## State Ownership

| State | Owner | Location |
|---|---|---|
| Evaluation results | manager and workers | `ImageWAM/evaluate_results/robotwin/<checkpoint stem>/<OUTPUT_TAG>/` |
| Paper reference values | reference file | `launch/robotwin_table12_flux2_4b.csv` |
| Result tables | comparison script | its output, copied into `docs/robotwin_evaluation_results.md` |
| Qwen-Image-2.1 evaluation text features | text server | `checkpoints/robotwin_eval_text_cache/qwenimage21/` |

# Interface

## Interfaces

- **`launch/robotwin_eval.sh`** (env `MODEL`, `CKPT`, `GPUS`, `OUTPUT_TAG`, `MAX_TASKS_PER_GPU`, `EPISODES`, `PHASES`, `TASKS`; Hydra overrides as arguments).
  - Resume is the override `MULTIRUN.resume=true`.
  - `GPUS` are CUDA indices. Each is resolved to its UUID through CUDA, and the launch stops if any of them holds a process.
- **Manager:** `MULTIRUN.resume: bool` (default `false`, `configs/sim_robotwin.yaml`).
  - When true, each task is either `done` (its rates are read) or `pending`. A pending task's existing directory is renamed first.
  - Without resume, an output directory that already holds task directories is refused.
  - `MULTIRUN.max_tasks_per_gpu`: one cap for every GPU, or a list with one cap per entry of `gpu_ids`.
- **`launch/robotwin_eval_compare.py --result <name>=<run dir> ... --reference launch/robotwin_table12_flux2_4b.csv --output <file.md>`:**
  - Input: `<task>/_result_{clean,random}.txt` and `<task>/episode_end_states.jsonl` of each run directory.
  - Typed records: `EpisodeRecord` (task, phase, seed, success, instruction, actions taken) and `TaskResult` (task, phase, success rate, episodes, run directory).
  - Output:
    - a per-task table (reference, then each model, clean and randomized) with means and standard errors;
    - a table of differences of means and paired differences on shared seeds;
    - printed checks: tasks and episodes per (task, phase), and per pair of models the shared seeds and instructions.
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

1. CPU only: the Table 12 reference and the comparison script; resume in the manager; the evaluation config check.
2. Smoke of Qwen-Image-2.1 on `nnmc71` GPUs 0–3 with 3 workers per GPU on randomized scenes; fix its workers per GPU.
3. Qwen-Image-2.1, full protocol.
4. Training section of both runs.
5. Released 4B, full protocol; compare with Table 12 (E3).
6. FLUX.2-9B: short memory check of 2 workers per GPU, then the full protocol.
7. Run the comparison script; write the evaluation section.

# Code Mapping

| Task | Files |
|---|---|
| Paper reference | `launch/robotwin_table12_flux2_4b.csv` |
| Comparison | `launch/robotwin_eval_compare.py` |
| Resume, per-GPU worker caps | `ImageWAM/experiments/robotwin/run_robotwin_manager.py`, `ImageWAM/configs/sim_robotwin.yaml` |
| Host and GPU check, workers per GPU | `launch/robotwin_eval.sh` |
| Text server memory | `ImageWAM/scripts/qwenimage21/qwen_text_server.py` |
| Evaluation config check, protocol facts | `docs/robotwin_training_setup.md` (Evaluation Protocol) |
| Evaluation results | `docs/robotwin_evaluation_results.md` |
| Training results | `docs/robotwin_training_results.md` |
| Run state, open problems | `PROJECT.md`, `issues.md` |

# Implementation

## Phase 1

Phase Status: completed

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

Result (2026-10-19):
- `launch/robotwin_table12_flux2_4b.csv`: 50 tasks, all matching `_eval_step_limit.yml`; means 93.20 (clean) and 93.56 (randomized), as in the paper.
- `launch/robotwin_eval_compare.py` on `smoke_2ep`: 2 clean episodes, seeds 4,300,000 and 4,300,001, success 1.0. On the three L40S smoke runs it reports the shared seeds and instructions per pair of models.
- The comparison reads the per-task result files, not `summary.json`, so it also reads runs that are still in progress.

## Phase 2

Phase Status: completed

Round: 1

### Goal

Manager resume (E4) and the launcher GPU check (E2).

### Files

`ImageWAM/experiments/robotwin/run_robotwin_manager.py`, `ImageWAM/configs/sim_robotwin.yaml`, `launch/robotwin_eval.sh`.

### Structures

`MULTIRUN.resume`.

### Affected Modules

Manager, launcher.

### Dependencies

None.

### Observation

- A resumed run keeps tasks with all result files and renames and reruns the others.
- The launcher refuses a GPU that holds a process and accepts an idle one.

Result (2026-10-19):
- `MULTIRUN.resume` in the manager and `configs/sim_robotwin.yaml`. Without resume, the manager refuses an output directory that already holds task directories.
- A Qwen-Image-2.1 run that had failed during loading was resumed with the same `OUTPUT_TAG`. The manager renamed its task directory to `beat_block_hammer.incomplete_0` and reran both phases; 10 of 10 episodes succeeded.
- The launcher resolves each CUDA index to its UUID through CUDA and refuses a GPU that holds a process. On `nnmc71` it refused GPU 0 (the FLUX.2-9B run); CUDA and `nvidia-smi` indices agree on all 8 `nnmc71` GPUs.

## Phase 3

Phase Status: completed

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
- Every difference is listed with its reason.
- An unexplained difference in a key that changes the network or the preprocessing stops the plan.

Result (2026-10-19): no unexplained difference; the list is in `docs/robotwin_training_setup.md` → Evaluation Protocol.

## Phase 4

Phase Status: completed

Round: 2

### Goal

Smoke of Qwen-Image-2.1 on `nnmc71` GPUs 0–3; its workers per GPU (E6).

### Files

`ImageWAM/experiments/robotwin/run_robotwin_manager.py`, `ImageWAM/configs/sim_robotwin.yaml` (per-GPU caps); `launch/robotwin_eval.sh` (Qwen-Image-2.1 default).

### Structures

Per-GPU worker caps.

### Affected Modules

Manager, launcher.

### Dependencies

Phases 2 and 3.

### Observation

Peak GPU memory per GPU, host memory, GPU utilization, episode time and success with 3 workers per GPU on randomized scenes.

Result (2026-10-19, `smoke_h100_qwen_w3`):
- 12 tasks × 5 randomized episodes, 12 workers, 12.9 min, no failure; success 60–100% per task.
- Peak GPU memory: 90.1 GB on GPU 0 (text server and 3 workers), 69.5–71.4 GB on GPUs 1–3. Mean GPU utilization 17–50%. Host memory available stayed above 610 GB.
- About 61 worker-seconds per episode with 3 workers per GPU, against 10–19 s for a lone worker on L40S, so the full run projects to about 14–16 h with 11 workers.
- Workers per GPU: 2 on the text-server GPU and 3 on the others (`MULTIRUN.max_tasks_per_gpu=[2,3,3,3]`, the launcher default for Qwen-Image-2.1).

## Phase 5

Phase Status: completed

Round: 3

### Goal

Qwen-Image-2.1 (step 65,000) under the full protocol.

### Files

None. The output is `ImageWAM/evaluate_results/robotwin/step_065000/qwenimage21_step065000_full/`.

### Structures

None.

### Affected Modules

Evaluation.

### Dependencies

Phases 1 and 4.

### Observation

- 100 episodes per (task, phase), clean and randomized, for all 50 tasks;
- per-task and mean success;
- wall-clock time, peak GPU memory, failed or resumed workers.

Result:
- Ran from 2026-10-19 16:49 to 2026-10-20 04:57 EDT (12.1 h) in tmux `imagewamft_eval_qwen`, log `checkpoints/robotwin_logs/eval_qwen_full.log`. 11 workers (`[2,3,3,3]`); no worker failed, no resume.
- 50 tasks × 2 phases × 100 episodes, 200 records per task.
- Mean success (`launch/robotwin_eval_compare.py`): clean 92.82 ± 0.33, randomized 94.00 ± 0.32. Paper Table 12 (FLUX.2-4B): 93.20, 93.56.
- Tasks beyond 3 binomial standard errors of Table 12: `open_microwave` clean 47 against 97 (ISSUE-014); `stamp_seal` clean 98 against 79 and randomized 97 against 84.
- 112.8 worker-hours, 40.6 worker-seconds per episode. One (task, phase) took 18 min (`press_stapler` clean) to 206 min (`open_microwave` clean).
- GPU memory, sampled every minute: peak 68.2 GB on GPU 0 (text server and 2 workers), 72.1–74.1 GB on GPUs 1–3. Mean GPU utilization 53–67%. Host memory available stayed above 280 GB.

## Phase 6

Phase Status: completed

Round: 4

### Goal

Training section of both runs.

### Files

`docs/robotwin_training_results.md` (Runs, Training).

### Structures

None.

### Affected Modules

Results.

### Dependencies

None.

### Observation

- Loss and validation curves of both runs up to their stops, with duplicate steps removed across the two FLUX.2-9B wandb runs (the newer run for steps 40,001–42,150).
- Validation metrics around step 65,000 for both runs, and wall-clock time.

Result: `docs/robotwin_training_results.md` → Training, and the final-weights and wall-clock rows of Runs. Done while the Phase 7 run occupied the GPUs.

## Phase 7

Phase Status: active

Round: 4

### Goal

Released FLUX.2-4B under the full protocol, compared with Table 12 (E3).

### Files

`docs/robotwin_evaluation_results.md`.

### Structures

None.

### Affected Modules

Evaluation.

### Dependencies

Phase 5 (GPUs).

### Observation

- Per-task and mean success, clean and randomized, against Table 12, with the per-task difference in binomial standard errors.
- Peak GPU memory with 3 workers per GPU; wall-clock time; failed or resumed workers.
- `open_microwave` clean success per microwave model (ISSUE-014).

Started 2026-10-20 04:59 EDT in tmux `imagewamft_eval_4b`, log `checkpoints/robotwin_logs/eval_4b_full.log`, 12 workers (3 per GPU). The output is `ImageWAM/evaluate_results/robotwin/model/flux2_4b_release_full/`.

## Phase 8

Phase Status: pending

Round: 5

### Goal

FLUX.2-9B (step 65,000) under the full protocol.

### Files

`launch/robotwin_eval.sh` (FLUX.2-9B default workers per GPU, from the memory check).

### Structures

None.

### Affected Modules

Evaluation.

### Dependencies

Phase 7 (pipeline check passed).

### Observation

- Memory check: peak GPU memory with 2 workers per GPU on randomized scenes.
- 100 episodes per (task, phase), per-task and mean success; wall-clock time; failed or resumed workers.

## Phase 9

Phase Status: pending

Round: 6

### Goal

Comparison of the three models and the evaluation write-up.

### Files

`docs/robotwin_evaluation_results.md`.

### Structures

None.

### Affected Modules

Results.

### Dependencies

Phases 1, 5, 7 and 8.

### Observation

- Per-task and mean success of the three models, with standard errors.
- Differences of means and paired differences on shared seeds: FLUX.2-9B − 4B, Qwen-Image-2.1 − 4B, and Qwen-Image-2.1 − FLUX.2-9B.
- The step and clipping differences against the 4B stated beside each comparison.
