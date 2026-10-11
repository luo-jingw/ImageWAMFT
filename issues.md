# ISSUE-001

Status: Resolved

Area: data / language conditioning

## Observation

drawer, biomanual and simple-pick all store the task string `pick up cube.`. Their recorded content is three different tasks (see `docs/lgg100_dataset.md` → Stored Task Strings and Observed Content).

## Impact

When the four datasets are trained jointly, the instruction cannot tell these three tasks apart, so the model has to infer the task from the scene alone. ImageWAM's conditioning relies on matching the instruction to the expected visual change. A physical-robot evaluation that issues the stored strings cannot select a task.

## Evidence

- `meta/tasks.jsonl`: one task per dataset.
- `meta/episodes.jsonl`: `tasks` is `["pick up cube."]` for all 100/50/100 episodes.
- Parquet `task_index`: 0 in every row.
- Hub `main` HEAD `meta/tasks.jsonl` is byte-identical to the local copy: drawer `b087c569`, biomanual `28f8d752`, simple-pick `c4bbc4d4`. Each of these repos has only `main` and two commits (initial and upload). Only Stack-the-cubes has an edited task string, shown by the commit "Delete meta/tasks.jsonl.bak".
- Sampled frames: drawer episode 3; biomanual episodes 0, 10, 20, 35, 49; simple-pick episodes 0, 30, 60, 95.

## Hypotheses

The three datasets were recorded with a reused default task string.

## Next Experiment

Build the Qwen3 text-embedding cache from the relabeled task strings.

## Resolution

The local task strings for drawer, biomanual and simple-pick are replaced with one instruction per task; they are listed in `docs/lgg100_dataset.md` → Task Strings. Stack-the-cubes keeps its hub string. The hub originals are kept in `dataset/LGG100_original_meta/`. Validation:
- No `pick up cube.` remains under `dataset/LGG100`.
- Every non-task field of `meta/episodes.jsonl` is identical to the hub original; episode counts are 100, 50 and 100.
- `LeRobotDatasetMetadata.tasks` and `LeRobotDataset[0]["task"]` (simple-pick episode 0) return the new strings.
- Only the six edited meta files changed under `dataset/LGG100`.

# ISSUE-002

Status: Open

Area: data

## Observation

Eleven episodes contain no valid demonstration: biomanual 24; simple-pick 22, 42, 43, 44, 45, 46, 47, 48 and 49; Stack-the-cubes 51 (see `docs/lgg100_dataset.md` → Data Quality Audit → Invalid Episodes). simple-pick 42–48, apart from 46, record a human resetting the scene while the robot is static. Stack-the-cubes 51 ends before the last block is stacked.

## Impact

Every frame is a training sample, so these 4,270 frames add targets that do not demonstrate the task: static robots with human arms in view, repeated attempts and an unfinished stack. simple-pick loses 16% of its frames when they are excluded. biomanual 24 (8 frames) is shorter than one 17-frame sample window.

## Evidence

- Episode-length IQR fences per dataset.
- Gripper closing counts against task structure; they are unchanged across two hysteresis settings.
- Head-camera frame grids of every flagged episode.

## Hypotheses

They are recording errors: reset phases recorded as episodes, multiple trials in one episode, or a premature stop.

## Next Experiment

Exclude the eleven episodes through a new `explicit_exclude` filter mode (`plan.md` → Phase 1, decision D7).

## Resolution

# ISSUE-003

Status: Resolved

Area: data provenance

## Observation

`info.json` declares `robot_type: Unitree_G1_Dex1_Sim`, while the sampled video frames show physical-robot footage: fabric texture, room background, motion blur and a human hand in simple-pick 46.

## Impact

A policy trained on simulator data and then evaluated on the physical robot would face a sim-to-real gap. A policy trained on physical-robot data would not.

## Evidence

`dataset/LGG100/*/meta/info.json`; frame grids from all four datasets.

## Hypotheses

`Unitree_G1_Dex1_Sim` is the name of the collection pipeline's robot configuration and does not describe the data source.

## Next Experiment

Confirm the data source with the dataset owner.

## Resolution

The data is physical-robot teleoperation on a Unitree G1 29-DoF with Dex1-1 grippers, as confirmed by the data owner.
- `Unitree_G1_Dex1_Sim` is the `unitree_lerobot` conversion config for the 3-camera layout. It has the same 16 motors as `Unitree_G1_Dex1`, which has 4 cameras.
- The joint ranges are consistent with the 29-DoF arm (`docs/lgg100_dataset.md` → Robot Kinematics).

# ISSUE-004

Status: Resolved

Area: training configuration

## Observation

The paper gives the real-world training length as both 30k steps (§4.1) and 10 epochs (Appendix 5.2). On LGG100 these two definitions give different training lengths.

## Impact

The training budget and the overfitting risk depend on which definition is used.

## Evidence

- With every frame as one sample, 30k steps = 10 epochs implies about 384k frames per epoch at the paper's real-world global batch of 128 (Table 10: 16 per GPU × 8 GPUs).
- LGG100 has 181,403 frames after excluding the ISSUE-002 episodes, about 47% of that.
- 10 epochs ≈ 28.1k optimizer steps with the repository 9B config: 4 per GPU × 8 GPUs × gradient accumulation 2 = effective batch 64.
- 10 epochs ≈ 14.0k steps at global batch 128.
- 30k steps at global batch 128 ≈ 21 epochs.

## Hypotheses

The epoch count is the data-relative quantity that carries over to a dataset of a different size. With the 9B config, 10 epochs also lands near 30k steps.

## Next Experiment

Fix the training length. Measure step time in a short run to estimate wall-clock time.

## Resolution

Training length is 10 epochs over the 181,403 frames left after the ISSUE-002 exclusions, the paper's Appendix 5.2 definition. That is about 28.1k optimizer steps at effective batch 64. Wall-clock time is measured in a short run.

# ISSUE-005

Status: Open

Area: data distribution

## Observation

With uniform per-frame sampling, Stack-the-cubes supplies 52.5% of the training samples after ISSUE-002 exclusions. drawer supplies 26.3%, biomanual 11.0% from 49 episodes, and simple-pick 10.2% from 91 episodes.

## Impact

biomanual is a bimanual, color-matching task with the fewest episodes. It receives about one fifth of Stack-the-cubes' gradient share.

## Evidence

Frame counts in `docs/lgg100_dataset.md` → Data Quality Audit → Invalid Episodes.

## Hypotheses

biomanual will have the lowest success rate under paper-default sampling.

## Next Experiment

Report physical-robot success per task. Task-balanced sampling is tracked as OPT-001.

## Resolution

# ISSUE-006

Status: Resolved

Area: robot model

## Observation

The robot's `mode_machine` is unknown, so the exact G1 URDF variant cannot be selected. The datasets do not record it.

## Impact

Wrist kinematics differ between variants. The `wrist_yaw_joint` origin is 0.051 m in mode 15 (5010 wrist motor) and 0.046 m in `rev_1_0` (mode 5, 4010 wrist motor). Forward kinematics, visualization and end-effector-space conversion of the joint data depend on the choice. Joint-space training does not.

## Evidence

`unitree_ros/robots/g1_description/README.md` variant table. Origin comparison of `g1_29dof_mode_15_with_dex1_1.urdf` and `g1_29dof_rev_1_0.urdf`.

## Hypotheses

The robot is mode 15, the only official 29-DoF variant with a Dex1-1 URDF.

## Next Experiment

Read Machine Type in the Unitree app (Device → Data → Robot).

## Resolution

Arm kinematics use the mode-15 arm chain from `g1_29dof_mode_15_with_dex1_1.urdf`. The wrist-yaw offset uncertainty of up to 5 mm against other 29-DoF revisions is accepted. Gripper and camera hardware are retrofits and are not modeled by the URDF.

# ISSUE-007

Status: Open

Area: compute availability

## Observation

On 2026-10-01, GPUs 0, 2 and 3 of the 8× H100 NVL host were occupied by other users' processes (about 82 GB each), and GPUs 4 and 5 ran other processes at high utilization. GPUs 1, 6 and 7 were free.

A FLUX.2-9B G1 smoke run on GPUs 1, 6 and 7 (ZeRO-2, batch 4 per GPU, gradient accumulation 2) ran out of memory in DeepSpeed `copy_grads_in_partition`. At that point each process used 89.30 GB: 77.28 GB allocated and 10.51 GB reserved but unallocated, and a 6.37 GB allocation failed.

## Impact

The plan's 8-GPU smoke measurements (Phases 5 and 12) cannot be taken while other jobs hold GPUs. ZeRO-2 shards the fp32 optimizer state (about 120 GB for about 10B trainable parameters) across GPUs, so per-GPU memory at 3 GPUs exceeds the 8-GPU configuration by about 25 GB.

## Evidence

`ImageWAM/runs/smoke_logs/flux2_9b.log`, `ImageWAM/runs/smoke_logs/flux2_9b_gpumem.csv`, `nvidia-smi` at launch.

## Hypotheses

The 8-GPU configuration fits: about 15 GB optimizer shard, 20 GB bf16 weights, gradient partition and buckets, and activations. Fragmentation causes the 3-GPU failure, which `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces.

## Next Experiment

Rerun the smoke test on the free GPUs with `expandable_segments:True`, and report memory and step time at that GPU count.

The allowed nodes are `nnmc61`–`nnmc71`. Only `nnmc71` has 94 GB GPUs (H100 NVL); `nnmc61`–`nnmc70` have 8× L40S with 46 GB each (`PROJECT.md` → Environment → Cluster). The paper's FLUX.2-9B recipe (full fine-tuning, ZeRO-2, 8 GPUs) needs roughly 50 GB per GPU: 20 GB bf16 weights, about 15 GB optimizer shard, gradient partition and buckets, and activations. That does not fit an L40S, which leaves two options:
- the paper recipe on `nnmc71` when 8 of its GPUs are free;
- a recipe change for L40S nodes: ZeRO-3, optimizer CPU offload, or multi-node ZeRO-2 over the cluster network.

## Resolution

# ISSUE-008

Status: Resolved

Area: training numerics (FLUX.2-9B, DeepSpeed ZeRO-2)

## Observation

The FLUX.2-9B G1 smoke run (`nnmc71` GPUs 1, 5, 6, 7; ZeRO-2; bf16; batch 4 per GPU; gradient accumulation 2; `gradient_clipping` 1.0) logged `loss=nan`, `loss_action=nan` and `loss_video=nan` at every logged step from step 10 to step 100, and validation at step 100 returned `val_loss=nan` and `action_l2=nan`. The offline wandb history records `train/grad_norm` = 1 at every logged step.

## Impact

The FLUX.2-9B run cannot train. The same DeepSpeed path will carry the Qwen-Image-2.1 run.

## Evidence

- The logged `grad_norm` of exactly 1 means the gradients were non-finite. DeepSpeed 0.18.5 ZeRO-1/2 (`stage_1_and_2.py`) replaces a non-finite partition norm with −1 (`mask_nan_or_inf_with_val_inplace`). The global norm `vector_norm(stack([-1]))` is then 1. The clip coefficient `(norm + 1e-6) / clip_grad` is clamped to at least 1, so nothing is clipped. bf16 mode does not check for overflow, so the non-finite gradients reach the AdamW update.
- Ruled out on one GPU (`nnmc71` GPU 6) without DeepSpeed, with initial weights and real training batches:
  - Pure bf16 without autocast, which is what accelerate does under DeepSpeed (`native_amp=False`): 4 batches of 4 samples gave finite losses of 1.6–4.0, finite gradients with norms of 25–67, and no non-finite forward output in any module.
  - A full-module `.bfloat16()` cast, as DeepSpeed performs: the model has no fp32 buffers, and the results are unchanged.
  - Inputs, normalization statistics and the action-DiT initialization are finite; the minimum std is 0.145.
  - No trainable parameter is left without a gradient (0 unused).
  - The action loss clamps its denominators at 1, so it cannot produce 0/0.

## Hypotheses

1. DeepSpeed ZeRO-2 gradient reduction or partitioning produces non-finite values, through `overlap_comm` with contiguous gradients and accumulation.
2. The first optimizer updates make the weights unstable, and a later forward or backward pass overflows.
3. A rare training sample, not among the 16 checked, produces a non-finite gradient.

## Next Experiment

- Reproduce under DeepSpeed with per-step logging (`log_every=1`) to find the first non-finite step. A LoRA-on-video variant fits on 2 H100s.
- Then run a grid over `gradient_clipping` ∈ {0, 1.0} × `overlap_comm` ∈ {true, false}.

## Resolution

Cause: a DeepSpeed ZeRO-2 per-rank flat partition larger than 2^31 elements produces non-finite gradients from the first optimizer step. Hypothesis 1 holds in its partition-size form; hypotheses 2 and 3 are refuted, since the loss is finite at step 1 and NaN from step 2 regardless of data.
- In the 3-cell grid (`docs/g1_training_setup.md` → Optimizer Parameter Groups), only the cell with a partition above 2^31 diverged. Total group size above 2^31 alone (cell B) did not cause it.
- Fix: `Trainer._split_param_groups` with `optimizer_max_group_numel: 1000000000` in both G1 task configs. Hyperparameters are identical across groups, so only the buffer layout changes.
- The fix applies equally to the Qwen-Image-2.1 run, which has about 8B trainable parameters.

# ISSUE-009

Status: Resolved

Area: training throughput (Qwen-Image-2.1, DeepSpeed ZeRO-2)

## Observation

The first Qwen-Image-2.1 G1 smoke run (`nnmc72` GPUs 4,5; ZeRO-2; bf16; 4 per GPU; gradient accumulation 8) trained at 0.08 optimizer steps/s, which projects to about 97 h for 27,950 steps. Per micro-batch, forward took 0.18 s and backward 1.5 s. On one GPU without DeepSpeed, the same model and batch took 0.18 s forward and 0.47 s backward.

## Evidence

- Not memory pressure: an allocator probe on both ranks showed a 78.2 GiB peak, 0 allocation retries and 8.5 GiB free.
- Not transport: `NCCL_DEBUG=INFO` shows P2P/IPC over the NVLink pair.
- CPU-side DeepSpeed overhead: a torch-profiler window of two backward calls on rank 0 recorded 3.8 s of CPU and 1.3 s of GPU. Of that, 2.1 s of self-CPU was in `AccumulateGrad`, with 707,404 `aten::view_as` calls.
- Cause: DeepSpeed 0.18.5 ZeRO-1/2 runs `count_used_parameters_in_backward(all_params_requiring_grad)` inside every gradient hook (`stage_1_and_2.py:1028`). That walks all 594 trainable tensors (`view_as` and a node lookup for each), making one backward O(n²) on the CPU.
- CPU optimizer offload (ZeRO-2 `offload_optimizer: cpu`, `DeepSpeedCPUAdam`, which matches torch AdamW to 1.4e-6 after 3 steps) ran at 17–20 s per step and does not help.

## Resolution

- `ImageWAM/src/imagewam/utils/deepspeed_patches.py` caches the count per autograd graph task. Within a graph task the count is fixed, because the engine's execution info is set when the task starts. The returned values are therefore identical and only the CPU work changes.
- Enabled through `deepspeed_cache_used_param_count: true` (default in `configs/train.yaml`) and applied by the trainer before DeepSpeed initialization.
- Result on the same GPUs: backward 0.69 s per micro-batch and 0.15 steps/s, about 52 h for the full run.
- The running FLUX.2-4B process started before the patch and has the same overhead on its smaller parameter set. Its math is unaffected.

# ISSUE-010

Status: Open

Area: shared nodes (`nnmc72` GPU processes)

## Observation

Since at least 2026-10-08, every process of ours that holds a GPU on `nnmc72` receives SIGTERM at :07 and :37 of each hour. This hit the Qwen-Image-2.1 RoboTwin cache twice (`plan.md` Phase 3).

## Impact

- `nnmc72` GPUs cannot run anything longer than about 30 minutes: training, the text cache, or RoboTwin evaluation (`plan.md` D6).
- The G1 runs on `nnmc72` (2026-10-01 to 10-05) were not affected, so this behavior is newer.

## Evidence

- All 8 cache processes exited with code 143 (SIGTERM), first at 12:37:05 and then at 13:07:05 PDT, each time partway through their shards. Logs: `checkpoints/robotwin_logs/qwen_cache_run1/` and `cache_qwen_shard*.log`.
- The `nnmc72` sshd log shows `user1` sessions from `10.145.87.57` (`nnmc57`) about 6 times per minute since at least 10:00. At :07 and :37 the count doubles to 14 per minute, which matches the kill times.
  - `nnmc57` runs other people's jobs under the shared `user1` account. This project does not use it, and it was not inspected.
- Probe at the 13:37 sweep:
  - Probes started at 13:09: a CUDA process on GPU 0, one on GPU 4, and a CPU-only Python process. All had `ImageWAMFT` in their command lines.
  - Both GPU processes received SIGTERM at 13:37:03. The CPU process was not touched.
- Over the same sweeps, this project's jobs on `nnmc71` (FLUX.2-9B cache) and `nnmc73` (Qwen-Image-2.1 cache) were not touched. The `nnmc71` sshd log visible to `user1` shows no `nnmc57` sessions.

## Hypotheses

Another user's automation on `nnmc57` terminates GPU processes on `nnmc72` every 30 minutes, possibly to reserve the node.

## Next Experiment

None from this project. Whoever owns the `nnmc57` automation has to exclude this project's processes, or the user decides that `nnmc72` is no longer available.

## Resolution

Open.
- Workaround for the cache: it moved to `nnmc73` (`checkpoints/robotwin_logs/cache_qwen_nnmc73.sh`, `.venv-qwenimage-nnmc73`).
- Cache writes are atomic, so the files written on `nnmc72` are valid.
- RoboTwin evaluation (`plan.md` D6) needs another GPU host or this issue resolved.

# ISSUE-011

Status: Open

Area: shared nodes (`nnmc73` GPUs during the RoboTwin Qwen-Image-2.1 run)

## Observation

From 2026-10-11 about 07:45 UTC, another job runs on all 8 `nnmc73` B300 next to the Qwen-Image-2.1 RoboTwin run. The run slowed from 4.05 s to about 20 s per optimizer step.

## Impact

- Qwen-Image-2.1 completion moves from about 22 hours after the slowdown to about 4.6 days, if the other job stays.
- The training math is unaffected: the step count, data order and optimizer are unchanged, and the log shows no errors or non-finite losses.

## Evidence

- Step times from `train.log`: 250 steps took 16.9 minutes until step 47,750 (07:42:58 UTC), then 47 minutes for the next 250 steps (08:30:16), then 75 to 87 minutes per 250 steps up to step 50,000 (15:23:25).
- `nvidia-smi` at 15:45 UTC: each GPU holds one of this run's processes (55 GB) and one `train_cora.py` process (`/home/user1/miniconda3/envs/nnmc_llm`, Llama-2-7B or Llama-3-8B DoRA fine-tuning, 105–155 GB). Six of those started at about 07:48 UTC and two at about 14:38 UTC.
- The `train_cora.py` command lines do not contain `ImageWAMFT`, so they belong to another user of the shared `user1` account.
- At the same time, every GPU of `nnmc72`, `nnmc74` and `nnmc75` was in use, so no other B300 node or 8-GPU node was free to move the run to.

## Hypotheses

Two processes share each GPU by time-slicing, and the other job takes most of the compute time.

## Next Experiment

None from this project. Another user's processes are not stopped from here. The run continues and returns to about 4 s per step when the other job ends. The latest resumable state is `checkpoints/state/step_050000`.

## Resolution

Open.
