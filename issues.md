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
