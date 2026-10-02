# LGG100 Datasets

Source: Hugging Face dataset repos `LGG100/drawer`, `LGG100/biomanual`, `LGG100/simple-pick` and `LGG100/Stack-the-cubes`. They are stored under `dataset/LGG100/<name>/`. Every file was checked against the hub file sizes.

## Format

- LeRobot `codebase_version` v2.1, 30 fps, one chunk (`chunk-000`).
- Data source: physical-robot teleoperation on a Unitree G1 (29-DoF). The grippers are retrofitted Dex1-1 units, and the camera hardware and mounting may differ from the Unitree standard. Only the arms and grippers are recorded.
- `info.json` `robot_type`: `Unitree_G1_Dex1_Sim`. This is the `unitree_lerobot` conversion config name (`unitree_lerobot/utils/constants.py` → `ROBOT_CONFIGS`). `Unitree_G1_Dex1` and `Unitree_G1_Dex1_Sim` define the same 16 motors and differ only in cameras: the first has 4 (`cam_left_high`, `cam_right_high`, `cam_left_wrist`, `cam_right_wrist`), the `_Sim` config has 3 (`cam_left_high`, `cam_left_wrist`, `cam_right_wrist`). The datasets have the 3-camera layout.
- Cameras: `observation.images.cam_left_high`, `observation.images.cam_left_wrist` and `observation.images.cam_right_wrist`. Each is 480×640×3, AV1-encoded, and decodes with torchcodec 0.5 (CPU) through `imagewam.datasets.lerobot.lerobot.datasets.video_utils.decode_video_frames`.
- `observation.state` and `action` are float32 with shape [16]: left arm joints 0–6 (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw), right arm joints 7–13 (same order), left gripper 14, right gripper 15.
- Each dataset has exactly one task (`task_index` 0 in every parquet row).

## Robot Kinematics

- The motor order matches `unitree_lerobot` `G1_DEX1_CONFIG`. The 14 arm dimensions map to URDF joints `{left,right}_{shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw}_joint`.
- Observed state and action ranges, over all 185,673 frames, lie within the arm joint limits of the official G1 29-DoF URDFs. The limits are the same in `g1_29dof_mode_15_with_dex1_1.urdf` and `g1_29dof_rev_1_0.urdf`.
- Wrist pitch and wrist yaw move through large ranges; for example, left wrist pitch spans −0.80 to 1.61 rad. The 23-DoF G1 has neither joint.
- The left and right shoulder-roll ranges are mirrored, matching the URDF's mirrored limits.
- Values reach joint limits: left and right wrist pitch at 1.614 rad (upper limit) and left elbow at −1.047 rad (lower limit).
- Gripper dimensions 14 and 15 are Dex1-1 motor positions `q` in rad, with offset `MAPPED_MIN` = 0 (`unitree_lerobot/eval_robot/robot_control/robot_hand_unitree.py`). In that code, 0 is the closed position at startup. Observed range: 0.38–5.42. The URDF models each Dex1-1 as two prismatic finger joints (−0.02 to 0.0245 m), so gripper values need a motor-to-finger mapping before URDF use.

### URDF Sources

`unitreerobotics/unitree_ros`, `robots/g1_description/` (variant table in its README):

| File | `mode_machine` | Wrist motor | End effector |
|---|---|---|---|
| `g1_29dof_mode_15_with_dex1_1.urdf` | 15 | 5010 | Dex1-1 (2 prismatic joints per gripper) |
| `g1_29dof_mode_15.urdf` | 15 | 5010 | none |
| `g1_29dof_rev_1_0.urdf` | 5 | 4010 | none |

- `left/right_wrist_yaw_joint` origin x is 0.051 m in the mode-15 files and 0.046 m in `rev_1_0`. The other arm joint origins are identical. Arm kinematics use the mode-15 arm chain, which leaves an uncertainty of up to 5 mm in the wrist-yaw offset.
- The robot's `mode_machine` is shown in the Unitree app under Device → Data → Robot → Machine Type.
- `unitreerobotics/xr_teleoperate`, `assets/g1/g1_body29_hand14.urdf`, is the 29-DoF body with Dex3 hands used for teleoperation IK.

## Statistics

Measured over all episodes.

| Dataset | Episodes | Frames | Frame share | Length min / median / max (s) | Static frames |
|---|---|---|---|---|---|
| drawer | 100 | 47,741 | 25.7% | 10.2 / 15.9 / 21.3 | 0.0% |
| biomanual | 50 | 19,951 | 10.7% | 0.3 / 13.4 / 18.2 | 0.0% |
| simple-pick | 100 | 22,015 | 11.9% | 2.1 / 6.7 / 68.5 | 1.4% |
| Stack-the-cubes | 100 | 95,966 | 51.7% | 21.2 / 31.3 / 50.9 | 3.6% |
| Total | 350 | 185,673 | 100% | | |

A frame counts as static when the L2 norm of its per-frame action change is below 1e-3 rad. Leading and trailing static segments are at most 0.3% of frames in any dataset.

## Task Strings

Each dataset has one task string, stored in `meta/tasks.jsonl` (`task_index` 0) and in every `meta/episodes.jsonl` `tasks` field. ImageWAM reads it through `LeRobotDatasetMetadata.tasks[task_index]`.

| Dataset | Task string | Hub original |
|---|---|---|
| drawer | `Use the arm nearer to the drawer to pull it open, then use the other arm to pick up the purple cube and place it inside the drawer, and finally close the drawer.` | `pick up cube.` |
| biomanual | `Pick up the blue cup with the nearer arm and place it on the blue coaster, then pick up the green cup with the nearer arm and place it on the now-empty green coaster.` | `pick up cube.` |
| simple-pick | `Pick up the pink cube with the nearer arm and place it into the green bin.` | `pick up cube.` |
| Stack-the-cubes | `Stack the blocks by color: put the red block in the center, then stack the blue block on the red block, then stack the yellow block on the blue block.` | identical |

The hub originals for drawer, biomanual and simple-pick are kept in `dataset/LGG100_original_meta/<name>/meta/`. Apart from the task strings, `meta/tasks.jsonl` and `meta/episodes.jsonl` match the hub files field for field.

## Data Quality Audit

Covers all 350 episodes and all 1,050 videos. Each automatic flag was checked against sampled head-camera frames.

### Integrity

- `frame_index` is contiguous in every episode, `episode_index` is consistent, there are no NaN values, and every episode starts at timestamp 0.
- Parquet timestamps step exactly 1/30 s (max deviation 0.007 ms). They are generated on a fixed grid and carry no capture-timing information.
- Every video is 640×480 AV1 at 30 fps. Each camera's video frame count equals the episode's parquet row count in all 350 episodes.

### Video Content

Every frame was decoded. Change is measured as the per-frame mean absolute difference on an 8×-subsampled grayscale image.

- No dark frames (mean luma < 15). Median luma per camera: head 115–117, wrists 145–152, consistent across datasets.
- Exact duplicate frames (zero change) make up a median 3–7.5% of frames per camera in every dataset, so capture runs below 30 fps with repeated frames.
- The longest duplicate runs are 21 frames (0.7 s). 23 episodes have a run of at least 8 frames. Runs often occur on all three cameras at once while the arm state keeps changing.
- Frames inside a duplicate run of at least 5 frames while an arm moves faster than 0.005 rad/frame: Stack-the-cubes 0.24%, drawer 0.10%, simple-pick 0.01%, biomanual 0%.

### Gripper Events

A closing event is counted on the state value with hysteresis: closed below 4.9, open above 5.2. The counts are the same with thresholds 5.0 / 5.25. Grasp closure depends on object and grip: about 0.6 on the drawer handle, 2.8–3.6 on cubes, and up to 4.6 on cup rims.

| Dataset | Expected closings | Episodes matching | Non-matching episodes |
|---|---|---|---|
| drawer | 2 | 100/100 | none |
| biomanual | 2 | 49/50 | 24 (0) |
| simple-pick | 1 | 91/100 | 22, 42, 43, 44, 45, 47, 48 (0); 46 (3); 49 (4) |
| Stack-the-cubes | 3 | 97/100 | 0, 1, 51 (2) |

Stack-the-cubes episodes 0 and 1 start with the red block already in the center, so 2 closings complete the task.

### Kinematics

- Action leads state by 3 frames (100 ms) in every episode, with two exceptions. Stack-the-cubes episodes 0–20 lead by 4 frames, apart from three episodes at 3. simple-pick episodes 42–48 show 5–8 frames, but their arms are nearly static, so the estimate is not meaningful there.
- Mean absolute action–state tracking error at that lag is 0.004–0.006 rad (median per dataset).
- The largest per-frame state change on any arm joint is 0.126 rad over all episodes. No state frame changes faster than 0.15 rad/frame.
- 14 valid episodes have isolated action frames above 0.15 rad/frame on an arm joint, max 0.226: Stack-the-cubes 6, 20, 29, 44, 52, 53, 59, 89, 93, 95 and drawer 50, 53, 77, 80. simple-pick 46 has 8 such frames, max 0.51.
- Frames within 0.02 rad of a URDF joint limit: drawer 1.31% for the left elbow, all other joints below 0.2%.

### Cross-Dataset Consistency

- Gripper open level, as the episode maximum of state: 5.37–5.42 in every dataset. The initial gripper value differs by at most 0.013 between datasets.
- Initial arm pose: per-joint means agree within 0.15 rad across datasets, except the elbows. Stack-the-cubes starts with straighter elbows, left −0.02 and right −0.06 rad, against −0.17 to −0.28 rad elsewhere; per-dataset std is 0.08–0.16.
- Joint-to-image consistency: frame pairs from different datasets with matching single-arm joint states (max per-joint difference ≤ 0.03 rad) show that gripper in the same head-camera image location. Pairs checked: Stack-the-cubes–biomanual, Stack-the-cubes–simple-pick, Stack-the-cubes–drawer, drawer–biomanual.
- Wrist cameras: the gripper fingers occupy the same image region in every dataset.
- Head-camera framing: Stack-the-cubes frames show only the tabletop, while the other three datasets show the wall line at the top edge. With the arm-to-camera relation unchanged, this reflects a different robot placement relative to the table. Stack-the-cubes was uploaded 2026-07-02; the other datasets were uploaded 2026-09-04 and 2026-09-11.

### Invalid Episodes

| Dataset | Episode | Frames | Content |
|---|---|---|---|
| biomanual | 24 | 8 | No task motion. |
| simple-pick | 22 | 62 | The cube is in the box for the whole episode. |
| simple-pick | 42, 43, 44, 45, 47, 48 | 89–135 each | A human arm resets the scene (cube from box to table) while the robot stays static. |
| simple-pick | 46 | 2,056 | Several attempts, including a human hand repositioning the cube. |
| simple-pick | 49 | 793 | The cube starts in the box, and the head camera view shifts during the episode. |
| Stack-the-cubes | 51 | 710 | Incomplete: the recording ends while the right arm reaches for the yellow block, which is never stacked. |

Without these 11 episodes, 339 episodes and 181,403 frames remain:

| Dataset | Episodes | Frames | Share |
|---|---|---|---|
| drawer | 100 | 47,741 | 26.3% |
| biomanual | 49 | 19,943 | 11.0% |
| simple-pick | 91 | 18,463 | 10.2% |
| Stack-the-cubes | 99 | 95,256 | 52.5% |

### Training and Validation Split

Measured with `BaseLerobotDataset` using `episode_index_filter: {mode: explicit_exclude}` (the 11 invalid episodes above) and `val_set_proportion: 0.01`, `val_split_level: episode`, seed 42:

| Split | Episodes | Frames |
|---|---|---|
| Training | 335 | 178,864 |
| Validation | 4: episode 8 of each root (drawer 496, biomanual 363, simple-pick 210, Stack-the-cubes 1,470 frames) | 2,539 |

At effective batch 64 (4 per GPU × 8 GPUs × gradient accumulation 2), one epoch is 2,795 optimizer steps, so 10 epochs are 27,950 steps.
