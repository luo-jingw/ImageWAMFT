# RoboTwin 2.0 Training Setup

Verified facts for training ImageWAM on RoboTwin 2.0 so that it matches the released FLUX.2-4B RoboTwin model. Sources are named per item. Checked on 2026-10-08.

## Released Reference Model

- Hugging Face `yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin`, stored at `checkpoints/imagewam_release/robotwin/flux2_klein_4b/` (`model.pt`, `config.yaml`, `dataset_stats.json`).
- It is the only released RoboTwin model. The authors' collection has no FLUX.2-9B or Qwen-Image RoboTwin model, and the paper reports no FLUX.2-9B RoboTwin result.
- `yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin-C2R-FT` is an InternData-A1 pretrain → fine-tune model released after the paper. It is not the paper recipe.
- `model.pt` stores `step` 69900 and `torch_dtype` bfloat16.
- The paper reports, for FLUX.2 4B on RoboTwin, 93.20% (clean) and 93.56% (randomized); per-task results are in its Table 12.

## Code Lineage

- The released `config.yaml` names `fastwam.*` modules (`fastwam.runtime.create_fastwam_flux2_klein`, `FastWAMProcessor`). It was produced by the authors' internal FastWAM-derived code.
- The public ImageWAM repository (`ImageWAM/`, upstream `5d4a341`, the latest upstream commit as of 2026-10-08) is that code renamed:
  - Every key of the released config exists in the public ImageWAM code, including the release-only keys `robotwin_camera_layout`, `nonidle_filter_path` and `endpoint_frames_only`.
  - None of these keys exist in the public FastWAM code (`yuantianyuan01/FastWAM`).
- Field-by-field diff between the released config and the config the repository composes (`task=robotwin_flux2_klein_4b_base_imagewam` plus the overrides of `scripts/flux2/run_train_flux2_klein_imagewam.sh`):
  - Differences are limited to module names, file paths and the validation split's non-idle filter.
  - The repository additionally has this project's keys `optimizer_impl` (default `torch_adamw`, the same AdamW), `optimizer_max_group_numel` (default `None`, one parameter group) and `deepspeed_cache_used_param_count` (a speed-only patch).
- This project's changes to `ImageWAM/` since import add Qwen-Image-2.1 code paths and an episode-exclusion loader mode. They remove no FLUX.2 code.
- `ImageWAM/third_party/flux2` is FLUX.2 at `50fe516` (2026-03-12). That commit is the last change to `src/flux2/model.py` upstream.

## Data

- Dataset: Hugging Face `yuanty/robotwin2.0-fastwam`, the FastWAM preprocessed RoboTwin 2.0 in LeRobot v2.1 format. It is extracted at `dataset/robotwin2.0/robotwin2.0/`, 75 GB.
- Size: 27,500 episodes and 6,075,103 frames at 50 fps.
  - 50 tasks × 550 episodes: per task, the first 50 episodes are clean scenes and the next 500 are randomized. The `_clean` configs keep `periodic_prefix` 50 of every 550.
- Normalization statistics: the dataset's `dataset_stats.json` is identical, value for value, to the one shipped with the released model. The configs read it through `pretrained_norm_stats`.
- State and action are 14-dimensional. The cameras are `cam_high`, `cam_left_wrist` and `cam_right_wrist`, each 480×640.
- Non-idle filter: `ImageWAM/scripts/data/compute_robotwin_nonidle_ranges.py` with default thresholds wrote `dataset/robotwin2.0/robotwin2.0/nonidle_ranges.json`. It keeps 89.29% of the steps.

## Instructions and Prompt

- Task strings: `meta/tasks.jsonl` holds 921,032 distinct strings. Each episode lists 41–50 of them in `meta/episodes.jsonl`.
- Assignment: `task_index` changes at almost every frame. A sample's instruction is the task string of its frame, so one trajectory is trained with about 50 phrasings.
- Templates: every one of 2,003 sampled strings matches a RoboTwin `seen` template (`third_party/RoboTwin/description/task_instruction/*.json`, 50 seen and 10 unseen templates per task).
- Prompt:
  - `DEFAULT_PROMPT.format(task=...)` with `DEFAULT_PROMPT` = "A video recorded from a robot's point of view executing the following instruction: {task}". It is identical in ImageWAM and FastWAM.
  - No high-level instruction: `drop_high_level_prob` defaults to 1.0. The authors confirm in ImageWAM issue #9 that no paper experiment uses one.
- Length: with the Qwen3 chat template, prompts have a median of 48 and a maximum of 79 tokens (Qwen3-4B tokenizer), so the training context length of 128 truncates none. Evaluation encodes online with the task config's 512 slots; the extra slots are masked padding.

## Training Recipe of the Released Model

From the released `config.yaml` and `model.pt`:

| Item | Value |
|---|---|
| Per-GPU batch × gradient accumulation | 12 × 4 |
| GPUs | 8 (69,900 steps = 5 epochs only at a global batch of 384) |
| Global batch | 384 |
| Epochs | 5 (69,900 optimizer steps) |
| Optimizer | AdamW, betas (0.9, 0.95), lr 1e-4, weight decay 0.01 |
| Schedule | cosine; warmup 5% of total steps (`Trainer` default for `warmup_steps: null`) |
| Precision, clipping | bf16; config `max_grad_norm` 1.0, but not applied (below) |
| Video | 17 frames (16 actions), reference + 1 future frame, `compact_288x256` layout |
| Video augmentation | p 0.5; color jitter 0.2/0.2/0.2/0.03, gamma 0.9–1.1, noise 0.01, crop scale 0.95–1.0, rotation 5° |
| Text | Qwen3 features (3 layers concatenated), 128 slots |
| Proprio | 14-dim, packed after the text |
| Validation split | 1% of episodes |

- Gradient clipping:
  - Under DeepSpeed, `Trainer` calls `accelerator.clip_grad_norm_`. In accelerate 1.12.0 that returns the norm and leaves clipping to the DeepSpeed config.
  - The upstream ImageWAM and FastWAM DeepSpeed configs (`scripts/ds_configs/ds_zero{1,2}_config.json`) set no `gradient_clipping`. The released model was therefore most likely trained without clipping, although paper Table 8 lists clipping 1.0.
  - This project's configs set `gradient_clipping` 1.0 (commit `34bf80d`); the upstream file is kept as `ds_zero2_upstream_config.json`.
  - In this project's G1 runs, the pre-clip global norm had a median of 1.3–2.3 and exceeded 1.0 in 62–83% of steps, so the two settings train differently.
- Samples per epoch: every kept frame is one sample (`BaseLeRobotDataset.__len__`, `sample_index_stride` 1). That is about 5.37 M training samples, or about 69.9k steps at a global batch of 384.
- The paper's main text says RoboTwin models are trained for 30k steps. The released model's 69,900 steps contradict this, and the released model is what is matched.
- Paper Table 10: FLUX.2 4B on RoboTwin trained for 5 days, effective per-GPU batch 48.
- The repository's FLUX.2-9B RoboTwin configs (`robotwin_flux2_klein_9b_base_imagewam.yaml` and `_clean_`) use 4 × 2 per GPU, a global batch of 64 on 8 GPUs. That is not the released recipe.

## Evaluation Protocol

- Paper: 50 tasks, clean and randomized settings, 100 trials per task.
  - The paper text states "100 trials per task" for RoboTwin. Table 12's FLUX.2-4B column has odd percentages (99, 67, 81), which 50 trials cannot produce; its OmniGen2 column is marked "50 trials".
  - The paper's "50 trials per task" refers to its real-world tasks.
- Repository launcher `scripts/flux2/run_eval_flux2_robotwin.sh`:
  - Defaults: 50 episodes per task, `action_horizon` 16, `replan_steps` 16, phases clean and random.
  - It always composes the `_clean_` task config; its model settings equal the non-clean config.
- Denoising steps: 10. The authors confirm in ImageWAM issue #7 that all RoboTwin and LIBERO evaluations use 10 steps.
- Instructions: `instruction_type: unseen`. Each episode draws one of the task's unseen templates.
- Seeds: the evaluator starts at seed `100000 × (1 + seed)` with `seed` 42. It keeps only seeds where the expert planner succeeds.
- Simulator: SAPIEN with Vulkan. `vulkaninfo` on `nnmc71` lists the H100 GPUs.
  - The upstream installer `scripts/setup/install_robotwin_env.sh` installs into the training `.venv`, so the evaluation environment must be separate.
  - The aloha-agilex embodiment config sets `planner: "curobo"`, so the expert check that selects the evaluation seeds needs cuRobo.
- Instruction draw: after `setup_demo(seed)` the loop fills templates with Python `random` and picks one with `np.random.choice`.
  - `setup_demo` reseeds numpy and torch but not Python `random` (`envs/_base_task.py` comments out `random.seed`).
  - An episode's instruction therefore depends on the process's earlier calls to `random`, not only on its seed.
- Text encoding: the FLUX.2 policy encodes each instruction online with its own Qwen3 encoder (`model.load_text_encoder=true`).
- This project's pipeline (`launch/robotwin_eval.sh`, `.venv-robotwin`, PROJECT.md):
  - Checked on 2026-10-08 with the released FLUX.2-4B: `beat_block_hammer`, clean, 2 episodes on one `nnmc72` H100. Both succeeded.
  - The checkpoint loaded with 0 missing and 0 unexpected keys. Seeds were 4,300,000 and 4,300,001.
  - The worker config had seed 42, `instruction_type: unseen`, 10 denoising steps, horizon 16 and replan 16.
  - Run time: 2 min 14 s including model loading.

## Storage

| Text cache | Encoder | Per prompt | 921,032 prompts |
|---|---|---|---|
| FLUX.2-4B | Qwen3-4B, 3 layers × 2560, 128 slots, bf16 | 1.97 MB | 1.8 TB |
| FLUX.2-9B | Qwen3-8B, 3 layers × 4096, 128 slots, bf16 | 3.1 MB | 2.9 TB |
| Qwen-Image-2.1 | Qwen3-VL, 4096, 128 slots, bf16 | 1.0 MB | 0.97 TB |

Free disk on 2026-10-08: `nnmc71` 2.7 TB, `nnmc72` 5.1 TB, `nnmc73` 22 TB.
