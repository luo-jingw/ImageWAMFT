# Plan

Plan Status: approved

# Problem

## Current

- Data: the four LGG100 roots (drawer, biomanual, simple-pick, Stack-the-cubes) are local, carry project task strings, and passed a data-quality audit (`docs/lgg100_dataset.md`).
  - 11 episodes are invalid (ISSUE-002).
  - `BaseLerobotDataset._filter_episode_indices` (`ImageWAM/src/imagewam/datasets/lerobot/base_lerobot_dataset.py:312-336`) supports only `periodic_prefix`, applied identically to every root. It cannot express a per-root exclusion list.
- FLUX.2 path: `ImageWAM/` (vendored upstream `5d4a341`) contains the FLUX.2-klein ImageWAM integration, with these gaps for G1:
  - The FLUX.2 source package (`black-forest-labs/flux2` at `50fe5162777813d869182b139e83b10743caef15`) is absent; `ImageWAM/src/imagewam/models/backbones/flux2_imports.py:7-23` cannot import it.
  - `ImageWAM/scripts/flux2/run_train_flux2_klein_imagewam.sh` accepts only `TASK_TYPE` `libero` or `robotwin`.
  - No data or task config exists for 3-camera, 16-dim G1 data.
  - `ImageWAM/scripts/ds_configs/ds_zero{1,2}_config.json` define no `gradient_clipping`. Under DeepSpeed, `Trainer`'s `max_grad_norm` call (`ImageWAM/src/imagewam/trainer.py:1108`) is not the effective clipping mechanism, so the paper's clip value of 1.0 is not guaranteed.
- Qwen-Image-2.1 path:
  - ImageWAM has no Qwen-Image-2.1 backbone.
  - The reference implementation (`transformer_qwenimage21.py`, `autoencoder_kl_qwenimage21.py`) exists only in diffusers git main (commit `6256aa7` or later). That needs `huggingface-hub>=1.32` and transformers 5.x, which is incompatible with `.venv` (transformers 4.56.1, huggingface-hub 0.36.2).
- Base weights are under `BaseModel/` (`PROJECT.md` → Project-Specific Constraints).

## Problem

Neither target model can be trained on LGG100 with the paper's real-world recipe (`docs/imagewam_paper_recipe.md`):
- The FLUX.2-klein-base-9B path lacks G1 data plumbing, episode exclusion, the FLUX.2 source and effective gradient clipping.
- The Qwen-Image-2.1 path lacks the whole backbone integration.

## Goal

1. A FLUX.2-klein-base-4B ImageWAM model (the backbone of the paper's real-world experiments) trained under these conditions:
   - Data: the 339 valid episodes (181,403 frames) of the four LGG100 roots, jointly.
   - Length: 10 epochs, 27,950 optimizer steps at effective batch 64 (4 per GPU × 2 GPUs × gradient accumulation 8).
   - Recipe: bf16, AdamW betas (0.9, 0.95), lr 1e-4, weight decay 1e-2, 5% linear warmup then cosine to 0.01× lr, effective gradient clipping 1.0, DeepSpeed ZeRO-2.
   - Trainable parts follow the paper (§3.2) and repository: the diffusion branch (the whole editing DiT, including its text and image input projections) and the action expert, plus the proprio encoder. The text encoder and the VAE are frozen.
   - Inputs and targets: 3-view 288×256 input, future frame 16 steps ahead, action chunk 16, λ_video 0.5, λ_action 1.0, train shift 5.
2. A Qwen-Image-2.1 ImageWAM model trained with the same data, split, recipe, length, trainable-part policy and validation. Only the backbone-bound components differ: the editing DiT, its text-encoder features, its VAE, and the action-DiT depth, which pairs layer by layer with the backbone.
3. Outputs per run, under `checkpoints/<run_name>/` at the project root:
   - weights every 5,000 steps (all kept) and only the latest resumable training state;
   - `dataset_stats.json`;
   - offline validation metrics on the held-out episodes (one per root), logged during training: `val_loss`, `action_l2`, `action_l1` on de-normalized actions, `psnr_*`, `ssim_*`;
   - the measured step time and peak GPU memory.
4. The Qwen-Image-2.1 transformer and VAE implementations used in training reproduce the diffusers reference outputs on shared inputs. The per-layer and final maximum absolute differences are reported.
5. Each full run is launched in its own `tmux` session on its training node once that model's smoke test passes; launch is authorized. Logging is local only: offline wandb under the run directory plus the launcher log.

Out of scope: physical-robot deployment and evaluation (no G1 policy adapter exists; `experiments/robotwin/imagewam_policy/deploy_policy.py` is the later template), and the FLUX.2-klein-base-4B run.

Constraint check against this goal:
- Raw-data read-only (`PROJECT.md`): satisfied, because exclusion is a loader filter and data files stay unchanged.
- `transformers==4.56.1` in `.venv` for FLUX.2: kept, because the Qwen-Image-2.1 training code avoids the diffusers-main stack (Structure → Decisions D3).
- Text context length: the launcher's `qwen_context_len` of 128 covers the G1 prompts. The longest is the drawer instruction wrapped in `DEFAULT_PROMPT`, under 100 tokens, to be verified in Phase 3.

# Structure

## Modules

| Id | Module | Responsibility |
|---|---|---|
| M1 | Episode selection | Per-root explicit episode exclusion in the LeRobot loader |
| M2 | G1 data spec | Camera order (`cam_left_high`, `cam_left_wrist`, `cam_right_wrist`), 16-dim action and state, 288×256 RoboTwin-style tiling, episode-level 1% validation split, exclusion list |
| M3 | FLUX.2 G1 run spec | Task config for FLUX.2-klein-base-9B on G1, and a `g1` branch in the FLUX.2 launcher (action dim 16, proprio dim 16, four dataset roots, Qwen3-8B text cache, no non-idle filter) |
| M4 | Recipe alignment | Effective gradient clipping 1.0 under DeepSpeed |
| M5 | FLUX.2 source | Vendored `flux2` package at the pinned commit, plus local path configuration (`ImageWAM/.env.local`) |
| M6 | Qwen-Image reference environment | `.venv-qwenimage`: diffusers main at a pinned commit, transformers 5.x, huggingface-hub ≥ 1.32. Used only for reference fixtures and Qwen3-VL text-feature precompute |
| M7 | Qwen-Image video expert | Self-contained Qwen-Image-2.1 DiT loaded from `BaseModel/Qwen-Image-2.1/transformer` and exposing the ImageWAM video-expert contract |
| M8 | Qwen-Image VAE | Vendored Apache-2.0 `AutoencoderKLQwenImage21` as a plain `nn.Module`: RGBA input, per-channel latent normalization |
| M9 | Qwen-Image action DiT | Single-stream action expert (residual width 1024, attention width 4096, 32 blocks) and its offline weight initialization from M7 weights |
| M10 | Qwen-Image stack | ImageWAM-level logic for the Qwen-Image backbone: input building, attention mask, loss, action and image inference |
| M11 | Backbone dispatch | Hooks in the shared ImageWAM files that route the `qwenimage21` stack and protocol to M7–M10 |
| M12 | Qwen-Image text cache | Offline text-only Qwen3-VL features per prompt, in a cache format the dataset loads |
| M13 | Qwen-Image G1 run spec | Model config, task config and launcher for Qwen-Image-2.1 on G1 |
| M14 | Parity checks | Reference fixture dump (in M6) and comparison against M7 and M8 (in `.venv`) |
| M15 | Results record | Measured training and validation results of both runs in `docs/` |

## Decisions

- **D1 Text conditioning: text-only cached features (option A2).** ImageWAM caches text-only features per prompt for FLUX.2 (Qwen3) and for OmniGen2 (Qwen2.5-VL). For OmniGen2 the native multimodal encoder also sees the reference image, but ImageWAM gives it the instruction only. Qwen-Image-2.1 follows that convention:
  - The Qwen3-VL template has the system message dropped and no image segment: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`.
  - This template equals the reference pipeline's text-to-image template (`prompt_template_t2i`, diffusers `578c9b2` `pipelines/qwenimage21/pipeline_qwenimage21.py:211-215`) after the system tokens are dropped. The DiT was therefore trained on text-only Qwen3-VL features in text-to-image mode.
  - The features are the last decoder layer before the final norm, matching the reference pipeline.
  - The prompt is the dataset's `DEFAULT_PROMPT`-wrapped task string.
- **D2 Sequence layout and mask: the ImageWAM convention (option B2).**
  - Joint sequence: `[text | reference image | target image | action]`.
  - Text and reference rows attend to text and reference.
  - Target rows attend to text, reference and target.
  - Action rows attend to text, reference and action, but not the noisy target.
  - No row attends to action. Padded text columns are masked.
  - This replaces Qwen-Image-2.1's native token-causal text mask and its image-before-instruction order.
- **D3 Environment: a single training environment (option C2).**
  - Training for both backbones runs in `.venv`.
  - The Qwen-Image-2.1 DiT is reimplemented in M7, without a diffusers dependency.
  - The VAE source is vendored in M8.
  - `.venv-qwenimage` (M6) is used only offline, for fixtures and text features.
- **D4 Prefix timestep: t = 0.** Context tokens (text, reference, proprio) are modulated with timestep 0 in both training and inference, and target tokens with the sampled video timestep. Qwen-Image-2.1's `causal_condition` does the same.
  - ImageWAM's FLUX.2 training modulates the prefix with the target timestep, because FLUX.2 has one modulation vector per stream, while its inference prefix cache uses timestep 0 (`imagewam.py:3562`).
  - ImageWAM has no backbone-independent rule for this choice.
- **D5 Positions.**
  - Text tokens use Qwen-Image-2.1's shared counter: axes (p, p, p).
  - The reference and target images are separate image blocks: the frame axis is the counter, height and width are centered, and the counter advances by max(h, w).
  - Action tokens continue the counter after the target block. Their positions are the same whether or not the target block is present, so the inference prefix cache stays consistent with training.
- **D6 Noise schedule and loss: the ImageWAM schedulers** (`ImageWAM/src/imagewam/models/backbones/schedulers/scheduler_continuous.py`, train shift 5) and the ImageWAM loss weighting, identical to the FLUX.2 run.
- **D7 Episode exclusion: a new filter mode `explicit_exclude`.** Its configuration maps a dataset root directory name to a list of excluded episode indices. It is applied before the validation split.
- **D8 Normalization statistics** are computed by `RobotVideoDataset` on the training split of the FLUX.2 run. The Qwen-Image-2.1 run reads the same file through `pretrained_norm_stats`, so both runs normalize identically.
- **D10 Trainable parts** follow the paper's frozen/trainable split: frozen text encoder (precomputed cache) and VAE; full fine-tuning of the editing DiT, the action DiT and the proprio encoder. LoRA stays disabled.
- **D11 Compute.** Each model trains on 2 idle GPUs from nodes `nnmc71`–`nnmc75` (at most 4 per node). If a model does not fit at 4 samples per GPU, the micro-batch drops to 2 and gradient accumulation rises to 16, so the effective batch stays 64.
- **D12 DeepSpeed configuration** changes only as concluded by ISSUE-008, applied identically to both runs.
- **D9 Code placement.** New Qwen-Image-2.1 logic lives in dedicated files (M7–M10, M12). The shared ImageWAM files (`imagewam.py`, `mot.py`, `runtime.py`, `trainer.py`, `robot_video_dataset.py`) receive only dispatch hooks.

## Responsibilities

- Data selection and the validation split: M1 and M2, inside the existing `BaseLerobotDataset` / `RobotVideoDataset` pipeline.
- The model forward pass for each backbone: M7 (video expert), M9 (action expert), M10 (stack logic), joined by the existing MoT joint attention in `mot.py`.
- Optimization, schedule, checkpointing and validation: the existing `Trainer` (`ImageWAM/src/imagewam/trainer.py`), configured by M3, M4 and M13.
- Reference correctness of the Qwen-Image-2.1 reimplementation: M14.

## State Ownership

| State | Owner (writer) | Readers |
|---|---|---|
| Excluded episode list | `ImageWAM/configs/data/g1_lgg100.yaml` (`episode_index_filter`) | M1 at dataset construction |
| Normalization statistics | FLUX.2 run output `ImageWAM/runs/g1_lgg100_flux2_klein_9b_base_imagewam/<run_id>/dataset_stats.json`, written by `RobotVideoDataset` | FLUX.2 run, Qwen-Image-2.1 run via `pretrained_norm_stats` |
| FLUX.2 text cache | `ImageWAM/scripts/flux2/precompute_flux2_qwen3_embeds.py` → `ImageWAM/data/g1_lgg100/flux2_qwen3_cache_9b/` | `RobotVideoDataset` (format `qwen3_flux2`) |
| Qwen-Image text cache | `ImageWAM/scripts/qwenimage21/precompute_qwenimage21_text_embeds.py` → `ImageWAM/data/g1_lgg100/qwenimage21_text_cache/` | `RobotVideoDataset` (format `qwen3vl_qwenimage21`) |
| FLUX.2 action-DiT initialization | `ImageWAM/scripts/flux2/preprocess_action_dit_flux2.py` → `ImageWAM/checkpoints/action_dit_flux2_9b_g1_init.pt` | `ActionDiTFlux2.from_pretrained` |
| Qwen-Image action-DiT initialization | `ImageWAM/scripts/qwenimage21/preprocess_action_dit_qwenimage21.py` → `ImageWAM/checkpoints/action_dit_qwenimage21_g1_init.pt` | `ActionDiTQwenImage21.from_pretrained` |
| Reference fixtures | `ImageWAM/scripts/qwenimage21/dump_reference_fixtures.py` (in `.venv-qwenimage`) → `ImageWAM/data/qwenimage21_fixtures/` | M14 comparison scripts |
| Trained weights and training state | `Trainer` → `checkpoints/<run_name>/checkpoints/{weights,state}/` (project root, git-ignored) | later deployment |
| Local paths (FLUX.2 source, weights, data root) | `ImageWAM/.env.local`, git-ignored, contents documented in `PROJECT.md` | `ImageWAM/scripts/common.sh` |

# Interface

## Interfaces

- **M1** `BaseLerobotDataset._filter_episode_indices(self, episode_indices: list[int], repo_id: str) -> list[int]`. New mode `explicit_exclude`:
  - Configuration `{mode: explicit_exclude, exclude: {<root dir name>: [int, ...]}}`.
  - It returns `episode_indices` minus the list stored under `Path(repo_id).name`.
  - An unknown root name in `exclude` raises `ValueError`.
- **M2** `configs/data/g1_lgg100.yaml`:
  - `train` and `val` nodes of `RobotVideoDataset`, with `shape_meta.images` keys `[cam_left_high, cam_left_wrist, cam_right_wrist]`, each `raw_shape [3,480,640]` and `shape [3,240,320]`;
  - `action`/`state` `raw_shape` and `shape` 16; `processor.action_output_dim` and `proprio_output_dim` 16;
  - `num_frames: 17`, `video_size: [288, 256]`, `concat_multi_camera: robotwin`, `robotwin_camera_layout: compact_288x256`, `endpoint_frames_only: true`;
  - `val_set_proportion: 0.01`, `val_split_level: episode`;
  - `nonidle_filter_path: null`, `pretrained_norm_stats: null`;
  - `episode_index_filter` with the ISSUE-002 exclusions.
- **M3**
  - `configs/task/g1_lgg100_flux2_klein_9b_base_imagewam.yaml`: overrides `/data: g1_lgg100` and `/model: imagewam_flux2_klein_9b_base`; sets the Goal 1 hyperparameters; `num_epochs: 10`, `max_steps: null`.
  - The launcher `TASK_TYPE=g1` branch sets:
    - `ACTION_DIM=16`;
    - `data.{train,val}.dataset_dirs` to the four roots;
    - `qwen_text_cache_dir`;
    - `qwen_context_len=128`;
    - `model.proprio_dim=16`.
- **M4** `scripts/ds_configs/ds_zero{1,2}_config.json`: `"gradient_clipping": 1.0`.
- **M7** `class QwenImage21VideoExpert(nn.Module)` in `qwenimage21_video_expert.py`.
  - Attributes: `block_protocol = "qwenimage21"`, `num_heads = 32`, `num_kv_heads = 32`, `attn_head_dim = 128`, `hidden_dim = 4096`, `blocks: nn.ModuleList` (32).
  - `from_pretrained(transformer_dir: Path, device: torch.device, torch_dtype: torch.dtype) -> QwenImage21VideoExpert` loads the safetensors with strict key matching.
  - `build_positions(text_len: int, ref_hw: tuple[int, int], target_hw: tuple[int, int], action_len: int) -> QwenImage21Positions`, a dataclass of rotary tables for the text, reference, target and action segments (D5).
  - `pre_dit(target_latents: Tensor | None, video_timestep: Tensor, text_states: Tensor, text_mask: Tensor, ref_latents: Tensor, positions: QwenImage21Positions) -> QwenImage21PreState`, a dataclass with tokens, rotary, per-token modulation (prefix t = 0, target t) and segment lengths. It accepts `target_latents=None` (prefix only).
  - `QwenImage21Block.prepare_qkv(x: Tensor, rotary: Tensor, modulation: QwenImage21Modulation) -> QwenImage21QKV`, with q, k, v of shape `[B, L, 4096]` after RMS QK-norm and RoPE, plus residual state.
  - `QwenImage21Block.apply_post(attn_out: Tensor, state: QwenImage21QKV) -> Tensor`.
  - `post_dit(x: Tensor, pre: QwenImage21PreState) -> Tensor` returns target velocity `[B, N_target, 64]`.
- **M8** `class QwenImage21VAE(nn.Module)` in `qwenimage21_vae.py`:
  - `from_pretrained(vae_dir: Path, device: torch.device, torch_dtype: torch.dtype) -> QwenImage21VAE`;
  - `encode(rgb: Tensor[B,3,H,W] in [-1,1]) -> Tensor[B,64,H/16,W/16]`, which appends an opaque alpha channel and normalizes with `latents_mean`/`latents_std`;
  - `decode(latents: Tensor) -> Tensor[B,3,H,W]`, which de-normalizes and drops alpha.
- **M9**
  - `class ActionDiTQwenImage21(nn.Module)` in `action_dit_qwenimage21.py`: `block_protocol = "qwenimage21"`, 32 blocks with `prepare_qkv` / `apply_post` signatures matching M7.
    - `from_pretrained(config: ActionDiTQwenImage21Config, pretrained_path: Path | None, device: torch.device, torch_dtype: torch.dtype) -> ActionDiTQwenImage21`;
    - `pre_dit(actions: Tensor[B,T,16], timestep: Tensor[B]) -> ActionPreState`;
    - `post_dit(x: Tensor, pre: ActionPreState) -> Tensor[B,T,16]`.
  - The script `preprocess_action_dit_qwenimage21.py --transformer-dir --action-dim 16 --output` writes `{"state_dict", "meta"}`. Block i copies from M7 block i, and widths are interpolated with alpha scaling, as in `preprocess_action_dit_flux2.py`.
- **M10** `class QwenImage21Stack` in `qwenimage21_stack.py`, holding references to M7, M8 and M9. It provides:
  - `build_inputs(sample: Mapping[str, Tensor]) -> QwenImage21Inputs`;
  - `attention_mask(inputs: QwenImage21Inputs) -> Tensor[B,1,L,L]`, which follows D2;
  - `training_loss(inputs: QwenImage21Inputs) -> QwenImage21Loss`, with video loss, action loss and total, following D6;
  - `infer_action(inputs: QwenImage21Inputs, num_steps: int) -> Tensor[B,T,16]`, using the prefix KV cache;
  - `infer_image(inputs: QwenImage21Inputs, num_steps: int) -> Tensor[B,3,H,W]`.
- **M11** dispatch hooks:
  - `ImageWAM.from_qwenimage21_pretrained(...)`; `stack == "qwenimage21"` routes `training_loss`, `infer_action`, `infer` and `load_checkpoint` to M10.
  - In `mot.py`, `_build_expert_attention_io` / `_apply_expert_post_block` add `block_protocol == "qwenimage21"` using the generic MoT loop and cache path. That path applies `use_gradient_checkpointing` per block.
  - `runtime.create_imagewam_qwenimage21(cfg) -> ImageWAM`.
  - `trainer.py` adds image-stack flags and VAE encode/decode routing for validation metrics.
- **M12**
  - Script `precompute_qwenimage21_text_embeds.py task=<task>` (run in `.venv-qwenimage`) writes `<sha256(prompt)>.qwen3vl_qwenimage21_len<L>.pt` containing `text_hidden_states [L, 4096]` (bf16) and `text_attention_mask [L]` (bool).
  - `RobotVideoDataset` accepts `qwen_text_cache_format: qwen3vl_qwenimage21`.
- **M14**
  - `dump_reference_fixtures.py --out <dir>` (in `.venv-qwenimage`) writes the fixed-seed inputs and the reference outputs of the diffusers transformer and VAE: per-block hidden states, final output, VAE latents and reconstruction.
  - `check_parity_dit.py --fixtures <dir>` and `check_parity_vae.py --fixtures <dir>` (in `.venv`) print the per-layer and final maximum absolute and relative differences.

## Inputs

- Data: the four roots in `dataset/LGG100/`.
- Weights:
  - `BaseModel/flux2/FLUX.2-klein-base-9B/flux-2-klein-base-9b.safetensors`
  - `BaseModel/flux2/FLUX.2-dev/ae.safetensors`
  - `BaseModel/Qwen3-8B/`
  - `BaseModel/Qwen-Image-2.1/{transformer,vae,text_encoder,processor}/`
- Configs: the M2, M3 and M13 files above.

## Outputs

- Per run: `ImageWAM/runs/<task>/<run_id>/{config.yaml, dataset_stats.json, checkpoints/weights/step_*.pt, checkpoints/state/, eval/}`, plus logged training and validation metrics.
- Parity reports (M14) and the results record (M15) in `docs/`.

## State Changes

- The data loader's episode set changes from all 350 episodes to 339 (M1, M2).
- The DeepSpeed configuration gains active gradient clipping (M4). This also affects the existing LIBERO and RoboTwin recipes, which become clipped at 1.0 under DeepSpeed, as the paper specifies.
- `ImageWAM/third_party/flux2/` becomes tracked source (M5). `ImageWAM/.gitignore` changes accordingly.

# Flow

## Main Flow

FLUX.2-klein-base-9B:
1. `ImageWAM/.env.local` provides `FLUX2_SRC=ImageWAM/third_party/flux2`, `FLUX2_MODEL_PATH`, `FLUX2_AE_MODEL_PATH`, `FLUX2_QWEN3_MODEL_SPEC=BaseModel/Qwen3-8B`, `FLUX2_VARIANT=9b`, `DATA_ROOT` and `ZERO_STAGE=2`.
2. `TASK_TYPE=g1 bash scripts/flux2/run_train_flux2_klein_imagewam.sh`:
   - `preprocess_action_dit_flux2.py --variant klein-base-9b --action-dim 16` → action-DiT initialization;
   - `precompute_flux2_qwen3_embeds.py` → 4 prompt caches;
   - `train_zero2.sh` → `accelerate launch scripts/train.py task=g1_lgg100_flux2_klein_9b_base_imagewam`.
3. `run_training`:
   - Hydra composes `train.yaml` < `data/g1_lgg100` < `model/imagewam_flux2_klein_9b_base` < task < CLI.
   - `RobotVideoDataset` loads the four roots and applies `explicit_exclude` and the episode-level 1% split.
   - It computes `dataset_stats.json`.
   - Each sample is built as follows:
     - images: frames t and t+16 of the three cameras are tiled to 288×256;
     - actions: 16 normalized 16-dim actions; proprio: the state at t;
     - text: the cached Qwen3 features (128 × 12288).
4. `Trainer`: the forward pass runs through `ImageWAM._training_loss_flux2` and MoT `_forward_flux2`, followed by AdamW (clip 1.0 via DeepSpeed) and the cosine schedule. It writes checkpoints every 2,500 steps and validates every 500 steps on 4 held-out episodes.

Qwen-Image-2.1:
1. In `.venv-qwenimage`: `dump_reference_fixtures.py` produces fixtures, and `precompute_qwenimage21_text_embeds.py` produces 4 prompt caches (128 × 4096).
2. In `.venv`: `check_parity_dit.py` and `check_parity_vae.py` compare M7 and M8 against the fixtures.
3. `preprocess_action_dit_qwenimage21.py` → action-DiT initialization.
4. `bash scripts/qwenimage21/run_train_qwenimage21_imagewam.sh` → `accelerate launch scripts/train.py task=g1_lgg100_qwenimage21_imagewam`, with `pretrained_norm_stats` pointing to the FLUX.2 run's `dataset_stats.json`.
5. `run_training`, with the same data pipeline. The model step is:
   - `QwenImage21Stack.build_inputs` (M8 encodes the reference and target; cached text states and the proprio token form the prefix);
   - `attention_mask` (D2);
   - `QwenImage21VideoExpert.pre_dit` and `ActionDiTQwenImage21.pre_dit`;
   - the MoT generic loop over 32 paired blocks (`prepare_qkv`, joint SDPA, `apply_post`) with per-block checkpointing;
   - `post_dit` for both experts;
   - `training_loss`;
   - the `Trainer` step, identical to the FLUX.2 run.
6. Validation: `QwenImage21Stack.infer_action` prefills the prefix KV cache once at t = 0 and then runs action denoising steps only. `infer_image` produces the image metrics.

# Code Mapping

## Modules

All paths are relative to the project root.

| Module | Files |
|---|---|
| M1 | `ImageWAM/src/imagewam/datasets/lerobot/base_lerobot_dataset.py` |
| M2 | `ImageWAM/configs/data/g1_lgg100.yaml` (new) |
| M3 | `ImageWAM/configs/task/g1_lgg100_flux2_klein_9b_base_imagewam.yaml` (new), `ImageWAM/scripts/flux2/run_train_flux2_klein_imagewam.sh` |
| M4 | `ImageWAM/scripts/ds_configs/ds_zero1_config.json`, `ImageWAM/scripts/ds_configs/ds_zero2_config.json` |
| M5 | `ImageWAM/third_party/flux2/` (new, vendored), `ImageWAM/.gitignore`, `ImageWAM/.env.local` (git-ignored), `PROJECT.md` |
| M6 | `.venv-qwenimage/` (git-ignored), `PROJECT.md` |
| M7 | `ImageWAM/src/imagewam/models/backbones/qwenimage21_video_expert.py` (new) |
| M8 | `ImageWAM/src/imagewam/models/backbones/qwenimage21_vae.py` (new, Apache-2.0 header and diffusers commit reference) |
| M9 | `ImageWAM/src/imagewam/models/backbones/action_dit_qwenimage21.py` (new), `ImageWAM/scripts/qwenimage21/preprocess_action_dit_qwenimage21.py` (new) |
| M10 | `ImageWAM/src/imagewam/models/backbones/qwenimage21_stack.py` (new) |
| M11 | `ImageWAM/src/imagewam/models/backbones/imagewam.py`, `ImageWAM/src/imagewam/models/backbones/mot.py`, `ImageWAM/src/imagewam/runtime.py`, `ImageWAM/src/imagewam/trainer.py` |
| M12 | `ImageWAM/scripts/qwenimage21/precompute_qwenimage21_text_embeds.py` (new), `ImageWAM/src/imagewam/datasets/lerobot/robot_video_dataset.py` |
| M13 | `ImageWAM/configs/model/imagewam_qwenimage21.yaml` (new), `ImageWAM/configs/task/g1_lgg100_qwenimage21_imagewam.yaml` (new), `ImageWAM/scripts/qwenimage21/run_train_qwenimage21_imagewam.sh` (new) |
| M14 | `ImageWAM/scripts/qwenimage21/dump_reference_fixtures.py` (new), `ImageWAM/scripts/qwenimage21/check_parity_dit.py` (new), `ImageWAM/scripts/qwenimage21/check_parity_vae.py` (new) |
| M15 | `docs/g1_training_results.md` (new), `docs/qwenimage21_parity.md` (new) |

## Interfaces

| Interface | File |
|---|---|
| `_filter_episode_indices` (`explicit_exclude`) | `ImageWAM/src/imagewam/datasets/lerobot/base_lerobot_dataset.py` |
| `QwenImage21VideoExpert`, `QwenImage21Block`, `QwenImage21PreState`, `QwenImage21Positions`, `QwenImage21Modulation`, `QwenImage21QKV` | `ImageWAM/src/imagewam/models/backbones/qwenimage21_video_expert.py` |
| `QwenImage21VAE` | `ImageWAM/src/imagewam/models/backbones/qwenimage21_vae.py` |
| `ActionDiTQwenImage21`, `ActionDiTQwenImage21Config`, `ActionPreState` | `ImageWAM/src/imagewam/models/backbones/action_dit_qwenimage21.py` |
| `QwenImage21Stack`, `QwenImage21Inputs`, `QwenImage21Loss` | `ImageWAM/src/imagewam/models/backbones/qwenimage21_stack.py` |
| `ImageWAM.from_qwenimage21_pretrained`, stack dispatch | `ImageWAM/src/imagewam/models/backbones/imagewam.py` |
| MoT `qwenimage21` protocol hooks | `ImageWAM/src/imagewam/models/backbones/mot.py` |
| `create_imagewam_qwenimage21` | `ImageWAM/src/imagewam/runtime.py` |
| `qwen3vl_qwenimage21` cache format | `ImageWAM/src/imagewam/datasets/lerobot/robot_video_dataset.py` |

## State

| State | Owner file |
|---|---|
| Excluded episode list | `ImageWAM/configs/data/g1_lgg100.yaml` |
| Normalization statistics | written by `ImageWAM/src/imagewam/datasets/lerobot/robot_video_dataset.py` into the FLUX.2 run directory |
| Text caches | `ImageWAM/scripts/flux2/precompute_flux2_qwen3_embeds.py`, `ImageWAM/scripts/qwenimage21/precompute_qwenimage21_text_embeds.py` |
| Action-DiT initializations | `ImageWAM/scripts/flux2/preprocess_action_dit_flux2.py`, `ImageWAM/scripts/qwenimage21/preprocess_action_dit_qwenimage21.py` |
| Reference fixtures | `ImageWAM/scripts/qwenimage21/dump_reference_fixtures.py` |
| Run outputs | `ImageWAM/src/imagewam/trainer.py` |

# Implementation


## Phase 1

Phase Status: completed

Round: 1

### Goal

Exclude the ISSUE-002 episodes per root at load time (M1).

### Files

`ImageWAM/src/imagewam/datasets/lerobot/base_lerobot_dataset.py`

### Structures

`explicit_exclude` mode in `_filter_episode_indices`

### Affected Modules

M1

### Dependencies

none

### Observation

Construct the training and validation datasets over the four roots with the exclusion list, and print per-root episode counts and frame totals before and after the split. Expected: 100 / 49 / 91 / 99 episodes and 181,403 frames before the split, with no excluded index present. An unknown root name in `exclude` raises `ValueError`.

## Phase 2

Phase Status: completed

Round: 1

### Goal

Vendor the FLUX.2 source at `50fe5162777813d869182b139e83b10743caef15` and configure local paths (M5).

### Files

`ImageWAM/third_party/flux2/`, `ImageWAM/.gitignore`, `ImageWAM/.env.local`, `PROJECT.md`

### Structures

none

### Affected Modules

M5

### Dependencies

none

### Observation

In `.venv`, check two groups of symbols:
- import `Flux2`, `Klein9BParams`, `timestep_embedding`, `apply_rope`, `QKNorm`, `SiLUActivation`, `MLPEmbedder` and `Modulation` from `flux2.model`, `AutoEncoder` from `flux2.autoencoder`, and `OUTPUT_LAYERS_QWEN3` from `flux2.text_encoder`;
- confirm that the block methods `_prepare_qkv`, `_apply_residuals`, `_qkv` and `_out` exist.

Then build `Flux2VideoExpert.from_pretrained` for 9B on the meta device. Print the missing and unexpected keys; both are expected to be empty.

## Phase 3

Phase Status: completed

Round: 2

### Goal

Specify the G1 data and the FLUX.2-9B G1 run (M2, M3).

### Files

`ImageWAM/configs/data/g1_lgg100.yaml`, `ImageWAM/configs/task/g1_lgg100_flux2_klein_9b_base_imagewam.yaml`, `ImageWAM/scripts/flux2/run_train_flux2_klein_imagewam.sh`

### Structures

`g1` launcher branch

### Affected Modules

M2, M3

### Dependencies

Phase 1 (exclusion mode used by the data config), Phase 2 (FLUX.2 import for the text-cache script)

### Observation

1. `DRY_RUN=true TASK_TYPE=g1` prints the three launcher commands with action dim 16, the four roots and `qwen_context_len=128`.
2. Run the text-cache precompute. It writes 4 files of `[128, 12288]`; the longest prompt's token count is printed and must be ≤ 128.
3. Construct one training sample and print the shapes and dtypes of the video tensor, the 16×16 action, the proprio and the text states.
4. Save the tiled 288×256 image to check the camera order.

## Phase 4

Phase Status: completed

Round: 1

### Goal

Make gradient clipping at 1.0 effective under DeepSpeed (M4).

### Files

`ImageWAM/scripts/ds_configs/ds_zero1_config.json`, `ImageWAM/scripts/ds_configs/ds_zero2_config.json`

### Structures

none

### Affected Modules

M4

### Dependencies

none

### Observation

In the Phase 5 run, print DeepSpeed's resolved `gradient_clipping` value from the engine configuration. Also log the global gradient norm for the first steps.

## Phase 5

Phase Status: completed

Round: 3

### Goal

Run a FLUX.2-klein-base-4B G1 smoke test on 2 GPUs, resolving ISSUE-008 (non-finite gradients under DeepSpeed ZeRO-2) first.

### Files

`ImageWAM/configs/task/g1_lgg100_flux2_klein_4b_base_imagewam.yaml` (new), `ImageWAM/.env.local` (git-ignored); any DeepSpeed or trainer change concluded by ISSUE-008

### Structures

none

### Affected Modules

M2–M5

### Dependencies

Phases 1, 2, 3, 4

### Observation

1. Report the ISSUE-008 experiment grid.
2. Generate the 4B action-DiT initialization (action dim 16) and the Qwen3-4B text cache (4 files `[128, 7680]`).
3. Run 200 optimizer steps on 2 GPUs (4 per GPU × gradient accumulation 8) and report:
   - step time;
   - peak memory per GPU;
   - finite video and action losses at every step;
   - one validation pass;
   - projected wall-clock time for 27,950 steps.

## Phase 6

Phase Status: active

Round: 4

### Goal

Launch FLUX.2-klein-base-4B full training (10 epochs) in a `tmux` session on 2 idle GPUs.

### Files

none (runtime artifacts under `checkpoints/g1_lgg100_flux2_klein_4b_base_imagewam/`)

### Structures

none

### Affected Modules

none

### Dependencies

Phase 5

### Observation

The session name, node and GPUs are recorded. The first 500 steps have finite losses, measured throughput and a projected end time. `save_every` is 5,000, and `output_dir` is under `checkpoints/`.

## Phase 7

Phase Status: completed

Round: 1

### Goal

Set up the Qwen-Image-2.1 reference environment and dump reference fixtures (M6, the fixture half of M14).

### Files

`.venv-qwenimage/`, `PROJECT.md`, `ImageWAM/scripts/qwenimage21/dump_reference_fixtures.py`

### Structures

fixture file layout

### Affected Modules

M6, M14

### Dependencies

none

### Observation

1. In `.venv-qwenimage`, `QwenImage21Pipeline` edits one G1 tiled frame at 288×256 with a task prompt. Save the output image as a zero-shot sanity check.
2. Fixtures are written for one fixed-seed batch: transformer per-block outputs, the final output, VAE latents and reconstruction.
3. Record the pinned diffusers commit and the transformers and huggingface-hub versions in `PROJECT.md`.

## Phase 8

Phase Status: pending

Round: 2

### Goal

Reimplement the Qwen-Image-2.1 transformer as the ImageWAM video expert (M7) and check its parity.

### Files

`ImageWAM/src/imagewam/models/backbones/qwenimage21_video_expert.py`, `ImageWAM/scripts/qwenimage21/check_parity_dit.py`

### Structures

`QwenImage21VideoExpert`, `QwenImage21Block`, `QwenImage21PreState`, `QwenImage21Positions`, `QwenImage21Modulation`, `QwenImage21QKV`

### Affected Modules

M7, M14

### Dependencies

Phase 7

### Observation

1. `from_pretrained` loads with no missing or unexpected keys; the parameter count is 7,115,124,736.
2. `check_parity_dit.py` reports the per-block and final maximum absolute and relative differences against the fixtures in bf16 and fp32. The native diffusers mask and order are reproduced through a test-only layout argument.

## Phase 9

Phase Status: pending

Round: 2

### Goal

Vendor the Qwen-Image-2.1 VAE (M8) and check its parity.

### Files

`ImageWAM/src/imagewam/models/backbones/qwenimage21_vae.py`, `ImageWAM/scripts/qwenimage21/check_parity_vae.py`

### Structures

`QwenImage21VAE`

### Affected Modules

M8, M14

### Dependencies

Phase 7

### Observation

1. Loads with no missing or unexpected keys; the parameter count is 337.7M.
2. Report the maximum absolute difference of latents and reconstruction against the fixtures.
3. Report the PSNR of the reconstruction of 16 G1 tiled frames at 288×256.

## Phase 10

Phase Status: pending

Round: 3

### Goal

Build the Qwen-Image-2.1 text cache and teach the dataset to load it (M12).

### Files

`ImageWAM/scripts/qwenimage21/precompute_qwenimage21_text_embeds.py`, `ImageWAM/src/imagewam/datasets/lerobot/robot_video_dataset.py`

### Structures

cache format `qwen3vl_qwenimage21`

### Affected Modules

M12

### Dependencies

Phase 3 (G1 data config and prompts), Phase 7 (environment)

### Observation

1. 4 cache files with `text_hidden_states [128, 4096]` bf16 and a mask whose true length equals the template token count.
2. A dataset sample built with `qwen_text_cache_format: qwen3vl_qwenimage21` carries those tensors.
3. The features match a direct pipeline `encode_prompt` call on the same text-only template; report the maximum absolute difference.

## Phase 11

Phase Status: pending

Round: 3

### Goal

Build the Qwen-Image-2.1 action DiT and its weight initialization (M9).

### Files

`ImageWAM/src/imagewam/models/backbones/action_dit_qwenimage21.py`, `ImageWAM/scripts/qwenimage21/preprocess_action_dit_qwenimage21.py`

### Structures

`ActionDiTQwenImage21`, `ActionDiTQwenImage21Config`, `ActionPreState`

### Affected Modules

M9

### Dependencies

Phase 8

### Observation

1. The initialization file is written. Print the copied, interpolated and freshly initialized parameter names and the parameter count.
2. `prepare_qkv` output shapes match M7's `[B, L, 4096]`.
3. A forward pass on random actions gives finite outputs `[B, 16, 16]`.

## Phase 12

Phase Status: pending

Round: 4

### Goal

Integrate the Qwen-Image-2.1 stack into ImageWAM, MoT, runtime and trainer, and add its run specification (M10, M11, M13).

### Files

`ImageWAM/src/imagewam/models/backbones/qwenimage21_stack.py`, `ImageWAM/src/imagewam/models/backbones/imagewam.py`, `ImageWAM/src/imagewam/models/backbones/mot.py`, `ImageWAM/src/imagewam/runtime.py`, `ImageWAM/src/imagewam/trainer.py`, `ImageWAM/configs/model/imagewam_qwenimage21.yaml`, `ImageWAM/configs/task/g1_lgg100_qwenimage21_imagewam.yaml`, `ImageWAM/scripts/qwenimage21/run_train_qwenimage21_imagewam.sh`

### Structures

`QwenImage21Stack`, `QwenImage21Inputs`, `QwenImage21Loss`

### Affected Modules

M10, M11, M13

### Dependencies

Phases 8, 9, 10, 11

### Observation

1. The attention mask for a small synthetic layout is printed as a block matrix and matches D2.
2. One training batch on one GPU: loss terms are finite, every trainable parameter group has a non-zero gradient, and the VAE and text encoder receive no gradients.
3. `infer_action` returns `[1, 16, 16]`.
4. With identical inputs, `infer_action` with the prefix KV cache equals a full-sequence recomputation within bf16 tolerance; report the maximum absolute difference.

## Phase 13

Phase Status: pending

Round: 5

### Goal

Run a Qwen-Image-2.1 G1 smoke test on 2 GPUs.

### Files

none (runtime artifacts only)

### Structures

none

### Affected Modules

M9–M13

### Dependencies

Phases 5 (normalization statistics and ISSUE-008 resolution), 12

### Observation

Same report as Phase 5, with per-block checkpointing on.

## Phase 14

Phase Status: pending

Round: 6

### Goal

Launch Qwen-Image-2.1 ImageWAM full training (10 epochs) in a `tmux` session on 2 idle GPUs.

### Files

none (runtime artifacts under `checkpoints/g1_lgg100_qwenimage21_imagewam/`)

### Structures

none

### Affected Modules

none

### Dependencies

Phase 13

### Observation

Same as Phase 6.

## Phase 15

Phase Status: pending

Round: 7

### Goal

Record verified results of both runs and the parity checks (M15).

### Files

`docs/g1_training_results.md`, `docs/qwenimage21_parity.md`

### Structures

none

### Affected Modules

M15

### Dependencies

Phases 6, 8, 9, 14, and both runs finished

### Observation

One table per run covering configuration, step count, wall-clock time, peak memory, final and best `val_loss` / `action_l2` / `action_l1` / PSNR / SSIM, and checkpoint paths. A parity table covering per-block and final differences for the transformer and the VAE.
