# Random Image-Token Masking on LIBERO

## Question

How sensitive is ImageWAM's action prediction to removing a random fraction of the reference-image tokens from the prefix the action expert reads? The answer decides the next step:
- if success holds up, analyze why the image tokens are redundant;
- if success drops, look for a masking strategy that keeps accuracy with fewer tokens.

## Model and Benchmark

- Checkpoint: `yuyangalin/ImageWAM-FLUX.2-4B-LIBERO`, stored in `ImageWAM/checkpoints/imagewam_release/libero/flux2_klein_4b/` with its training `config.yaml` and `dataset_stats.json`.
- Paper LIBERO success rates for this model: Spatial 97.2, Object 99.2, Goal 98.8, Long 98.4.
- Input: two cameras (`image`, `wrist_image`) concatenated horizontally to 224×448, which gives 14 × 28 = 392 reference-image tokens.
- Prefix: 512 text positions from the online Qwen3-4B encoder (mostly padding), 1 proprio token, and the 392 image tokens.
- Action: chunks of 16 seven-dimensional actions with 10 denoising steps. 12 actions of each chunk are executed before replanning (official evaluation settings).

## Mechanism Under Test

- **Masked set**: a random subset of the 392 reference-image tokens, resampled at every inference call. The sampling seed is fixed per episode.
- **How a token is masked**: its column is disabled in the attention mask for every row, during prefix prefill and during action denoising. No token can read it, and the remaining tokens keep their positions. Mathematically this equals removing the token; it measures accuracy only, not speed.
- **Untouched**: text and proprio tokens.

## Grid

| Variable | Levels |
|---|---|
| Masked fraction of image tokens | 0, 1/3 (131 tokens), 2/3 (261 tokens) |

Fixed:
- 4 suites × 10 tasks × 5 trials = 200 episodes per level, 600 in total;
- the official initial states and episode seeds, identical across levels;
- 10 action denoising steps, replanning every 12 steps.

## Recorded Data

- **Per episode**: suite, task id, trial and initial-state index, seed, masked fraction, success, steps to success, and a rollout video of both cameras.
- **Per inference call (masked levels)**: a paired unmasked inference from the same observation and the same initial action noise. Only the masked action is executed. Recorded per call:
  - the episode step;
  - the proprio;
  - the 392-bit token mask;
  - the masked action chunk and the paired unmasked action chunk;
  - the executed actions;
  - prefill and action-denoising time.
- **Per inference call, attention summary**: action-to-prefix attention averaged over heads, action tokens and denoising steps, as [layers × prefix tokens] fp16. For the masked levels this is recorded for both the masked and the paired unmasked inference.
- **Per replanning step**: the MuJoCo simulator state, so any state can be restored for offline re-evaluation.

## Analysis

1. Success rate per masked fraction, overall and per suite and task, with binomial standard errors.
2. Paired action deviation, masked vs. unmasked, per masked fraction. Relation of the deviation to episode failure.
3. Attention profile of the unmasked model:
   - the share of attention on text, proprio and image tokens per layer;
   - concentration over image tokens (entropy, top-k mass);
   - image heatmaps over the two cameras.
4. Whether the paired action deviation is explained by the unmasked attention mass that fell on the masked tokens. If it is, attention-guided token selection is the next experiment.

## Environment

- `.venv-libero`: the `.venv` package set, plus mujoco 3.3.2, robosuite 1.4.0, bddl 1.0.1, gym 0.25.2 and LIBERO, editable from `ImageWAM/third_party/LIBERO`.
- Rendering: `MUJOCO_GL=egl`, using the driver EGL.
- LIBERO configuration: project-local, through `LIBERO_CONFIG_PATH`.
- GPUs: idle H100 on `nnmc71`.

## Stop Conditions

- The unmasked level does not reproduce the paper's success rates within the sampling error of 50 episodes per suite. Fix the evaluation pipeline before reading the masked levels.
- With 5 trials per task the levels are not separable. Report this before adding trials.
