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
- Rendering: OSMesa, the backend of the official evaluation. It uses Mesa 24.3 from the project prefix `.osmesa-libero`; Mesa 25.1 and later no longer ship OSMesa. Rendering runs on the CPU with llvmpipe at about 0.2 s per step.
  - EGL renders the same state with 60–67% of the pixels different, by 1.1–1.3 of 255 on average and up to 115 at edges (`render_probe.py`).
  - On the unmasked Spatial suite, EGL rendering gave 34/39 successes and OSMesa 50/50.
  - EGL also aborted intermittently inside robosuite `read_pixels`. The EGL episodes in `runs/20261004_random_mask_v2/` are kept as a rendering-shift condition and are not part of the grid.
- LIBERO configuration: project-local, through `LIBERO_CONFIG_PATH`.
- GPUs: idle H100 on `nnmc71`.

## Stop Conditions

- The unmasked level does not reproduce the paper's success rates within the sampling error of 50 episodes per suite. Fix the evaluation pipeline before reading the masked levels.
- With 5 trials per task the levels are not separable. Report this before adding trials.

## Strategy Comparison Grid (2026-10-04)

**Question.** At the same budget of computed reference-image tokens, does the way tokens are reduced matter?
- Remove: random vs. a fixed regular pattern.
- Merge: ToMe.
- Reuse: take the K/V of tokens that are not recomputed from earlier calls.

**Strategies** (`eval_token_mask.py`; a condition is `strategy:n` with n tokens reduced):

| Strategy | What happens to the n tokens | n |
|---|---|---|
| `random` | Removed (attention column disabled); resampled every call | 196 (131 and 261 come from the first grid) |
| `grid` | Removed; fixed pattern on the 14 × 28 grid | 131: (r+c)%3==0; 196: checkerboard (r+c)%2==0; 261: all but (r+c)%3==0 |
| `tome` | Merged away by ToMe bipartite soft matching on the VAE latent tokens | 131, 196 (one round); 261 (rounds 196 + 65) |
| `kv_random` | Not recomputed; reused from the per-episode K/V canvas; resampled every call | 131, 261 |
| `kv_topk` | Not recomputed; only the 9 tokens with the most action attention at the previous call are recomputed | 383 |
| `kv_randk` | As `kv_topk`, with 9 random tokens recomputed (control for `kv_topk`) | 383 |

Strategy details:
- **ToMe:**
  - Positions are merged by the same size-weighted average.
  - Proportional attention adds log(size) to the attention logits of merged tokens.
  - The sequence becomes shorter.
- **K/V reuse:**
  - The K/V canvas holds, per prefix layer, each image token's K/V from the latest call that computed it.
  - Reused tokens' K/V replace the fresh ones before attention in every prefix layer, so the cached prefix the action expert reads carries them too. Text and proprio are always recomputed.
  - The first call of an episode computes every token.

**Fixed**, as in the first grid:
- 200 episodes per condition: 4 suites × 10 tasks × 5 initial states;
- OSMesa rendering, seed 42;
- a paired unreduced inference at every call.

The no-reduction and random 131/261 episodes are reused from `runs/20261004_osmesa_grid/`. 11 new conditions × 200 = 2,200 episodes, in `runs/20261004_reduce_grid/`.

**Sanity checks** (`sanity_check.py`, LIBERO-Spatial task 0):
- No reduction equals the official prediction path (max diff 0).
- Removed tokens receive zero attention.
- ToMe with zero merges goes through the float-mask path and equals no reduction exactly.
- Merged sizes sum to 392.
- K/V reuse is exact on the first call and when the canvas comes from the same observation. Reused canvas entries stay unchanged.

**Analysis** (`make_figures.py`, which rebuilds `figures/` from all runs):
- success rate per condition and suite, with the McNemar test against no reduction;
- paired action deviation (K/V conditions: reuse calls only);
- K/V entry age and the deviation per call index;
- where `kv_topk` refreshes;
- what each strategy does to one frame, and the frame the K/V canvas encodes.

## Attention-Guided Pruning and Merging Variants

**Question.** In the strategy comparison grid, ToMe merging failed beyond 1/3 (39% at 1/2), while a fixed checkerboard removal of 1/2 kept 98%. Two follow-ups:
- Does selecting the kept tokens by the model's own action attention beat random and grid selection at the same budget?
- Does ToMe fail because averaging shrinks the merged tokens' norms?

**Strategies** (`eval_token_mask.py`):

| Strategy | Classic method | What happens to the n tokens | n |
|---|---|---|---|
| `attn_topk` | top-k pruning (FastV-style selection) | the n lowest-scored tokens are removed, as in `random` | 131, 196, 261 |
| `evit` | EViT token reorganization (Liang et al., ICLR 2022) | the n lowest-scored tokens are fused into one token: the score-weighted average of their VAE latents and positions, appended after the kept tokens, without proportional attention | 131, 196, 261 |
| `tofu` | Token Fusion MLERP (Kim et al., WACV 2024) | ToMe matching and proportional attention; each merged token is rescaled to the largest norm among the tokens it merges | 196, 261 |

**Score.** The score is computed per call, on the current observation, by a full-input pass: the prefix prefill plus the first action denoising step (with the same noise, this step equals step 1 of the 10-step schedule). It is the action-to-image attention averaged over layers, heads and action queries.

The previous call's attention cannot serve as the score: removed tokens receive zero attention, so the kept set would freeze after the first call.

The removal scope equals that of `random` and `grid` (prefix prefill and all action steps), so the three selections compare at equal budget.

**Fixed:** as in the strategy comparison grid. 8 conditions × 200 = 1,600 episodes.

**Recorded:** additionally, the per-call score (float16 [392]).

## K/V Reuse at Proportional Budgets

`kv_topk` at 131, 196 and 261 reused tokens: the tokens with the most action attention at the previous call are recomputed, and the rest reuse their K/V. `kv_random` at 196 completes the random control at 1/3, 1/2 and 2/3. Settings as in the strategy comparison grid; 4 conditions × 200 = 800 episodes, in `runs/20261005_kv_topk_ratio/`.

The 9-token conditions are kept unchanged.
