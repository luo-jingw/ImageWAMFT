# OPT-001

Status: Proposed

Area: data sampling

## Observation

Uniform per-frame sampling weights tasks by recorded duration (ISSUE-005). biomanual receives 10.9% of samples and Stack-the-cubes 52.5%.

## Opportunity

Use task-balanced or episode-balanced sampling, for example one weight per dataset root.

## Expected Mechanism

A larger share of samples from short or underrepresented tasks raises their effective number of passes without adding data.

## Required Evidence

Per-task physical-robot success rates from a paper-default run, showing that the underrepresented tasks lag behind.

## Promotion Condition

biomanual or simple-pick success is clearly below the other tasks under paper-default sampling.

# OPT-002

Status: Proposed

Area: data loading throughput

## Observation

All LGG100 videos are AV1 at 480×640 and are decoded on CPU by torchcodec. Each sample decodes two endpoint frames from each of three cameras.

## Opportunity

Transcode the videos to H.264, or pre-decode the resized frames, in a separate dataset version.

## Expected Mechanism

AV1 decoding costs more CPU per frame than H.264, so cheaper decoding would raise dataloader throughput.

## Required Evidence

Dataloader samples/s compared with the GPU step rate, measured in a short training run. GPU utilization stalls caused by the input pipeline.

## Promotion Condition

Dataloader throughput is below the GPU step rate, so training is input-bound.

# OPT-003

Status: Proposed

Area: data sampling

## Observation

Capture repeats frames. Inside duplicate runs of at least 5 frames, the image stays frozen while the arm state keeps changing. This affects 0.24% of Stack-the-cubes frames and 0.10% of drawer frames, with stale images up to 0.7 s old (`docs/lgg100_dataset.md` → Data Quality Audit → Video Content).

## Opportunity

Drop training samples whose observation or target frames fall inside such runs, using a per-frame validity mask.

## Expected Mechanism

Removes image–action pairs where the image lags the robot state.

## Required Evidence

A per-sample ablation showing that these samples measurably affect action loss or success rate. At 0.1–0.24% of frames, an effect is unlikely without it.

## Promotion Condition

Physical-robot failures concentrated in phases that overlap camera stalls.

# OPT-004

Status: Proposed

Area: Qwen-Image-2.1 text conditioning

## Observation

The Qwen-Image-2.1 run conditions on text-only Qwen3-VL features, computed once per prompt with the text-to-image template (`plan.md` → D1). In its image-editing mode, Qwen-Image-2.1 instead feeds the reference image to Qwen3-VL, so the instruction features depend on the image.

## Opportunity

Image-aware instruction features computed once per episode from its first frame, with the editing template. Training stores one feature tensor per episode (339 in total). Inference runs Qwen3-VL once at episode start, so per-step latency is unchanged.

## Expected Mechanism

Instruction features grounded in the episode's initial scene, closer to the editing-mode pretraining distribution.

## Required Evidence

Qwen-Image-2.1 results under D1 compared with the FLUX.2-9B run on the same validation episodes, and per-task physical-robot success.

## Promotion Condition

The Qwen-Image-2.1 run under D1 falls clearly behind expectations, or fails to separate tasks that differ only in instruction or scene.

# OPT-005

Status: Proposed

Area: visual robustness (augmentation)

## Observation

The released ImageWAM-FLUX.2-4B LIBERO checkpoint loses success on the unmasked Spatial suite when the simulator renders with EGL instead of OSMesa: 34/39 against 50/50. The other settings are identical, including a fixed inference noise seed (`research/libero_token_mask/`).
- On the same state, the two backends differ by 1–2 gray levels on 54–58% of pixels.
- EGL is darker by 0.7–0.9 levels on average and slightly softer (gradient energy 0.95–0.98 of OSMesa).
- Only 0.6–1.3% of pixels differ by more than 10, and edge pixels carry 23–28% of the difference.
- The checkpoint was trained with photometric augmentation an order of magnitude stronger than this shift: `VideoAugmentation` with p 0.65, color jitter brightness/contrast 0.3, saturation 0.25, hue 0.04, plus corruption types. The G1 configuration has a similar augmentation, with gamma 0.9–1.1.

## Opportunity

Pick a training-time augmentation for perturbations of this size and type, once the source of the sensitivity is known. Candidates:
- per-pixel quantization noise of ±1–2 gray levels;
- small global gray-level offsets;
- light blur and sharpening;
- training on mixed renderers;
- an action-consistency loss between a clean and a perturbed view of the same state.

## Expected Mechanism

If the action expert reads features that move under sub-level pixel changes, training on such changes makes its actions invariant to them.

## Required Evidence

1. Closed-loop variance without a visual change: the unmasked Spatial success rate under OSMesa for several inference noise seeds. A spread comparable to the EGL drop means closed-loop amplification of any perturbation, not visual sensitivity.
2. Open-loop sensitivity on identical states: the action deviation between EGL and OSMesa renderings of the saved simulator states, compared with the deviation from a different noise seed and from masking 1/3 of the image tokens.

## Promotion Condition

The renderer-induced open-loop action deviation is comparable to or larger than the seed-induced deviation, and the success drop exceeds the seed-to-seed spread.
