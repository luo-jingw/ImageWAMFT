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
