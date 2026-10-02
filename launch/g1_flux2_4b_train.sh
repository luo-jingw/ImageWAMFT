#!/usr/bin/env bash
# Full G1 training of FLUX.2-klein-base-4B ImageWAM (plan.md Phase 6): 2 GPUs, 10 epochs,
# effective batch 64 (4 per GPU x 2 GPUs x gradient accumulation 8), weights every 5,000 steps.
#
# Run on an H100 node where /home/user1/workspace/jingwu/ImageWAMFT resolves (nnmc71, or nnmc72 via its symlink):
#   GPUS=6,7 bash /home/user1/workspace/jingwu/ImageWAMFT/launch/g1_flux2_4b_train.sh
# Resume after an interruption by appending the latest state directory:
#   GPUS=6,7 bash .../launch/g1_flux2_4b_train.sh resume=<OUT>/checkpoints/state/step_XXXXXX
set -euo pipefail

PROJECT_ROOT=/home/user1/workspace/jingwu/ImageWAMFT
RUN_NAME=g1_lgg100_flux2_klein_4b_base_imagewam
OUT="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"
GPUS="${GPUS:?set GPUS to two GPU indices, e.g. GPUS=6,7}"

mkdir -p "${OUT}"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/.venv/bin/activate"
cd "${PROJECT_ROOT}/ImageWAM"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[launch] $(date '+%F %T') host=$(hostname) gpus=${GPUS} out=${OUT} args=$*" | tee -a "${OUT}/train.log"
CUDA_VISIBLE_DEVICES="${GPUS}" GPU_PER_NODE=2 TASK_TYPE=g1 FLUX2_VARIANT=4b RUN_ID="${RUN_NAME}" \
  bash scripts/flux2/run_train_flux2_klein_imagewam.sh \
  "output_dir=${OUT}" \
  "$@" 2>&1 | tee -a "${OUT}/train.log"
