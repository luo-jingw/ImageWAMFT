#!/usr/bin/env bash
# Full G1 training of FLUX.2-klein-base-9B ImageWAM: same recipe, data and text features as
# launch/g1_flux2_4b_train.sh, normalization statistics from the FLUX.2-4B run (plan.md D8).
# Effective batch 64 = 4 per GPU x number of GPUs x gradient accumulation. About 10B parameters are trained, so it
# needs one B300 on nnmc73 (.venv-nnmc73) or nnmc74/nnmc75 (.venv-b300), or four H100 on nnmc71/nnmc72 (.venv):
#   GPUS=6 bash /shared/user71/workspace/jingwu/ImageWAMFT/launch/g1_flux2_9b_train.sh          # nnmc73-75
#   GPUS=0,1,2,3 bash /home/user1/workspace/jingwu/ImageWAMFT/launch/g1_flux2_9b_train.sh   # nnmc71 / nnmc72
# Resume after an interruption by appending the latest state directory:
#   GPUS=6 bash .../launch/g1_flux2_9b_train.sh resume=<OUT>/checkpoints/state/step_XXXXXX
set -euo pipefail

case "$(hostname)" in
  nnmc73|nnmc74|nnmc75) PROJECT_ROOT=/shared/user71/workspace/jingwu/ImageWAMFT ;;
  nnmc71|nnmc72) PROJECT_ROOT=/home/user1/workspace/jingwu/ImageWAMFT ;;
  *) echo "No verified environment for $(hostname); use nnmc71-nnmc75" >&2; exit 2 ;;
esac
RUN_NAME=g1_lgg100_flux2_klein_9b_base_imagewam
OUT="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"
GPUS="${GPUS:?set GPUS to the GPU indices, e.g. GPUS=6}"
NUM_GPUS=$(awk -F, '{print NF}' <<< "${GPUS}")
if (( 64 % (4 * NUM_GPUS) != 0 )); then
  echo "Effective batch 64 is not divisible by 4 x ${NUM_GPUS} GPUs" >&2
  exit 2
fi
ACCUM=$(( 64 / (4 * NUM_GPUS) ))
NORM_STATS="${PROJECT_ROOT}/checkpoints/g1_lgg100_flux2_klein_4b_base_imagewam/dataset_stats.json"

mkdir -p "${OUT}"
cp -n "${NORM_STATS}" "${OUT}/dataset_stats.json"
# shellcheck disable=SC1091
case "$(hostname)" in
  nnmc73) source "${PROJECT_ROOT}/node_env/nnmc73.sh" ;;
  nnmc74|nnmc75) source "${PROJECT_ROOT}/node_env/b300.sh" ;;
  *) source "${PROJECT_ROOT}/.venv/bin/activate" ;;
esac
cd "${PROJECT_ROOT}/ImageWAM"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT="${MASTER_PORT:-29531}"

echo "[launch] $(date '+%F %T') host=$(hostname) gpus=${GPUS} accumulation=${ACCUM} out=${OUT} args=$*" | tee -a "${OUT}/train.log"
CUDA_VISIBLE_DEVICES="${GPUS}" GPU_PER_NODE="${NUM_GPUS}" TASK_TYPE=g1 FLUX2_VARIANT=9b RUN_ID="${RUN_NAME}" \
  bash scripts/flux2/run_train_flux2_klein_imagewam.sh \
  "output_dir=${OUT}" \
  "gradient_accumulation_steps=${ACCUM}" \
  "data.train.pretrained_norm_stats=${NORM_STATS}" \
  "$@" 2>&1 | tee -a "${OUT}/train.log"
