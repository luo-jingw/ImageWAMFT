#!/usr/bin/env bash
# Full G1 training of Qwen-Image-2.1 ImageWAM (plan.md Phase 14): 2 GPUs, 10 epochs, effective batch 64
# (4 per GPU x 2 GPUs x gradient accumulation 8), weights every 5,000 steps; same recipe and data as
# launch/g1_flux2_4b_train.sh, normalization statistics taken from the FLUX.2-4B run (plan.md D8).
#
# Run on an H100 node where /home/user1/workspace/jingwu/ImageWAMFT resolves (nnmc71, or nnmc72 via its symlink):
#   GPUS=4,5 bash /home/user1/workspace/jingwu/ImageWAMFT/launch/g1_qwenimage21_train.sh
# OFFLOAD=1 keeps the AdamW state on the CPU (ZeRO-2 optimizer offload, DeepSpeedCPUAdam, same update).
# Resume after an interruption by appending the latest state directory:
#   GPUS=4,5 bash .../launch/g1_qwenimage21_train.sh resume=<OUT>/checkpoints/state/step_XXXXXX
set -euo pipefail

PROJECT_ROOT=/home/user1/workspace/jingwu/ImageWAMFT
RUN_NAME=g1_lgg100_qwenimage21_imagewam
OUT="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"
GPUS="${GPUS:?set GPUS to two GPU indices, e.g. GPUS=4,5}"
NORM_STATS="${PROJECT_ROOT}/checkpoints/g1_lgg100_flux2_klein_4b_base_imagewam/dataset_stats.json"

mkdir -p "${OUT}"
cp -n "${NORM_STATS}" "${OUT}/dataset_stats.json"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/.venv/bin/activate"
cd "${PROJECT_ROOT}/ImageWAM"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT="${MASTER_PORT:-29521}"

EXTRA=()
if [ "${OFFLOAD:-0}" = "1" ]; then
  # DeepSpeed's op builder rejects torch cu118 against the system CUDA 12.4; CPU Adam is CPU-only C++.
  export DS_SKIP_CUDA_CHECK=1
  export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero2_offload_ds.yaml
  EXTRA+=(optimizer_impl=deepspeed_cpu_adam)
fi

echo "[launch] $(date '+%F %T') host=$(hostname) gpus=${GPUS} offload=${OFFLOAD:-0} out=${OUT} args=$*" | tee -a "${OUT}/train.log"
CUDA_VISIBLE_DEVICES="${GPUS}" GPU_PER_NODE=2 RUN_ID="${RUN_NAME}" NORM_STATS="${NORM_STATS}" \
  bash scripts/qwenimage21/run_train_qwenimage21_imagewam.sh \
  "output_dir=${OUT}" \
  "${EXTRA[@]}" \
  "$@" 2>&1 | tee -a "${OUT}/train.log"
