#!/usr/bin/env bash
# Qwen-Image-2.1 ImageWAM on RoboTwin 2.0 with the released FLUX.2-4B recipe (plan.md; docs/robotwin_training_setup.md):
# global batch 384 = MICRO_BATCH x GRAD_ACCUM x GPUs, 5 epochs, DeepSpeed ZeRO-2, on nnmc73 B300 (.venv-nnmc73).
# Dataset copy, Qwen3-VL text cache and run directory are on nnmc73's local disk (DATA_73 below).
#
#   GRAD_CLIP=1.0 MICRO_BATCH=48 GPUS=0,1,2,3,4,5,6,7 bash /shared/user71/workspace/jingwu/ImageWAMFT/launch/robotwin_qwenimage21_train.sh
#
# GRAD_CLIP (required): `none` uses the upstream DeepSpeed config without gradient clipping, the configuration the
# released model was trained with; `1.0` uses the project config with gradient_clipping 1.0 (paper Table 8).
# SMOKE_STEPS=N stops after N optimizer steps and writes to <run>_smoke.
# Resume after an interruption by appending resume=<OUT>/checkpoints/state/step_XXXXXX.
set -euo pipefail
if [ "$(hostname)" != nnmc73 ]; then
  echo "run on nnmc73" >&2
  exit 2
fi
# shellcheck disable=SC1091
source /shared/user71/workspace/jingwu/ImageWAMFT/node_env/nnmc73.sh
RUN_NAME=robotwin_qwenimage21_imagewam
DATA_73=/home/user1/workspace/jingwu/ImageWAMFT_nnmc73
GPUS="${GPUS:?set GPUS, e.g. GPUS=0,1,2,3}"
NUM_GPUS=$(awk -F, '{print NF}' <<< "${GPUS}")
MICRO_BATCH="${MICRO_BATCH:?set MICRO_BATCH, the per-GPU micro-batch (plan.md D2)}"
if (( 384 % (MICRO_BATCH * NUM_GPUS) != 0 )); then
  echo "Global batch 384 is not divisible by ${MICRO_BATCH} x ${NUM_GPUS} GPUs" >&2
  exit 2
fi
ACCUM=$(( 384 / (MICRO_BATCH * NUM_GPUS) ))
GRAD_CLIP="${GRAD_CLIP:?set GRAD_CLIP=none (released model, upstream DeepSpeed config) or GRAD_CLIP=1.0 (paper Table 8)}"
case "${GRAD_CLIP}" in
  none) export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero2_upstream_ds.yaml ;;
  1.0) export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero2_ds.yaml ;;
  *) echo "GRAD_CLIP must be none or 1.0" >&2; exit 2 ;;
esac

export ROBOTWIN_ROOT="${DATA_73}/dataset/robotwin2.0/robotwin2.0"
export NONIDLE_FILTER_PATH="${ROBOTWIN_ROOT}/nonidle_ranges.json"
export QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-${DATA_73}/text_cache/qwenimage21_text_cache}"
NORM_STATS="${DATA_73}/dataset/robotwin2.0/dataset_stats.json"
EXTRA=()
OUT="${DATA_73}/checkpoints/${RUN_NAME}"
if [ -n "${SMOKE_STEPS:-}" ]; then
  OUT="${OUT}_smoke"
  EXTRA+=("max_steps=${SMOKE_STEPS}" "save_every=1000000" "eval_every=${SMOKE_STEPS}")
fi
mkdir -p "${OUT}"
cp -n "${NORM_STATS}" "${OUT}/dataset_stats.json"
cd "${PROJECT_ROOT}/ImageWAM"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT="${MASTER_PORT:-29551}"
echo "[launch] $(date '+%F %T') host=$(hostname) gpus=${GPUS} micro=${MICRO_BATCH} accumulation=${ACCUM} grad_clip=${GRAD_CLIP} out=${OUT} args=$*" \
  | tee -a "${OUT}/train.log"
CUDA_VISIBLE_DEVICES="${GPUS}" GPU_PER_NODE="${NUM_GPUS}" TASK_TYPE=robotwin RUN_ID="${RUN_NAME}" NORM_STATS="${NORM_STATS}" \
  bash scripts/qwenimage21/run_train_qwenimage21_imagewam.sh \
  "output_dir=${OUT}" \
  "batch_size=${MICRO_BATCH}" \
  "gradient_accumulation_steps=${ACCUM}" \
  "${EXTRA[@]}" \
  "$@" 2>&1 | tee -a "${OUT}/train.log"
