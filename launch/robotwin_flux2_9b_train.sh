#!/usr/bin/env bash
# FLUX.2-klein-base-9B ImageWAM on RoboTwin 2.0 with the released FLUX.2-4B recipe (plan.md; docs/robotwin_training_setup.md):
# global batch 384 = 4 per GPU x gradient accumulation x GPUs, 5 epochs, DeepSpeed ZeRO-2, on nnmc71
# (8 H100, user-approved exception to the 4-GPU policy). The Qwen3-8B text cache (2.9 TB) lives on nnmc73's disk and
# is read over NFS.
#
#   GRAD_CLIP=none GPUS=0,1,2,3,4,5,6,7 bash launch/robotwin_flux2_9b_train.sh [hydra overrides]
#
# GRAD_CLIP (required): `none` uses the upstream DeepSpeed config without gradient clipping, the configuration the
# released model was trained with; `1.0` uses the project config with gradient_clipping 1.0 (paper Table 8).
# MODE=cache builds the text cache for all task strings on the given GPUs and exits.
# SMOKE_STEPS=N stops after N optimizer steps and writes to checkpoints/<run>_smoke.
# Resume after an interruption by appending resume=<OUT>/checkpoints/state/step_XXXXXX.
set -euo pipefail
PROJECT_ROOT=/home/user1/workspace/jingwu/ImageWAMFT
if [ "$(hostname)" != nnmc71 ]; then
  echo "run on nnmc71" >&2
  exit 2
fi
RUN_NAME=robotwin_flux2_klein_9b_base_b384_imagewam
GPUS="${GPUS:?set GPUS, e.g. GPUS=0,1,2,3,4,5,6,7}"
NUM_GPUS=$(awk -F, '{print NF}' <<< "${GPUS}")
MODE="${MODE:-train}"

export ROBOTWIN_ROOT="${PROJECT_ROOT}/dataset/robotwin2.0/robotwin2.0"
export NONIDLE_FILTER_PATH="${ROBOTWIN_ROOT}/nonidle_ranges.json"
export QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-/shared/user73/workspace/jingwu/ImageWAMFT_nnmc73/text_cache/flux2_qwen3_cache_9b}"
NORM_STATS="${PROJECT_ROOT}/dataset/robotwin2.0/dataset_stats.json"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/.venv/bin/activate"
cd "${PROJECT_ROOT}/ImageWAM"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT="${MASTER_PORT:-29541}"

if [ "${MODE}" = cache ]; then
  mkdir -p "${QWEN_CACHE_DIR}"
  # scripts/common.sh resolves FLUX2_SRC and FLUX2_QWEN3_MODEL_SPEC (BaseModel/Qwen3-8B) from .env.local.
  export FLUX2_VARIANT=9b
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    scripts/flux2/precompute_flux2_qwen3_embeds.py \
    task="${RUN_NAME}" \
    qwen_cache_batch_size="${QWEN_CACHE_BATCH_SIZE:-32}" \
    qwen_cache_save_workers="${QWEN_CACHE_SAVE_WORKERS:-8}" \
    qwen_cache_overwrite=false \
    model.flux2_src_path="${FLUX2_SRC}" \
    model.variant=klein-base-9b \
    model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
    flux2_qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
    "data.train.dataset_dirs=[${ROBOTWIN_ROOT}]" \
    "data.val.dataset_dirs=[${ROBOTWIN_ROOT}]" \
    "data.train.nonidle_filter_path=${NONIDLE_FILTER_PATH}" \
    "data.train.qwen_text_cache_dir=${QWEN_CACHE_DIR}" \
    "data.val.qwen_text_cache_dir=${QWEN_CACHE_DIR}" \
    "data.train.pretrained_norm_stats=${NORM_STATS}" \
    "$@"
  exit 0
fi

GRAD_CLIP="${GRAD_CLIP:?set GRAD_CLIP=none (released model, upstream DeepSpeed config) or GRAD_CLIP=1.0 (paper Table 8)}"
case "${GRAD_CLIP}" in
  none) export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero2_upstream_ds.yaml ;;
  1.0) export ACCELERATE_CONFIG_FILE=scripts/accelerate_configs/accelerate_zero2_ds.yaml ;;
  *) echo "GRAD_CLIP must be none or 1.0" >&2; exit 2 ;;
esac
if (( 384 % (4 * NUM_GPUS) != 0 )); then
  echo "Global batch 384 is not divisible by 4 x ${NUM_GPUS} GPUs" >&2
  exit 2
fi
ACCUM=$(( 384 / (4 * NUM_GPUS) ))
EXTRA=()
OUT="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"
if [ -n "${SMOKE_STEPS:-}" ]; then
  OUT="${OUT}_smoke"
  EXTRA+=("max_steps=${SMOKE_STEPS}" "save_every=1000000" "eval_every=${SMOKE_STEPS}")
fi
mkdir -p "${OUT}"
cp -n "${NORM_STATS}" "${OUT}/dataset_stats.json"
echo "[launch] $(date '+%F %T') host=$(hostname) gpus=${GPUS} accumulation=${ACCUM} grad_clip=${GRAD_CLIP} out=${OUT} args=$*" \
  | tee -a "${OUT}/train.log"
CUDA_VISIBLE_DEVICES="${GPUS}" GPU_PER_NODE="${NUM_GPUS}" TASK_TYPE=robotwin FLUX2_VARIANT=9b \
  ROBOTWIN_TASK_NAME="${RUN_NAME}" RUN_ID="${RUN_NAME}" ZERO_STAGE=2 \
  bash scripts/flux2/run_train_flux2_klein_imagewam.sh \
  "output_dir=${OUT}" \
  "gradient_accumulation_steps=${ACCUM}" \
  "data.train.pretrained_norm_stats=${NORM_STATS}" \
  "data.val.nonidle_filter_path=null" \
  "${EXTRA[@]}" \
  "$@" 2>&1 | tee -a "${OUT}/train.log"
