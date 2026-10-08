#!/usr/bin/env bash
# Qwen-Image-2.1 ImageWAM training. TASK_TYPE=g1 (default): the LGG100 Unitree G1 data; TASK_TYPE=robotwin: RoboTwin 2.0
# (task robotwin_qwenimage21_imagewam; ROBOTWIN_ROOT and NONIDLE_FILTER_PATH required).
# Data overrides match the corresponding branch of scripts/flux2/run_train_flux2_klein_imagewam.sh; the text cache
# (scripts/qwenimage21/precompute_qwenimage21_text_embeds.py, run in .venv-qwenimage) must already exist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-2}"
ZERO_STAGE="${ZERO_STAGE:-2}"

imagewam_require_env DATA_ROOT
imagewam_require_env MODEL_ROOT

QWENIMAGE21_ROOT="${QWENIMAGE21_ROOT:-${MODEL_ROOT}/Qwen-Image-2.1}"
TASK_TYPE="${TASK_TYPE:-g1}"
DATA_OVERRIDES=()
case "${TASK_TYPE}" in
  g1)
    TASK_NAME="g1_lgg100_qwenimage21_imagewam"
    ACTION_DIM=16
    G1_ROOT="${G1_ROOT:-${DATA_ROOT}/LGG100}"
    DATASET_DIRS="[${G1_ROOT}/drawer,${G1_ROOT}/biomanual,${G1_ROOT}/simple-pick,${G1_ROOT}/Stack-the-cubes]"
    QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-${REPO_ROOT}/data/g1_lgg100/qwenimage21_text_cache}"
    ACTION_INIT="${ACTION_INIT:-checkpoints/action_dit_qwenimage21_g1_init.pt}"
    ;;
  robotwin)
    TASK_NAME="robotwin_qwenimage21_imagewam"
    ACTION_DIM=14
    imagewam_require_env ROBOTWIN_ROOT
    imagewam_require_env NONIDLE_FILTER_PATH
    imagewam_require_env QWEN_CACHE_DIR
    DATASET_DIRS="[${ROBOTWIN_ROOT}]"
    ACTION_INIT="${ACTION_INIT:-checkpoints/action_dit_qwenimage21_robotwin_init.pt}"
    DATA_OVERRIDES=(
      "data.train.nonidle_filter_path=${NONIDLE_FILTER_PATH}"
      "data.val.nonidle_filter_path=null"
    )
    ;;
  *) echo "Invalid TASK_TYPE=${TASK_TYPE}; expected g1 or robotwin" >&2; exit 1 ;;
esac
NORM_STATS="${NORM_STATS:?set NORM_STATS to the normalization statistics dataset_stats.json}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [ ! -d "${QWEN_CACHE_DIR}" ] || [ -z "$(ls -A "${QWEN_CACHE_DIR}")" ]; then
  echo "Missing Qwen-Image-2.1 text cache in ${QWEN_CACHE_DIR}" >&2
  exit 2
fi
if [ ! -f "${NORM_STATS}" ]; then
  echo "Missing normalization statistics ${NORM_STATS}" >&2
  exit 2
fi

imagewam_print_config TASK_NAME DATA_ROOT QWENIMAGE21_ROOT QWEN_CACHE_DIR NORM_STATS ACTION_INIT ZERO_STAGE GPU_PER_NODE

if [ "${REBUILD_ACTION_INIT:-false}" = "true" ] || [ ! -f "${ACTION_INIT}" ]; then
  imagewam_run imagewam_python scripts/qwenimage21/preprocess_action_dit_qwenimage21.py \
    --transformer-dir "${QWENIMAGE21_ROOT}/transformer" \
    --action-dim "${ACTION_DIM}" \
    --output "${ACTION_INIT}"
fi

case "${ZERO_STAGE}" in
  1|zero1) TRAIN_SCRIPT="scripts/train_zero1.sh" ;;
  2|zero2) TRAIN_SCRIPT="scripts/train_zero2.sh" ;;
  *) echo "Invalid ZERO_STAGE=${ZERO_STAGE}; expected 1, zero1, 2, or zero2" >&2; exit 1 ;;
esac

TASK="${TASK_NAME}" imagewam_run bash "${TRAIN_SCRIPT}" "${GPU_PER_NODE}" \
  task="${TASK_NAME}" \
  "data.train.dataset_dirs=${DATASET_DIRS}" \
  "data.val.dataset_dirs=${DATASET_DIRS}" \
  "data.train.qwen_text_cache_dir=${QWEN_CACHE_DIR}" \
  "data.val.qwen_text_cache_dir=${QWEN_CACHE_DIR}" \
  "data.train.pretrained_norm_stats=${NORM_STATS}" \
  "model.proprio_dim=${ACTION_DIM}" \
  "model.transformer_dir=${QWENIMAGE21_ROOT}/transformer" \
  "model.vae_dir=${QWENIMAGE21_ROOT}/vae" \
  "model.action_dit_pretrained_path=${ACTION_INIT}" \
  "${DATA_OVERRIDES[@]}" \
  "$@"
