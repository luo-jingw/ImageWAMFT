#!/usr/bin/env bash
# Runs the PLAN.md grid with WORKERS_PER_GPU evaluator processes per GPU on this node (nnmc71).
#   GPUS=2,3,4,5 WORKERS_PER_GPU=3 bash research/libero_token_mask/run_grid.sh <out dir>
set -euo pipefail
PROJECT_ROOT=/home/user1/workspace/jingwu/ImageWAMFT
OUT="${1:?usage: run_grid.sh <out dir>}"
GPUS="${GPUS:?set GPUS, e.g. GPUS=2,3,4,5}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-3}"
IFS=, read -r -a GPU_LIST <<< "${GPUS}"
NUM_WORKERS=$(( ${#GPU_LIST[@]} * WORKERS_PER_GPU ))
mkdir -p "${OUT}/logs"
export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/research/libero_token_mask/libero_config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1
pids=()
for ((w = 0; w < NUM_WORKERS; w++)); do
  gpu="${GPU_LIST[$(( w % ${#GPU_LIST[@]} ))]}"
  CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" \
    "${PROJECT_ROOT}/.venv-libero/bin/python" "${PROJECT_ROOT}/research/libero_token_mask/eval_token_mask.py" \
    --out-dir "${OUT}" --worker-index "${w}" --num-workers "${NUM_WORKERS}" > "${OUT}/logs/worker${w}.log" 2>&1 &
  pids+=($!)
done
echo "[run_grid] $(date '+%F %T') started ${NUM_WORKERS} workers on GPUs ${GPUS}"
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
echo "[run_grid] $(date '+%F %T') finished, status ${status}"
