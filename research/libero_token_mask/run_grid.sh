#!/usr/bin/env bash
# Runs the PLAN.md grid with WORKERS_PER_GPU evaluator processes per GPU on this node (nnmc71).
#   GPUS=2,3,4,5 WORKERS_PER_GPU=3 bash research/libero_token_mask/run_grid.sh <out dir>
# Workers start STAGGER_SECONDS apart: concurrent start-up of several workers on one GPU aborted in the EGL/CUDA
# initialization. WORKER_INDICES restarts selected shards (shard i is units[i::num_workers]; finished episodes are
# skipped).
set -euo pipefail
PROJECT_ROOT=/home/user1/workspace/jingwu/ImageWAMFT
OUT="${1:?usage: run_grid.sh <out dir>}"
GPUS="${GPUS:?set GPUS, e.g. GPUS=2,3,4,5}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-3}"
STAGGER_SECONDS="${STAGGER_SECONDS:-60}"
IFS=, read -r -a GPU_LIST <<< "${GPUS}"
NUM_WORKERS=$(( ${#GPU_LIST[@]} * WORKERS_PER_GPU ))
mkdir -p "${OUT}/logs"
export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/research/libero_token_mask/libero_config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1
read -r -a INDICES <<< "${WORKER_INDICES:-$(seq -s ' ' 0 $(( NUM_WORKERS - 1 )))}"
pids=()
for w in "${INDICES[@]}"; do
  gpu="${GPU_LIST[$(( w % ${#GPU_LIST[@]} ))]}"
  # Supervised: an abnormal exit (intermittent abort inside robosuite EGL `read_pixels`) restarts the worker, which
  # skips finished episodes; at most MAX_RESTARTS restarts per worker.
  (
    for ((attempt = 0; attempt <= ${MAX_RESTARTS:-20}; attempt++)); do
      CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" \
        "${PROJECT_ROOT}/.venv-libero/bin/python" -X faulthandler \
        "${PROJECT_ROOT}/research/libero_token_mask/eval_token_mask.py" \
        --out-dir "${OUT}" --worker-index "${w}" --num-workers "${NUM_WORKERS}" >> "${OUT}/logs/worker${w}.log" 2>&1
      code=$?
      if (( code == 0 )); then break; fi
      echo "[run_grid] $(date '+%F %T') worker ${w} exited with status ${code}; restart $(( attempt + 1 ))"
      sleep 10
    done
  ) &
  pids+=($!)
  echo "[run_grid] $(date '+%F %T') worker ${w} on GPU ${gpu}"
  sleep "${STAGGER_SECONDS}"
done
echo "[run_grid] $(date '+%F %T') started workers ${INDICES[*]} of ${NUM_WORKERS} on GPUs ${GPUS}"
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
echo "[run_grid] $(date '+%F %T') finished, status ${status}"
