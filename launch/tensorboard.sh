#!/usr/bin/env bash
# Live TensorBoard for the G1 runs, readable by anyone who can reach this node: mirrors the offline wandb logs of
# both runs into tensorboard/ every 2 minutes (ImageWAM/scripts/wandb_offline_to_tensorboard.py) and serves it on
# all interfaces. No wandb login or global setting is used or changed.
#   PORT=6006 bash /home/user1/workspace/jingwu/ImageWAMFT/launch/tensorboard.sh
set -euo pipefail

PROJECT_ROOT=/home/user1/workspace/jingwu/ImageWAMFT
LOGDIR="${PROJECT_ROOT}/tensorboard"
PORT="${PORT:-6006}"
export WANDB_MODE=offline
mkdir -p "${LOGDIR}"

"${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/ImageWAM/scripts/wandb_offline_to_tensorboard.py" \
  --logdir "${LOGDIR}" \
  --run "flux2_klein_4b=${PROJECT_ROOT}/checkpoints/g1_lgg100_flux2_klein_4b_base_imagewam" \
  --run "qwenimage21=${PROJECT_ROOT}/checkpoints/g1_lgg100_qwenimage21_imagewam" \
  --interval 120 > "${LOGDIR}/sync.log" 2>&1 &
SYNC_PID=$!
trap 'kill ${SYNC_PID} 2>/dev/null' EXIT

"${PROJECT_ROOT}/.venv/bin/tensorboard" --logdir "${LOGDIR}" --host 0.0.0.0 --port "${PORT}" --reload_interval 60
