#!/usr/bin/env bash
# G1 policy server (deploy/g1_server, docs/g1_policy_server.md): the WebSocket endpoint for the robot client
# (tmp/g1-client/fastwam/main.py --server-host <this node> --server-port WS_PORT) and the web page on UI_PORT.
# Run it inside tmux; GPU is one idle GPU of this node (check nvidia-smi first; at most 4 project GPUs per node).
# MODELS: models loaded at start, the first one active. SHARE=1 adds a public Gradio share link; AUTH_FILE (user:password)
# protects the page.
#   tmux new-session -d -s imagewamft_g1_server \
#     "GPU=0 MODELS=flux2_4b,qwenimage21,flux2_9b SHARE=1 AUTH_FILE=deploy/g1_server/ui_auth.txt bash launch/g1_server.sh"
set -euo pipefail

PROJECT_ROOT=/home/user1/workspace/jingwu/ImageWAMFT
GPU="${GPU:?set GPU, the index of an idle GPU on this node}"
MODELS="${MODELS:-flux2_4b}"
WS_PORT="${WS_PORT:-8000}"
UI_PORT="${UI_PORT:-7860}"
LOG_DIR="${PROJECT_ROOT}/deploy/g1_server/logs"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PROJECT_ROOT}/deploy${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False WANDB_MODE=offline
"${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/deploy/g1_server/main.py" \
  --models "${MODELS}" --ws-port "${WS_PORT}" --ui-port "${UI_PORT}" ${SHARE:+--share} \
  ${AUTH_FILE:+--auth-file "${AUTH_FILE}"} \
  2>&1 | tee -a "${LOG_DIR}/server_$(date +%Y%m%d_%H%M%S).log"
