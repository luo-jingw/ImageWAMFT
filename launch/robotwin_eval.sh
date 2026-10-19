#!/usr/bin/env bash
# RoboTwin 2.0 evaluation under the paper protocol (plan.md Goal 3; docs/robotwin_training_setup.md "Evaluation Protocol"):
# 50 tasks, clean and randomized, seed 42, unseen instructions, 10 denoising steps, action horizon 16, replan 16,
# 100 episodes per task. Runs the repository's evaluation manager with .venv-robotwin on one node: nnmc71, or an
# L40S node with the project symlink and the uv CPython 3.11.13 (PROJECT.md). Several nodes split the tasks with
# TASKS lists from launch/robotwin_eval_shards.py, each with its own OUTPUT_TAG (plan.md E5, E9).
#
#   MODEL=flux2_4b_release GPUS=4,5,6,7 bash launch/robotwin_eval.sh [hydra overrides]
#   MODEL=flux2_9b CKPT=<run>/checkpoints/weights/step_XXXXXX.pt GPUS=4,5,6,7 bash launch/robotwin_eval.sh
#   MODEL=qwenimage21 CKPT=<run>/checkpoints/weights/step_XXXXXX.pt GPUS=4,5,6,7 bash launch/robotwin_eval.sh
#
# MODEL: flux2_4b_release (the released FLUX.2-4B), flux2_9b or qwenimage21 (CKPT required; statistics from the run
# directory unless DATASET_STATS is set).
# - FLUX.2 runs the repository launcher scripts/flux2/run_eval_flux2_robotwin.sh; the policy encodes instructions with
#   its own Qwen3 encoder.
# - Qwen-Image-2.1 has no in-process text encoder. The policy fetches the training-identical Qwen3-VL features from
#   scripts/qwenimage21/qwen_text_server.py (.venv-qwenimage); the features are kept in
#   checkpoints/robotwin_eval_text_cache/qwenimage21/. TEXT_SERVER=<host>:<port> uses a running server, e.g. one
#   started with --host 0.0.0.0 on its own GPU for several nodes (a server and a worker do not fit one 46 GB L40S).
#   Without TEXT_SERVER the launcher starts a local server on the first GPU.
# GPUS: absolute device ids; the evaluation worker sets CUDA_VISIBLE_DEVICES itself.
# EPISODES (default 100), PHASES (default [clean,random]), TASKS (comma list, default all 50 tasks),
# MAX_TASKS_PER_GPU (default 1 on L40S; on H100 3 for 4B, 2 otherwise), OUTPUT_TAG (results subdirectory name, default a timestamp).
# Results: ImageWAM/evaluate_results/robotwin/<checkpoint tag>/<OUTPUT_TAG>/ (summary.csv, summary.json, manager.log).
# Resume an interrupted run: the same command and OUTPUT_TAG plus MULTIRUN.resume=true (complete tasks are kept,
# the others rerun from their first episode).
set -euo pipefail
PROJECT_ROOT=/home/user1/workspace/jingwu/ImageWAMFT
if ! "${PROJECT_ROOT}/.venv-robotwin/bin/python" -c "" 2>/dev/null; then
  echo "${PROJECT_ROOT}/.venv-robotwin/bin/python does not run on $(hostname): it needs ${PROJECT_ROOT} (a symlink to" \
    "/shared/user71/workspace/jingwu/ImageWAMFT off nnmc71) and the uv CPython 3.11.13" >&2
  exit 2
fi
MODEL="${MODEL:?set MODEL=flux2_4b_release, flux2_9b or qwenimage21}"
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5,6,7}"
NUM_GPUS=$(awk -F, '{print NF}' <<< "${GPUS}")
# GPUS are CUDA indices, which the workers pass to CUDA_VISIBLE_DEVICES. They can differ from nvidia-smi indices
# (nnmc64: one GPU is not visible to CUDA), so each index is resolved to its UUID through CUDA. A GPU is used only
# when no process of anyone holds it (PROJECT.md GPU policy).
GPU_APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
for gpu in ${GPUS//,/ }; do
  uuid=$(CUDA_VISIBLE_DEVICES="${gpu}" "${PROJECT_ROOT}/.venv-robotwin/bin/python" -c \
    "import torch; print('GPU-' + str(torch.cuda.get_device_properties(0).uuid))" 2>/dev/null) || {
    echo "CUDA index ${gpu} does not exist on $(hostname)" >&2; exit 2; }
  smi_index=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v u="${uuid}" '$2 == u {print $1}')
  if grep -qF "${uuid}" <<< "${GPU_APPS}"; then
    echo "GPU ${gpu} (${uuid}, nvidia-smi ${smi_index}) on $(hostname) has a process; not launching" >&2
    exit 2
  fi
  echo "[gpu] CUDA ${gpu} = ${uuid} = nvidia-smi ${smi_index}, no process"
done
BASE="${PROJECT_ROOT}/BaseModel"
run_stats() { echo "${DATASET_STATS:-$(dirname "$(dirname "$(dirname "$1")")")/dataset_stats.json}"; }
case "${MODEL}" in
  flux2_4b_release)
    export FLUX2_VARIANT=4b
    export CKPT_PATH="${PROJECT_ROOT}/checkpoints/imagewam_release/robotwin/flux2_klein_4b/model.pt"
    export DATASET_STATS_PATH="${PROJECT_ROOT}/checkpoints/imagewam_release/robotwin/flux2_klein_4b/dataset_stats.json"
    export FLUX2_MODEL_PATH="${BASE}/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"
    export FLUX2_QWEN3_MODEL_SPEC="${BASE}/Qwen3-4B"
    DEFAULT_TASKS_PER_GPU=3
    ;;
  flux2_9b)
    export FLUX2_VARIANT=9b
    export CKPT_PATH="${CKPT:?set CKPT to a weights file of the FLUX.2-9B RoboTwin run}"
    DATASET_STATS_PATH="$(run_stats "${CKPT_PATH}")"
    export DATASET_STATS_PATH
    export FLUX2_MODEL_PATH="${BASE}/flux2/FLUX.2-klein-base-9B/flux-2-klein-base-9b.safetensors"
    export FLUX2_QWEN3_MODEL_SPEC="${BASE}/Qwen3-8B"
    DEFAULT_TASKS_PER_GPU=2
    ;;
  qwenimage21)
    CKPT_PATH="${CKPT:?set CKPT to a weights file of the Qwen-Image-2.1 RoboTwin run}"
    DATASET_STATS_PATH="$(run_stats "${CKPT_PATH}")"
    DEFAULT_TASKS_PER_GPU=2
    ;;
  *) echo "MODEL must be flux2_4b_release, flux2_9b or qwenimage21" >&2; exit 2 ;;
esac
export PYTHON_BIN="${PROJECT_ROOT}/.venv-robotwin/bin/python"
export HF_HUB_OFFLINE=1
export NUM_GPUS
# One worker per 46 GB L40S for every model: a worker peaks at 24 GB (FLUX.2-4B, Qwen-Image-2.1) and the FLUX.2-9B
# worker loses its Vulkan device in randomized scenes (plan.md Phase 4). The per-model defaults are for 94 GB H100.
if [ "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${uuid}")" -lt 60000 ]; then
  DEFAULT_TASKS_PER_GPU=1
fi
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-${DEFAULT_TASKS_PER_GPU}}"
export EVAL_NUM_EPISODES="${EPISODES:-100}"
export PHASES="${PHASES:-[clean,random]}"
export ACTION_HORIZON=16 REPLAN_STEPS=16
OUTPUT_TAG="${OUTPUT_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXTRA=("MULTIRUN.gpu_ids=[${GPUS}]" "EVALUATION.output_dir=./evaluate_results/robotwin/${MODEL}/${OUTPUT_TAG}")
if [ -n "${TASKS:-}" ]; then
  EXTRA+=("EVALUATION.task_name=${TASKS}")
fi
unset CUDA_VISIBLE_DEVICES
cd "${PROJECT_ROOT}/ImageWAM"
echo "[launch] $(date '+%F %T') host=$(hostname) model=${MODEL} ckpt=${CKPT_PATH} gpus=${GPUS} episodes=${EVAL_NUM_EPISODES} phases=${PHASES} tasks=${TASKS:-all} tag=${OUTPUT_TAG} args=$*"

if [ "${MODEL}" != qwenimage21 ]; then
  bash scripts/flux2/run_eval_flux2_robotwin.sh "${EXTRA[@]}" "$@"
  exit 0
fi

# Qwen-Image-2.1: text-feature server, then the manager with the same settings as run_eval_flux2_robotwin.sh.
TEXT_CACHE="${PROJECT_ROOT}/checkpoints/robotwin_eval_text_cache/qwenimage21"
if [ -z "${TEXT_SERVER:-}" ]; then
  TEXT_PORT="${TEXT_PORT:-29610}"
  TEXT_SERVER="127.0.0.1:${TEXT_PORT}"
  TEXT_LOG="${PROJECT_ROOT}/checkpoints/robotwin_logs/qwen_text_server_${OUTPUT_TAG}.log"
  mkdir -p "${TEXT_CACHE}" "$(dirname "${TEXT_LOG}")"
  (cd scripts/qwenimage21 && CUDA_VISIBLE_DEVICES="${GPUS%%,*}" exec "${PROJECT_ROOT}/.venv-qwenimage/bin/python" -u qwen_text_server.py \
    --model-dir "${BASE}/Qwen-Image-2.1" --cache-dir "${TEXT_CACHE}" --context-len 128 --port "${TEXT_PORT}" \
    --device cuda:0 > "${TEXT_LOG}" 2>&1) &
  TEXT_PID=$!
  trap 'kill "${TEXT_PID}" 2>/dev/null || true' EXIT
  until grep -q "^ready" "${TEXT_LOG}" 2>/dev/null; do
    if ! kill -0 "${TEXT_PID}" 2>/dev/null; then
      echo "text server exited; see ${TEXT_LOG}" >&2
      exit 1
    fi
    sleep 2
  done
fi
echo "[launch] text server ${TEXT_SERVER}"
export PYTHONPATH="${PROJECT_ROOT}/ImageWAM/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" experiments/robotwin/run_robotwin_manager.py \
  --config-name sim_robotwin \
  task=robotwin_qwenimage21_imagewam \
  ckpt="${CKPT_PATH}" \
  EVALUATION.dataset_stats_path="${DATASET_STATS_PATH}" \
  model.transformer_dir="${BASE}/Qwen-Image-2.1/transformer" \
  model.vae_dir="${BASE}/Qwen-Image-2.1/vae" \
  model.proprio_dim=14 \
  model.pack_proprio_after_text=true \
  "data.train.qwen_text_cache_dir=${TEXT_CACHE}" \
  "data.val.qwen_text_cache_dir=${TEXT_CACHE}" \
  data.train.nonidle_filter_path=null \
  batch_size=1 \
  gradient_accumulation_steps=1 \
  EVALUATION.qwen_text_server="${TEXT_SERVER}" \
  MULTIRUN.num_gpus="${NUM_GPUS}" \
  MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU}" \
  EVALUATION.action_horizon="${ACTION_HORIZON}" \
  EVALUATION.replan_steps="${REPLAN_STEPS}" \
  EVALUATION.eval_num_episodes="${EVAL_NUM_EPISODES}" \
  MULTIRUN.phases="${PHASES}" \
  EVALUATION.skip_get_obs_within_replan=true \
  EVALUATION.robotwin_camera_layout=compact_288x256 \
  "${EXTRA[@]}" "$@"
