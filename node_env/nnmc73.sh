# Runtime environment for running ImageWAMFT on node nnmc73 (B300, compute capability 10.3).
# Usage on nnmc73:  source /shared/user71/workspace/jingwu/ImageWAMFT/node_env/nnmc73.sh
# GPUs are chosen per launch with CUDA_VISIBLE_DEVICES; only the granted GPUs may be used.

export PROJECT_ROOT=/shared/user71/workspace/jingwu/ImageWAMFT
export VIRTUAL_ENV="${PROJECT_ROOT}/.venv-nnmc73"
export PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

# DeepSpeed needs a CUDA toolkit at import time; nnmc73 has none in /usr/local, so use the pip nvcc 13.0.
export CUDA_HOME="${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cu13"

# torchcodec loads FFmpeg shared libraries; nnmc73 has none system-wide, so use the project FFmpeg 7.1 env.
export LD_LIBRARY_PATH="${PROJECT_ROOT}/.ffmpeg-nnmc73/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export HF_HUB_OFFLINE=1
