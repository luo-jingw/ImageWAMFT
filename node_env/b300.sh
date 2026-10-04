# Runtime environment for ImageWAMFT on the B300 nodes nnmc73-nnmc75 (compute capability 10.3).
# Usage on any of them:  source /shared/user71/workspace/jingwu/ImageWAMFT/node_env/b300.sh
# .venv-b300 links to the project-local CPython in .uv-python/, so it resolves on every node through /shared/user71.
# GPUs are chosen per launch with CUDA_VISIBLE_DEVICES; only idle GPUs may be used.

export PROJECT_ROOT=/shared/user71/workspace/jingwu/ImageWAMFT
export VIRTUAL_ENV="${PROJECT_ROOT}/.venv-b300"
export PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

# DeepSpeed needs a CUDA toolkit at import time; use the pip nvcc 13.0 inside the environment.
export CUDA_HOME="${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cu13"

# torchcodec loads FFmpeg shared libraries from the project FFmpeg 7.1 prefix (the B300 nodes have no system FFmpeg).
export LD_LIBRARY_PATH="${PROJECT_ROOT}/.ffmpeg-nnmc73/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export HF_HUB_OFFLINE=1
