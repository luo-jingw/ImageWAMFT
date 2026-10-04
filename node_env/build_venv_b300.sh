#!/usr/bin/env bash
# Builds .venv-b300 (B300 nodes nnmc73-75) with the .venv-nnmc73 recipe and the project-local CPython 3.11.15.
set -euo pipefail
P=/shared/user71/workspace/jingwu/ImageWAMFT
UV=/shared/user73/.local/bin/uv
export UV_CACHE_DIR=$P/tmp/uv-cache-b300
cd "$P"
"$UV" venv .venv-b300 --python "$P/.uv-python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11"
"$UV" pip install --python .venv-b300/bin/python --link-mode copy torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
cd "$P/ImageWAM"
"$UV" pip install --python ../.venv-b300/bin/python --link-mode copy --no-sources -e '.[shared]' --override ../node_env/nnmc73_overrides.txt
cd "$P"
"$UV" pip install --python .venv-b300/bin/python --link-mode copy nvidia-cuda-nvcc==13.0.88
echo BUILD_DONE
