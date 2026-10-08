#!/usr/bin/env bash
# Builds .venv-qwenimage-nnmc73 on nnmc73: the .venv-qwenimage packages (Qwen-Image-2.1 / Qwen3-VL text features,
# transformers 5, diffusers at 578c9b2) with torch 2.9.1 and torchvision 0.24.1 for CUDA 13.0, the torch build of
# .venv-nnmc73, because the .venv-qwenimage torch (2.7.1+cu118) has no B300 (compute capability 10.3) support.
#
#   (on nnmc71) write the freeze below, then: ssh nnmc73 bash /shared/user71/workspace/jingwu/ImageWAMFT/node_env/build_venv_qwenimage_nnmc73.sh
set -euo pipefail
P=/shared/user71/workspace/jingwu/ImageWAMFT
V="${P}/.venv-qwenimage-nnmc73"
WORK="${P}/tmp/venv_qwenimage_nnmc73_build"
UV="${UV:-$HOME/.local/bin/uv}"
if [ "$(hostname)" != nnmc73 ]; then
  echo "run on nnmc73" >&2
  exit 2
fi
if [ -e "${V}" ]; then
  echo "${V} exists; remove it first to rebuild" >&2
  exit 2
fi
# The .venv-qwenimage interpreter resolves only on nnmc71, so its freeze is written there first:
#   uv pip freeze --python .venv-qwenimage/bin/python > tmp/venv_qwenimage_nnmc73_build/freeze_qwenimage.txt
grep -v -E '^(torch==|torchvision==|triton==|nvidia-)' "${WORK}/freeze_qwenimage.txt" > "${WORK}/requirements.txt"
printf 'torch==2.9.1\ntorchvision==0.24.1\n' > "${WORK}/overrides.txt"
"${UV}" venv "${V}" --python 3.11
"${UV}" pip install --python "${V}/bin/python" --link-mode copy torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130
"${UV}" pip install --python "${V}/bin/python" --link-mode copy -r "${WORK}/requirements.txt" \
  --override "${WORK}/overrides.txt"
"${UV}" pip freeze --python "${V}/bin/python" > "${WORK}/freeze_final.txt"
echo finished
