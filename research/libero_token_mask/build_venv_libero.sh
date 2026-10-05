#!/usr/bin/env bash
# Builds .venv-libero: the .venv package set plus the LIBERO simulator stack (ImageWAM/scripts/setup/_install_libero_env.sh
# without apt-get; rendering uses the driver EGL via MUJOCO_GL=egl).
set -euo pipefail
P=/home/user1/workspace/jingwu/ImageWAMFT
cd "$P"
uv venv .venv-libero --python 3.11
uv pip install --python .venv-libero/bin/python -r tmp/freeze_venv.txt --extra-index-url https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match
uv pip install --python .venv-libero/bin/python --no-deps -e ImageWAM
uv pip install --python .venv-libero/bin/python mujoco==3.3.2 robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 easydict thop future cloudpickle
if [ ! -d ImageWAM/third_party/LIBERO ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ImageWAM/third_party/LIBERO
fi
touch ImageWAM/third_party/LIBERO/libero/libero/__init__.py
.venv-libero/bin/python - <<'PATCHPY'
from pathlib import Path
path = Path("ImageWAM/third_party/LIBERO/libero/libero/benchmark/__init__.py")
text = path.read_text()
old = "init_states = torch.load(init_states_path)"
new = "init_states = torch.load(init_states_path, weights_only=False)"
if old in text:
    path.write_text(text.replace(old, new))
elif new not in text:
    raise RuntimeError("could not patch torch.load")
PATCHPY
uv pip install --python .venv-libero/bin/python --no-deps -e ImageWAM/third_party/LIBERO
# LIBERO has no top-level libero/__init__.py, so the PEP 660 editable install maps no packages;
# put the repository root on sys.path instead (what `setup.py develop` did).
echo "$P/ImageWAM/third_party/LIBERO" > .venv-libero/lib/python3.11/site-packages/libero_repo_root.pth
echo BUILD_DONE
