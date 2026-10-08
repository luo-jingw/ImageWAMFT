#!/usr/bin/env bash
# Builds .venv-robotwin on nnmc71: the RoboTwin 2.0 evaluation environment (plan.md Phase 6). The evaluation worker runs
# the ImageWAM policy and the SAPIEN simulator in one process, so the environment holds both.
#
# Differences from the upstream installer (ImageWAM/scripts/setup/install_robotwin_env.sh -> RoboTwin script/_install_uv.sh):
# - It installs into a new environment instead of the training .venv.
# - ImageWAM packages equal the .venv freeze, except torch 2.7.1 and torchvision 0.22.1 come from PyPI (CUDA 12.6 builds)
#   instead of the cu118 index: cuRobo v0.7.8 is compiled against the local toolkit (/usr/local/cuda, 12.6), and torch
#   refuses to build extensions across CUDA major versions.
# - RoboTwin requirements are constrained to that freeze so they change no ImageWAM package.
# - No apt step (Vulkan is already present on nnmc71 and nnmc72).
#
#   bash node_env/build_venv_robotwin.sh
set -euo pipefail
P=/home/user1/workspace/jingwu/ImageWAMFT
V="${P}/.venv-robotwin"
ROBOTWIN_DIR="${P}/ImageWAM/third_party/RoboTwin"
WORK="${P}/tmp/venv_robotwin_build"
if [ "$(hostname)" != nnmc71 ]; then
  echo "run on nnmc71" >&2
  exit 2
fi
if [ -e "${V}" ]; then
  echo "${V} exists; remove it first to rebuild" >&2
  exit 2
fi
mkdir -p "${WORK}"
cd "${P}"
uv venv "${V}" --python 3.11

# 1. ImageWAM packages: the .venv freeze, torch and torchvision from PyPI.
uv pip freeze --python .venv/bin/python | grep -v -E '^(-e |torch==|torchvision==|triton==|nvidia-)' > "${WORK}/overrides.txt"
printf 'torch==2.7.1\ntorchvision==0.22.1\n' >> "${WORK}/overrides.txt"
(cd ImageWAM && uv pip install --python "${V}/bin/python" --link-mode copy --no-sources -e '.[shared]' \
  --override "${WORK}/overrides.txt")
uv pip freeze --python "${V}/bin/python" > "${WORK}/freeze_imagewam.txt"

# 2. RoboTwin requirements (script/requirements_mod.txt), constrained to step 1; setuptools follows the upstream pin below.
grep -v -E '^(-e |setuptools==)' "${WORK}/freeze_imagewam.txt" > "${WORK}/constraints.txt"
uv pip install --python "${V}/bin/python" --link-mode copy -c "${WORK}/constraints.txt" \
  -r "${ROBOTWIN_DIR}/script/requirements_mod.txt"
uv pip install --python "${V}/bin/python" --link-mode copy 'setuptools<81'

# 3. The upstream source patches of sapien's URDF loader and mplib's planner.
SITE=$("${V}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "${SITE}/sapien/wrapper/urdf_loader.py"
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
  "${SITE}/mplib/planner.py"

# 4. warp and cuRobo v0.7.8 (cloned into the git-ignored RoboTwin/envs/curobo, as upstream).
uv pip install --python "${V}/bin/python" --link-mode copy warp-lang==1.12.0
if [ ! -d "${ROBOTWIN_DIR}/envs/curobo" ]; then
  git clone https://github.com/NVlabs/curobo.git "${ROBOTWIN_DIR}/envs/curobo"
fi
git -C "${ROBOTWIN_DIR}/envs/curobo" fetch --tags
git -C "${ROBOTWIN_DIR}/envs/curobo" checkout v0.7.8
uv pip install --python "${V}/bin/python" --link-mode copy ninja wheel
(cd "${ROBOTWIN_DIR}/envs/curobo" && PATH="${V}/bin:/usr/local/cuda/bin:${PATH}" CUDA_HOME=/usr/local/cuda \
  TORCH_CUDA_ARCH_LIST="8.9;9.0" MAX_JOBS=32 \
  uv pip install --python "${V}/bin/python" --link-mode copy -e . --no-build-isolation)
uv pip freeze --python "${V}/bin/python" > "${WORK}/freeze_final.txt"
echo finished
