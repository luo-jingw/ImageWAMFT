# PROJECT.md

Project-specific context that supplements `AGENTS.md`.

`AGENTS.md` is fixed across projects.
`PROJECT.md` accumulates what is specific to this one.

Update this file when a new project-specific constraint, environment
detail, or convention is discovered during work.

Do not duplicate:
- verified architecture → `docs/`;
- current work → `plan.md`;
- unresolved problems → `issues.md`.

## Template

- Initialized from (source, version, or commit): explicit-agent 0.1.0 (4d8d5299102a)
- Last synced to (source, version, or commit): explicit-agent 0.1.0 (4d8d5299102a)
- Last applied migration id (see the template's `migrations.json`): 0

## Purpose

This project fine-tunes ImageWAM (arXiv 2606.19531, code in `ImageWAM/`) on the four LGG100 datasets: drawer, biomanual, simple-pick and Stack-the-cubes. They are physical-robot dual-arm teleoperation data from a Unitree G1 29-DoF with retrofitted Dex1-1 grippers and non-standard camera hardware. It compares two image-editing backbones, both trained with the paper's real-world recipe:

- FLUX.2-klein-base-9B, using the existing ImageWAM FLUX.2 integration;
- Qwen-Image-2.1, a 7B single-stream DiT with a Qwen3-VL-8B text encoder, which requires a new backbone integration.

FLUX.2-klein-base-4B, the backbone the paper reports for its real-world experiments, is available as a reference baseline. Trained policies are evaluated on the physical robot.

## Environment

- Platform / hardware: Linux 6.8 with 8× NVIDIA H100 NVL (94 GB each) and driver 580.159.03. The only system CUDA toolkits are 12.4 and 12.6; there is no CUDA 11 toolkit or CUDA 11 NPP library.
- Cluster: this machine is node 71 (`nnmc71`) of a cluster with nodes `nnmc57`–`nnmc76`. All facts below were verified on 2026-10-01 with read-only commands.
  - Shared storage: every node mounts every node's `/home/user1` over NFS4 at `/shared/user<N>`, served from `10.145.87.<N>`. `/shared/` also holds `Datasets`.
  - Project path: on `nnmc71` the project is `/home/user1/workspace/jingwu/ImageWAMFT`. On any node it is `/shared/user71/workspace/jingwu/ImageWAMFT`, which is the same files (identical inode via both paths).
  - Other nodes' `/home/user1` is their own local disk, and the project does not exist there. On other nodes, use only the `/shared/user71/...` path.
  - Access: passwordless `ssh nnmc<N>` works from `nnmc71`, and from other nodes back to `nnmc71` (checked with `nnmc72`).
  - Using another node's GPUs: `ssh nnmc<N>`, then work in `/shared/user71/workspace/jingwu/ImageWAMFT`.
  - Allowed nodes and GPU policy: nodes `nnmc61`–`nnmc71` were granted first; idle GPUs on `nnmc71`–`nnmc75` may be used directly, at most 4 per node. Exceptions granted by the user (`plan.md` D11): the RoboTwin FLUX.2-9B run uses all 8 GPUs of `nnmc71` up to its step-45,000 state (2026-10-08) and GPUs 0–3 after it (2026-10-13); the RoboTwin Qwen-Image-2.1 run uses all 8 GPUs of `nnmc73` (2026-10-08). GPUs with another user's process are never used. Each model trains on 2 GPUs unless its memory needs more: FLUX.2-9B needs one B300 or four H100. All nodes run jobs from other people under the same `user1` account, so a process counts as this project's only when its command line contains `ImageWAMFT`.
  - `nnmc73`–`nnmc75` have 8× B300 SXM6 AC with 275,040 MiB, compute capability 10.3. `nnmc73`: driver 595.71.05 (CUDA 13.2), no CUDA toolkit in `/usr/local`, Ubuntu 24.04, uv 0.11.28. `nnmc74` and `nnmc75` lack the uv CPython 3.11.15 that `.venv-nnmc73` links to, so that environment runs on `nnmc73` only.
  - `nnmc73` NVLink is inactive (checked 2026-10-08): `nvidia-smi nvlink -s -i 0` reports all links inactive, and `nvidia-smi topo -m` shows PCIe paths (NODE within a socket for GPUs 0–3 and 4–7, SYS across) except NV18 between GPUs 3 and 6. Multi-GPU communication therefore runs over PCIe. `nvidia-smi -q` did not return within 2 minutes on this node.
  - A node may need its own Python environment, because GPU models, CUDA toolkits and uv-managed interpreters differ between nodes. Each node gets its own uv environment built with the `.venv` recipe below, and rebuilding per node is acceptable.
  - Hardware of the allowed nodes (read-only survey, 2026-10-01). All have `uv` at `/home/user1/.local/bin/uv` and 128 CPUs, except `nnmc71` with 96:

    | Nodes | GPUs | CUDA toolkits in `/usr/local` |
    |---|---|---|
    | `nnmc71` | 8× H100 NVL, 95,830 MiB | 12.4, 12.6 |
    | `nnmc61`, `nnmc63`, `nnmc64` | 8× L40S, 46,068 MiB | 11.7, 12.1, 12.2, 12.6 |
    | `nnmc62`, `nnmc65` | 8× L40S, 46,068 MiB | 11.7, 12.1 |
    | `nnmc66`, `nnmc67` | 8× L40S, 46,068 MiB | 11.7, 12.1, 12.4 |
    | `nnmc68` | 8× L40S, 46,068 MiB | 11.7 |
    | `nnmc69` | 8× L40S, 46,068 MiB | 12, 12.6 |
    | `nnmc70` | 8× L40S, 46,068 MiB | none: no `nvcc`, so DeepSpeed import fails with `MissingCUDAException: CUDA_HOME does not exist` |

  - The uv-managed CPython that `.venv` links to (`/home/user1/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu`) exists on `nnmc72` but not on `nnmc57`. A node-specific environment therefore needs its own `uv venv`.
  - Nodes `nnmc57`–`nnmc72` report 8 GPUs with 95,830 MiB each. Nodes `nnmc73`–`nnmc76` report per-GPU memory use above 96 GB, so their GPU model differs.
  - GPU availability changes over time; check it with `nvidia-smi` on the target node before launching.
- `nnmc72` (H100 NVL, same uv CPython path as `nnmc71`, system FFmpeg 4, `/usr/local/cuda`) runs the `nnmc71` `.venv` unchanged through the symlink `/home/user1/workspace/jingwu/ImageWAMFT -> /shared/user71/workspace/jingwu/ImageWAMFT`, created on `nnmc72` on 2026-10-01. With it, every absolute path baked into `.venv` resolves; remove the symlink to undo.
- `nnmc70` (8× L40S, 46,068 MiB) runs `.venv-libero` unchanged through the symlink `/home/user1/workspace/jingwu/ImageWAMFT -> /shared/user71/workspace/jingwu/ImageWAMFT`, created on `nnmc70` on 2026-10-05. `nnmc70` has the uv CPython 3.11.13 that the environment links to. Verified: torch 2.7.1+cu118 on L40S (compute capability 8.9), LIBERO with OSMesa rendering. One LIBERO evaluation worker with simulated quantization (about 27 GB) fits per L40S. Remove the symlink to undo.
- `nnmc73` environment, used for the B300 GPUs. Run every command from `/shared/user71/workspace/jingwu/ImageWAMFT`.
  - Python environment: `.venv-nnmc73`, built on `nnmc73` with uv:
    1. `uv venv .venv-nnmc73 --python 3.11` (uses the `nnmc73` uv-managed CPython 3.11.15).
    2. `uv pip install --python .venv-nnmc73/bin/python --link-mode copy torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130`
    3. In `ImageWAM/`: `uv pip install --python ../.venv-nnmc73/bin/python --link-mode copy --no-sources -e '.[shared]' --override ../node_env/nnmc73_overrides.txt`. The override file is the `.venv` freeze without torch, torchvision, torchcodec, triton and `nvidia-*`, plus torch 2.9.1, torchvision 0.24.1 and torchcodec 0.8.1.
    4. `uv pip install --python .venv-nnmc73/bin/python --link-mode copy nvidia-cuda-nvcc==13.0.88`
  - FFmpeg: `.ffmpeg-nnmc73`, a conda prefix built with `/home/user1/miniconda3/bin/conda create -p .ffmpeg-nnmc73 -c conda-forge --override-channels ffmpeg=7.1`. It provides `libavcodec.so.61` and the `libdav1d` AV1 decoder, which torchcodec needs; `nnmc73` has no system FFmpeg.
  - Runtime: `source node_env/nnmc73.sh` sets `PROJECT_ROOT`, the `.venv-nnmc73` interpreter and `PATH`, `CUDA_HOME` (pip nvcc 13.0, which DeepSpeed needs at import), `LD_LIBRARY_PATH` (FFmpeg) and `HF_HUB_OFFLINE=1`.
  - Verified on 2026-10-01:
    - torch 2.9.1+cu130 runs bf16 matmuls on B300. Its arch list has no explicit `sm_103`; the `sm_100` kernels run on compute capability 10.3.
    - DeepSpeed 0.18.5 and imagewam import.
    - torchcodec 0.8.1 decodes the LGG100 AV1 videos.
    - All 36,188 installed files match their RECORD hashes.
    - Other versions equal `.venv`: transformers 4.56.1, huggingface-hub 0.36.2, accelerate 1.12.0, numpy 1.26.4, opencv-python-headless 4.11.0.86.
- B300 environment for `nnmc73`–`nnmc75`: `.venv-b300`. Run every command from `/shared/user71/workspace/jingwu/ImageWAMFT`.
  - Interpreter: `.uv-python/cpython-3.11.15-linux-x86_64-gnu`, a copy of the uv-managed CPython 3.11.15 from `nnmc73` (`/shared/user73/.local/share/uv/python/`). It lives in the project, so the environment resolves on every node through `/shared/user71`; `nnmc74` has no uv and `nnmc75` has no CPython 3.11.15 of its own.
  - Build: `node_env/build_venv_b300.sh`, run on `nnmc75` with the `nnmc73` uv 0.11.28 (`/shared/user73/.local/bin/uv`) and the uv cache in `tmp/uv-cache-b300`; same steps 2–4 as `.venv-nnmc73`.
  - Runtime: `source node_env/b300.sh`, the `nnmc73.sh` settings with `.venv-b300`.
  - Verified on 2026-10-03: `uv pip freeze` equals `.venv-nnmc73` (120 packages); on `nnmc74` and `nnmc75`, torch 2.9.1+cu130, DeepSpeed 0.18.5 and imagewam import, torchcodec decodes the LGG100 AV1 videos, and the DeepSpeed patch applies. All three nodes run driver 595.71.05 on Ubuntu 24.04 (glibc 2.39). The GPU path is the same torch build verified on `nnmc73`.
- RoboTwin 2.0 evaluation environment: `.venv-robotwin` (`nnmc71`, `nnmc72`). The evaluation worker runs the policy and the SAPIEN simulator in one process.
  - Build: `node_env/build_venv_robotwin.sh` on `nnmc71`. It follows the upstream installer (`ImageWAM/scripts/setup/install_robotwin_env.sh` → RoboTwin `script/_install_uv.sh`) but into its own environment, which the upstream installer does not do.
  - Packages: the `.venv` freeze, except that torch 2.7.1 and torchvision 0.22.1 are the PyPI CUDA 12.6 builds. cuRobo v0.7.8 is compiled against `/usr/local/cuda` 12.6, and torch does not build extensions across CUDA major versions.
  - It adds the RoboTwin requirements (constrained to that freeze; sapien 3.0.0b1, mplib 0.2.1), `setuptools<81`, the upstream sapien and mplib source patches, warp-lang 1.12.0 and cuRobo v0.7.8. cuRobo is cloned into the git-ignored `ImageWAM/third_party/RoboTwin/envs/curobo` and built for compute capability 8.9 and 9.0. The aloha-agilex embodiment plans with cuRobo.
  - The serving packages of `.venv` (gradio, websockets) are not installed.
  - Assets: `TianxingChen/RoboTwin2.0` (`background_texture`, `embodiments`, `objects`; 16 GB) are unpacked in the git-ignored `ImageWAM/third_party/RoboTwin/assets/`. `script/update_embodiment_config_path.py` wrote the absolute `/home/user1/workspace/jingwu/ImageWAMFT/...` asset paths into the embodiment configs; they resolve on `nnmc72` through its symlink.
  - Vulkan: `vulkaninfo` lists the H100 GPUs on `nnmc71` and `nnmc72`.
  - The tracked symlink `ImageWAM/third_party/RoboTwin/policy/imagewam_policy` pointed to the authors' cluster path; it now points to `../../../experiments/robotwin/imagewam_policy`.
- Qwen3-VL text features on B300: `.venv-qwenimage-nnmc73` (`nnmc73`).
  - Built by `node_env/build_venv_qwenimage_nnmc73.sh` with the `nnmc73` uv CPython 3.11.15.
  - Packages: the `.venv-qwenimage` freeze, except torch 2.9.1 and torchvision 0.24.1 for CUDA 13.0, the `.venv-nnmc73` torch build. The `.venv-qwenimage` torch (2.7.1+cu118) does not support compute capability 10.3. Its non-torch packages equal `.venv-qwenimage`.
  - For the same prompt, its features differ from `.venv-qwenimage` on an H100 by 2.2–2.6% relative L2 (cosine ≥ 0.9997, identical masks; 12 RoboTwin prompts, 2026-10-08).
- Qwen-Image-2.1 reference environment: `.venv-qwenimage` on `nnmc71`. It has torch 2.7.1+cu118, diffusers from git commit `578c9b2c6636ab2424a0e56186268b83623656b2` (0.41.0.dev0), transformers 5.18.0 and huggingface-hub 1.33.0, installed with `--link-mode copy`. It is used offline only, for reference fixtures and Qwen3-VL text features.
- Running `.venv` from another node (paths written at build time point to `/home/user1/workspace/jingwu/ImageWAMFT`):
  - The interpreter is the uv-managed CPython 3.11.13 at `/home/user1/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/bin/python3.11`. It exists at the same path on `nnmc72`.
  - Console scripts (`.venv/bin/accelerate`, `.venv/bin/torchrun`) have the shebang `/home/user1/workspace/jingwu/ImageWAMFT/.venv/bin/python`, which does not resolve on other nodes. Invoke the tools through `/shared/user71/workspace/jingwu/ImageWAMFT/.venv/bin/python -m accelerate.commands.launch` or `-m torch.distributed.run`.
  - The editable install (`__editable__.imagewam-0.1.0.pth`) points to `/home/user1/workspace/jingwu/ImageWAMFT/ImageWAM/src`. The launchers prepend `${REPO_ROOT}/src` to `PYTHONPATH`, and `REPO_ROOT` resolves from the script location, so `imagewam` imports when scripts are called through `/shared/user71/...`.
  - `ImageWAM/.env.local` defaults `PROJECT_ROOT` to the `nnmc71` path. On other nodes, set `PROJECT_ROOT=/shared/user71/workspace/jingwu/ImageWAMFT` in the shell first.
- Shared with (other users or projects on the same account or machine): other projects run under the same Linux account. The uv cache (`~/.cache/uv`) and the machine-wide Hugging Face token (`~/.cache/huggingface/token`) are shared across them. Packages that uv installs from the cache are hard-linked to cache files that other environments also link to.
- Constraints imposed by the environment:
  - Python environment: `.venv` (Python 3.11.13). It is managed through `uv venv` / `uv pip`; `uv sync`, `uv lock` and `uv run` are not used. `ImageWAM/uv.lock` is stale relative to `ImageWAM/pyproject.toml`: it has no `shared` extra and pins older transformers and huggingface-hub. Re-locking fails while building `mmcv-full` from the `dim` extra.
  - Build recipe for `.venv`:
    1. `uv venv .venv --python 3.11`
    2. Inside `ImageWAM/`: `uv pip install --python ../.venv/bin/python -e ".[shared]"`. Running inside `ImageWAM/` makes uv use its cu118 torch index.
    3. `uv pip install --python .venv/bin/python transformers==4.56.1` (the FLUX.2 requirement).
    4. `uv pip install --python .venv/bin/python torchcodec==0.5 --index-url https://pypi.org/simple --reinstall-package torchcodec`
    5. `uv pip install --python .venv/bin/python opencv-python-headless==4.11.0.86 --reinstall-package opencv-python-headless --no-cache --link-mode copy`
    6. Serving dependencies for the G1 policy server: `uv pip install --python .venv/bin/python --link-mode copy -c <constraints> gradio==6.17.3 websockets==17.2`. `<constraints>` is the `.venv` freeze before this step, without the editable `imagewam` line. This adds 23 packages and changes none. gradio 4 and 5 require pillow < 12 and conflict with pillow 12.0.
  - Resulting key versions: torch 2.7.1+cu118, torchvision 0.22.1+cu118, torchcodec 0.5 (CPU build), transformers 4.56.1, huggingface-hub 0.36.2, diffusers 0.34.0, deepspeed 0.18.5, accelerate 1.12.0, numpy 1.26.4.
  - `uv pip check` reports two known conflicts: `imagewam` pins `transformers==4.51.3` and `huggingface-hub==0.30.0`. Both come from the FLUX.2 transformers upgrade.
  - The cu118 build of torchcodec needs `libnppicc.so.11`, which is absent, so the CPU build is installed. ImageWAM decodes video on CPU only. Re-running step 2 can bring back the cu118 build.
  - The shared uv cache copy of `opencv-python-headless==4.11.0.86` was modified in place by another environment through a hard link, so a cached install yields a `cv2` that fails to import (`libpng16-ef62451c.so.16.44.0` missing). That package must be installed with `--no-cache`.
  - DeepSpeed JIT ops cannot be built: nvcc is 12.6 while torch is cu118. The DeepSpeed configs in `ImageWAM/scripts/ds_configs/` define no optimizer, so training uses torch AdamW and needs no JIT build.

## Credentials

Services that require a scoped identity in this project (git remotes,
package registries, model hubs, cloud APIs).

| Service | Scope | Source | Notes |
|---|---|---|---|
| Hugging Face Hub | project | `HF_TOKEN` file at the project root, passed per command as `HF_TOKEN="$(tr -d '[:space:]' < HF_TOKEN)"` | Account JingwuLuo, fine-grained token. It has access to the gated `black-forest-labs/FLUX.2-klein-base-9B`, `FLUX.2-klein-base-4B` and `FLUX.2-dev`. The machine-wide default token belongs to a different account and is not used. |
| Git commit identity | repository | repository-local `.git/config`: `user.name` Jingwu Luo, `user.email` grouponeclaude@gmail.com | The machine-wide git identity is a shared-machine default and is not used. |
| Git remote | repository | `https://github.com/luo-jingw/ImageFT` (public); token file `/home/user1/workspace/jingwu/github_token` (outside the repository), read per command by a credential helper in `launch/publish_github.sh` | Publishing runs only through `launch/publish_github.sh`. It pushes a scratch-clone copy of `main` with `papers/` removed from every commit (the arXiv PDF may not be redistributed), so published hashes differ from the local ones. The local repository has no remote configured, and its `.git/hooks/pre-push` hook still rejects direct pushes. |

Do not assume a global or default identity applies.
See `AGENTS.md` → Credentials and Environment Isolation.

Do not record token values or other secrets here.

## Project-Specific Constraints

- `dataset/LGG100/` holds the hub data with project task strings. In drawer, biomanual and simple-pick, the `meta/tasks.jsonl` and `meta/episodes.jsonl` `tasks` fields carry the project instructions (`docs/lgg100_dataset.md` → Task Strings). The hub originals are kept in `dataset/LGG100_original_meta/<name>/meta/`. All other dataset content is read-only, and filtered versions are written to a separate location. Re-running `hf download` into `dataset/LGG100/<name>` can restore the hub task strings. These task strings are not pushed to the hub.
- RoboTwin 2.0 data (`docs/robotwin_training_setup.md`):
  - `dataset/robotwin2.0/robotwin2.0/` holds the FastWAM LeRobot release (`yuanty/robotwin2.0-fastwam`) and the generated `nonidle_ranges.json`. `dataset/robotwin2.0/dataset_stats.json` is the dataset's normalization file.
  - `nnmc73` keeps a byte-identical copy and the RoboTwin runs' large files on its local disk under `/home/user1/workspace/jingwu/ImageWAMFT_nnmc73/` (`/shared/user73/workspace/jingwu/ImageWAMFT_nnmc73/` from other nodes):
    - `dataset/robotwin2.0/`;
    - `text_cache/flux2_qwen3_cache_9b/` (Qwen3-8B, 2.9 TB);
    - `text_cache/qwenimage21_text_cache/` (Qwen3-VL, 0.97 TB);
    - `checkpoints/` (the Qwen-Image-2.1 RoboTwin run).
  - The released FLUX.2-4B RoboTwin model is in `checkpoints/imagewam_release/robotwin/flux2_klein_4b/`.
- Base weights live under `BaseModel/`:
  - `flux2/FLUX.2-klein-base-9B/flux-2-klein-base-9b.safetensors`
  - `flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors`
  - `flux2/FLUX.2-dev/ae.safetensors`
  - `Qwen3-8B/`: text encoder for 9B
  - `Qwen3-4B/`: text encoder for 4B
  - `Qwen-Image-2.1/`: full diffusers pipeline with transformer, Qwen3-VL-8B text_encoder, vae, processor and scheduler
- `ImageWAM/` is a vendored copy of `https://github.com/yuyangalin/ImageWAM.git` at commit `5d4a341ed20a95cdb08f0293f3d44778b9a9e05a`, tracked as plain files in this repository. All 940 files were verified identical to that commit at import, and project changes are committed here. The upstream history is kept, git-ignored, at `tmp/imagewam_upstream_5d4a341.git`. `ImageWAM/.gitignore` ignores `*.txt`, so new `.txt` files under `ImageWAM/` need `git add -f`.
- `ImageWAM/third_party/flux2/` is a vendored copy of `https://github.com/black-forest-labs/flux2` at commit `50fe5162777813d869182b139e83b10743caef15` (code under Apache-2.0). It was taken with `git archive`, and the `assets/` images were left out. Training puts `third_party/flux2/src` on `PYTHONPATH` through `FLUX2_SRC`.
- `ImageWAM/.env.local` (git-ignored) is sourced by `ImageWAM/scripts/common.sh`. It sets:
  - `PROJECT_ROOT`, `DATA_ROOT=<project>/dataset`, `MODEL_ROOT=<project>/BaseModel`, `PYTHON_BIN=<project>/.venv/bin/python`;
  - `HF_HUB_OFFLINE=1`;
  - `FLUX2_SRC`, `FLUX2_VARIANT=9b`, `FLUX2_AE_MODEL_PATH`, `ZERO_STAGE=2`;
  - `FLUX2_QWEN3_MODEL_SPEC`: the local `Qwen3-8B` for `9b`, the local `Qwen3-4B` for `4b`.
  Values set in the calling shell take precedence.
- FLUX.2 weights are under the FLUX Non-Commercial License. Qwen-Image-2.1 is under the Qwen Research License.
- The FLUX.2 path requires `transformers==4.56.1`. The reference implementation of Qwen-Image-2.1 is only in diffusers git main (from commit `6256aa7`, version 0.41.0.dev0, which requires `huggingface-hub>=1.32`). Its Qwen3-VL text encoder needs transformers ≥ 4.57, and with that hub constraint transformers 5.x. Neither fits the `.venv` stack.

## Onboarding

- Read `AGENTS.md`, this file, the ImageWAM paper (arXiv 2606.19531; a local copy is `papers/ImageWAM_2606.19531.pdf`, git-ignored), `docs/imagewam_paper_recipe.md` and `docs/lgg100_dataset.md`.
- Activate the environment with `source .venv/bin/activate`. The training launchers call `accelerate` from `PATH`.
- Download from Hugging Face with the project token only: `HF_TOKEN="$(tr -d '[:space:]' < HF_TOKEN)" hf download ...`.
