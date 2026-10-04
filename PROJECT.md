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
  - Allowed nodes for this project: `nnmc61`–`nnmc71`.
  - GPU policy (2026-10-01): idle GPUs on nodes `nnmc71`–`nnmc75` may be used directly, at most 4 GPUs per node. Training uses 2 GPUs per model, and GPUs that other users occupy are not used.
  - GPU policy: idle GPUs on `nnmc71`–`nnmc75` may be used, at most 4 per node; GPUs with another user's process are never used. All nodes run jobs from other people under the same `user1` account, so a process counts as this project's only when its command line contains `ImageWAMFT`.
  - `nnmc73`–`nnmc75` have 8× B300 SXM6 AC with 275,040 MiB, compute capability 10.3. `nnmc73`: driver 595.71.05 (CUDA 13.2), no CUDA toolkit in `/usr/local`, Ubuntu 24.04, uv 0.11.28. `nnmc74` and `nnmc75` lack the uv CPython 3.11.15 that `.venv-nnmc73` links to, so that environment runs on `nnmc73` only.
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
| Git remote | repository | none | The repository is local-only. No remote is configured, and the local `.git/hooks/pre-push` hook rejects every push. |

Do not assume a global or default identity applies.
See `AGENTS.md` → Credentials and Environment Isolation.

Do not record token values or other secrets here.

## Project-Specific Constraints

- `dataset/LGG100/` holds the hub data with project task strings. In drawer, biomanual and simple-pick, the `meta/tasks.jsonl` and `meta/episodes.jsonl` `tasks` fields carry the project instructions (`docs/lgg100_dataset.md` → Task Strings). The hub originals are kept in `dataset/LGG100_original_meta/<name>/meta/`. All other dataset content is read-only, and filtered versions are written to a separate location. Re-running `hf download` into `dataset/LGG100/<name>` can restore the hub task strings. These task strings are not pushed to the hub.
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

- Read `AGENTS.md`, this file, `papers/ImageWAM_2606.19531.pdf`, `docs/imagewam_paper_recipe.md` and `docs/lgg100_dataset.md`.
- Activate the environment with `source .venv/bin/activate`. The training launchers call `accelerate` from `PATH`.
- Download from Hugging Face with the project token only: `HF_TOKEN="$(tr -d '[:space:]' < HF_TOKEN)" hf download ...`.
