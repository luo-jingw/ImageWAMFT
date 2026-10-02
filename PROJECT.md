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
- FLUX.2 weights are under the FLUX Non-Commercial License. Qwen-Image-2.1 is under the Qwen Research License.
- The FLUX.2 path requires `transformers==4.56.1`. The reference implementation of Qwen-Image-2.1 is only in diffusers git main (from commit `6256aa7`, version 0.41.0.dev0, which requires `huggingface-hub>=1.32`). Its Qwen3-VL text encoder needs transformers ≥ 4.57, and with that hub constraint transformers 5.x. Neither fits the `.venv` stack.

## Onboarding

- Read `AGENTS.md`, this file, `papers/ImageWAM_2606.19531.pdf`, `docs/imagewam_paper_recipe.md` and `docs/lgg100_dataset.md`.
- Activate the environment with `source .venv/bin/activate`.
- Download from Hugging Face with the project token only: `HF_TOKEN="$(tr -d '[:space:]' < HF_TOKEN)" hf download ...`.
