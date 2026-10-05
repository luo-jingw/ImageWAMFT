"""The trained G1 ImageWAM runs the server can load, and their saved weights."""

import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"


@dataclass(frozen=True)
class ModelSpec:
    name: str  # identifier used by the web page and the command line
    label: str  # display name
    run_dir: Path  # holds config.yaml, dataset_stats.json and checkpoints/weights/step_*.pt
    stack: str  # "flux2" or "qwenimage21"


MODEL_SPECS = (
    ModelSpec("flux2_4b", "FLUX.2-klein-base-4B", CHECKPOINT_ROOT / "g1_lgg100_flux2_klein_4b_base_imagewam", "flux2"),
    ModelSpec("qwenimage21", "Qwen-Image-2.1", CHECKPOINT_ROOT / "g1_lgg100_qwenimage21_imagewam", "qwenimage21"),
    ModelSpec("flux2_9b", "FLUX.2-klein-base-9B", CHECKPOINT_ROOT / "g1_lgg100_flux2_klein_9b_base_imagewam", "flux2"),
)


def spec_by_name(name: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.name == name:
            return spec
    raise ValueError(f"unknown model {name!r}; known: {[spec.name for spec in MODEL_SPECS]}")


def weights_path(spec: ModelSpec, step: int) -> Path:
    return spec.run_dir / "checkpoints" / "weights" / f"step_{step:06d}.pt"


def available_steps(spec: ModelSpec) -> list[int]:
    """Saved weight steps, ascending; a run still training gains steps over time."""
    pattern = re.compile(r"step_(\d+)\.pt")
    folder = spec.run_dir / "checkpoints" / "weights"
    steps = [int(m.group(1)) for p in folder.glob("step_*.pt") if (m := pattern.fullmatch(p.name))]
    return sorted(steps)
