"""Model registry: which models the app knows, where their files are, and their validation thresholds."""
import csv
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS_CSV = ROOT / "results" / "tables" / "thresholds_from_validation.csv"

MAX_CHARS = 4000   # prompts are cut to this many characters (as in training)
MAX_TOKENS = 256   # and to this many tokens (as in training)
SUPPORTED_KINDS = {"distilbert"}


@dataclass(frozen=True)
class ModelSpec:
    key: str           # name used in the thresholds file
    label: str         # name shown in the app
    kind: str          # type of model: "distilbert", "qwen_lora" or "qwen_qlora"
    local_dir: Path    # where the files live on this machine
    hub_env_var: str   # environment variable holding a Hugging Face repo id (used if local_dir is missing)
    needs_gpu: bool
    description: str


MODELS = [
    ModelSpec(
        key="DistilBERT", label="DistilBERT", kind="distilbert",
        local_dir=ROOT / "models" / "distilbert_v2", hub_env_var="PROMPTSHIELD_DISTILBERT_REPO",
        needs_gpu=False, description="Small and fast. Runs on a laptop CPU.",
    ),
    ModelSpec(
        key="Qwen LoRA", label="Qwen2.5-1.5B + LoRA", kind="qwen_lora",
        local_dir=ROOT / "models" / "qwen_lora_epoch2", hub_env_var="",
        needs_gpu=True, description="Most accurate in our tests. Needs an NVIDIA GPU.",
    ),
    ModelSpec(
        key="Qwen QLoRA", label="Qwen2.5-1.5B + QLoRA", kind="qwen_qlora",
        local_dir=ROOT / "models" / "qwen_qlora_epoch1", hub_env_var="",
        needs_gpu=True, description="Like the LoRA model, with a smaller memory footprint. Needs an NVIDIA GPU.",
    ),
]


def get_spec(key):
    for spec in MODELS:
        if spec.key == key:
            return spec
    raise KeyError(key)


def load_thresholds():
    """Read the cut-offs that were chosen on the validation set."""
    if not THRESHOLDS_CSV.exists():
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLDS_CSV}")
    thresholds = {}
    with open(THRESHOLDS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            thresholds[row["model"]] = float(row["threshold_1pct_fp"])
    return thresholds


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def source_for(spec):
    """Where to load the model from: a local folder if it exists, otherwise a Hugging Face repo id."""
    if spec.local_dir.exists():
        return str(spec.local_dir)
    repo = os.environ.get(spec.hub_env_var, "") if spec.hub_env_var else ""
    return repo or None


def availability(spec):
    """Return (usable, reason). The reason is shown in the app when a model is unavailable."""
    if spec.needs_gpu and not cuda_available():
        return False, "Needs an NVIDIA GPU, which this machine does not have."
    if spec.kind not in SUPPORTED_KINDS:
        return False, "Not enabled in this version of the app."
    if source_for(spec) is None:
        return False, "Model files were not found on this machine."
    return True, ""