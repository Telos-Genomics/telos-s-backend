"""
Telos-S shared runtime helpers — device, mask lookup, ESM-2 loader, paths.

Extracted verbatim behavior from variant_comparator.py / mutations_oracle.py
to remove duplication. No scientific logic changed.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))
try:
    from telos_config import BACKEND_ROOT, ESM_DEFAULT_MODEL, OUTPUT_DIR
except ImportError:  # package-style import (pytest / modules.*)
    from modules.telos_config import BACKEND_ROOT, ESM_DEFAULT_MODEL, OUTPUT_DIR


def get_device() -> torch.device:
    """Best available device: CUDA > MPS > CPU (identical messages)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 CUDA detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("🍎 MPS detected (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    return device


def resolve_device(force_cpu: bool = False) -> torch.device:
    """force_cpu=True (CLI --cpu) always wins; else auto-detect."""
    if force_cpu:
        print("💻 CPU forced by user.")
        return torch.device("cpu")
    return get_device()


def find_mask_index(input_ids: torch.Tensor, mask_token_id: int) -> int | None:
    """
    Index of [MASK] in a single sequence. Accepts 1D [seq] (comparator)
    or 2D [1, seq] (oracle) tensors. MPS-safe linear scan.
    """
    flat = input_ids[0] if input_ids.dim() == 2 else input_ids
    for idx in range(flat.shape[0]):
        if flat[idx].item() == mask_token_id:
            return idx
    return None


def load_esm(model_name: str | None = None, device: torch.device | None = None):
    """Load ESM-2 tokenizer + model onto device, CPU fallback. Returns (tok, model, device)."""
    from transformers import EsmForMaskedLM, EsmTokenizer

    name = model_name or os.environ.get("ESM_2_SIZE", ESM_DEFAULT_MODEL)
    print(f"\n📥 Loading model {name}...")
    try:
        tokenizer = EsmTokenizer.from_pretrained(name)
        model = EsmForMaskedLM.from_pretrained(name, torch_dtype=torch.float32)
    except Exception as e:
        print(f"❌ Error loading model components: {e}")
        raise

    device = device or get_device()
    try:
        model = model.to(device)
        print(f"✅ Model loaded onto {device}")
    except Exception as e:
        print(f"⚠️ Failed to move model to {device}: {e}. Falling back to CPU.")
        device = torch.device("cpu")
        model.to(device)
    model.eval()
    return tokenizer, model, device


def backend_path(*parts: str) -> Path:
    """Absolute path under backend root (robust to cwd)."""
    return BACKEND_ROOT.joinpath(*parts)


def output_path(*parts: str) -> Path:
    """Absolute path under backend output/ (robust to cwd)."""
    return OUTPUT_DIR.joinpath(*parts)


def ensure_output_dirs() -> None:
    """Create pipeline output dirs (same set as entrypoint.sh)."""
    for d in ["uploads", "jobs", "s/spike", "s/spike_aligned", "s/reports", "prophet"]:
        (OUTPUT_DIR / d).mkdir(parents=True, exist_ok=True)
