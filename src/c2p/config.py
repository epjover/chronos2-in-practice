"""Project-wide settings: repository paths, fast mode, random seeds and device selection.

Every notebook starts with::

    from c2p.config import get_paths, is_fast_mode, set_seed, get_device

    FAST_MODE = is_fast_mode(default=False)
    PATHS = get_paths(fast_mode=FAST_MODE)
    set_seed()
    DEVICE = get_device()

Environment variables (all optional) override the defaults, which lets CI run every notebook
quickly on CPU without editing them:

- ``C2P_FAST_MODE``: ``1``/``true`` forces fast mode, ``0``/``false`` disables it.
- ``C2P_DEVICE``: ``cpu`` or ``cuda`` forces the device.
- ``C2P_ROOT``: repository root, when it cannot be found by walking up from the working directory.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "GITHUB_REPO",
    "SEED",
    "Paths",
    "find_repo_root",
    "get_device",
    "get_paths",
    "is_fast_mode",
    "set_seed",
]

SEED = 42
GITHUB_REPO = "epjover/chronos2-in-practice"
PROJECT_NAME = "chronos2-in-practice"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root: the closest parent with this project's ``pyproject.toml``.

    ``C2P_ROOT`` takes precedence when set. Searching from the working directory (instead of
    this file's location) keeps it working when the package is installed non-editable, e.g. on
    Colab.
    """
    if env_root := os.environ.get("C2P_ROOT"):
        return Path(env_root).resolve()
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and f'name = "{PROJECT_NAME}"' in pyproject.read_text("utf-8"):
            return candidate
    raise FileNotFoundError(
        f"Could not find the {PROJECT_NAME} repository root above {current}. "
        "Run from inside the repository or set the C2P_ROOT environment variable."
    )


@dataclass(frozen=True)
class Paths:
    """Folders used by the notebooks. Results of fast-mode runs go to ``results/fast/``."""

    root: Path
    data_raw: Path
    data_reference: Path
    results: Path
    figures: Path
    models: Path

    def ensure(self) -> Paths:
        """Create the writable folders (raw data, results, figures, models) if missing."""
        for folder in (self.data_raw, self.results, self.figures, self.models):
            folder.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(fast_mode: bool = False, root: Path | None = None) -> Paths:
    """Build the project paths, keeping fast-mode outputs apart from the published results."""
    root = root or find_repo_root()
    results = root / "results" / "fast" if fast_mode else root / "results"
    return Paths(
        root=root,
        data_raw=root / "data" / "raw",
        data_reference=root / "data" / "reference",
        results=results,
        figures=results / "figures",
        models=root / "models",
    ).ensure()


def _env_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise ValueError(f"{name}={value!r} is not a boolean; use 1/0 or true/false.")


def is_fast_mode(default: bool = False) -> bool:
    """Return the fast-mode flag: ``C2P_FAST_MODE`` if set, otherwise ``default``."""
    flag = _env_flag("C2P_FAST_MODE")
    return default if flag is None else flag


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - third-party code may still use the global RNG
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> str:
    """Return ``"cuda"`` when a GPU is available and wanted, else ``"cpu"``.

    ``C2P_DEVICE`` overrides the choice; asking for ``cuda`` without a GPU raises an error
    instead of silently falling back.
    """
    import torch

    forced = os.environ.get("C2P_DEVICE", "").strip().lower()
    if forced:
        if forced not in {"cpu", "cuda"}:
            raise ValueError(f"C2P_DEVICE must be 'cpu' or 'cuda', got {forced!r}.")
        if forced == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("C2P_DEVICE=cuda but PyTorch cannot see a CUDA GPU.")
        return forced
    return "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"
