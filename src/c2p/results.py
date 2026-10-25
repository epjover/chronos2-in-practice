"""Traceable results: every number in the articles comes from a CSV written here.

Notebooks save each experiment with :func:`save_results` and draw figures from the file read back
with :func:`load_results`, so a figure can never show something the CSV does not contain.
"""

from __future__ import annotations

import importlib.metadata
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from c2p.config import SEED, Paths

_NAME = re.compile(r"^\d{2}_[a-z0-9_]+$|^bench_[a-z0-9_]+$")
_PACKAGES = ("chronos-forecasting", "torch", "pandas", "numpy", "fev", "statsforecast")


def _csv_path(paths: Paths, name: str) -> Path:
    if not _NAME.match(name):
        raise ValueError(
            f"Result name {name!r} must look like '00_experiment' (notebook number, underscore, "
            "lowercase words) or 'bench_experiment'."
        )
    return paths.results / f"{name}.csv"


def save_results(df: pd.DataFrame, name: str, paths: Paths) -> Path:
    """Write ``df`` to ``<results>/<name>.csv`` and return the path."""
    path = _csv_path(paths, name)
    df.to_csv(path, index=False)
    return path


def load_results(name: str, paths: Paths, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV written by :func:`save_results`, parsing ``timestamp``/``cutoff`` as dates."""
    path = _csv_path(paths, name)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist: run the cell that creates it first.")
    df = pd.read_csv(path)
    for column in parse_dates or [c for c in ("timestamp", "cutoff") if c in df.columns]:
        df[column] = pd.to_datetime(df[column])
    return df


def run_metadata(*, device: str, fast_mode: bool, seed: int = SEED) -> dict[str, str]:
    """Versions and settings of the current run, to print at the top of every notebook."""
    meta = {
        "python": platform.python_version(),
        "device": device,
        "fast_mode": str(fast_mode),
        "seed": str(seed),
        "run_at_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
    }
    for package in _PACKAGES:
        try:
            meta[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            meta[package] = "not installed"
    return meta
