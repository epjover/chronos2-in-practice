"""Traceable results: every number in the articles comes from a CSV written here.

Notebooks save each experiment with :func:`save_results` and draw figures from the file read back
with :func:`load_results`, so a figure can never show something the CSV does not contain.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import subprocess
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


def cpu_model() -> str:
    """The CPU's marketing name (e.g. ``12th Gen Intel(R) Core(TM) i9-12900K``), best effort.

    A timing without the CPU it ran on is not reproducible, and ``platform.processor()`` is
    empty on Linux and a family code on Windows, so this reads the OS's own record instead.
    """
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
            ).strip()
        elif system == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception:  # a missing name must never break a run
        pass
    return platform.processor() or platform.machine() or "unknown"


def cores_available() -> int:
    """CPU cores this process may use (the cgroup/affinity limit, not the socket's count)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # Windows and macOS have no affinity API
        return os.cpu_count() or 1


def hardware_info() -> dict[str, str]:
    """CPU, cores, PyTorch thread count and GPU, so every timing names what it ran on.

    ``torch_threads`` is the number PyTorch will actually use for CPU inference. It is not
    always the core count: containers and sandboxes can leave OpenMP at one thread while
    ``nproc`` says eight, and a benchmark that does not record it is a benchmark of the
    environment, not of the model.
    """
    import torch

    info = {
        "cpu": cpu_model(),
        "cores_available": str(cores_available()),
        "torch_threads": str(torch.get_num_threads()),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info["gpu"] = f"{properties.name} ({properties.total_memory / 2**30:.0f} GiB)"
    else:
        info["gpu"] = "none"
    return info


def run_metadata(*, device: str, fast_mode: bool, seed: int = SEED) -> dict[str, str]:
    """Versions, hardware and settings of the current run, to print at the top of every notebook."""
    meta = {
        "python": platform.python_version(),
        "device": device,
        "fast_mode": str(fast_mode),
        "seed": str(seed),
        "run_at_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
    }
    meta.update(hardware_info())
    for package in _PACKAGES:
        try:
            meta[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            meta[package] = "not installed"
    return meta
