"""Latency and throughput of Chronos-2 against batch size, context length and checkpoint.

Article 5 needs two numbers an accuracy benchmark never reports: how long one ``predict`` call
takes, and how many series per second a machine sustains. Both depend on three things the caller
controls (how many series go in one call, how much context each carries, and which checkpoint is
loaded), so this script sweeps all three and writes one row per combination.

Two deliberate choices:

- **Real data, tiled.** The series are copies of the Spanish load series with distinct ids, not
  random noise. Throughput depends on shapes rather than values, but real data keeps the numbers
  honest if anyone later puts caching or early exit in the loop.
- **Many repeats, the whole distribution reported.** The first call pays one-off costs (CUDA
  context, kernel autotuning) and is discarded. Each cell is then timed ``--repeats`` times (30
  by default) or until ``--budget`` seconds have been spent, whichever comes first, with a floor
  of ``MIN_REPEATS``. The summary row carries the median (the throughput number the articles
  quote), the mean, the standard deviation, the 5th and 95th percentiles, the fastest and the
  slowest call, and the coefficient of variation; every individual timing is also written to
  ``results/<name>_samples.csv``, so the spread can be plotted or re-summarised later.
- **Threads set explicitly, hardware recorded.** PyTorch's CPU thread count is not always the
  core count: a container can leave OpenMP at one thread while ``nproc`` says eight, and the
  first version of this benchmark measured exactly that: 3x slower than the machine could do.
  The script now sets ``torch.set_num_threads`` itself (``--threads``, default: every core the
  process may use) and writes the CPU model, the core count and the thread count into every row.

Usage::

    uv run python bench/bench_cpu_gpu.py --device cpu
    uv run python bench/bench_cpu_gpu.py --device cuda
    uv run python bench/bench_cpu_gpu.py --device cuda --dtype bfloat16 \\
        --name bench_latency_cuda_bf16

Results go to ``results/bench_latency.csv`` (one row per cell) and
``results/bench_latency_samples.csv`` (one row per timed call); ``--fast`` runs a reduced sweep
for a smoke test.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from c2p.config import get_paths, set_seed
from c2p.data import spain_energy_frame, validate_and_prepare
from c2p.experiments import FREQ, LOAD
from c2p.models import CHRONOS2, CHRONOS2_SMALL, DTYPES, load_chronos2, parameter_count
from c2p.results import cores_available, cpu_model, save_results
from c2p.schema import ID_COL, TIMESTAMP_COL

DEFAULT_BATCHES = (1, 8, 32, 128)
DEFAULT_CONTEXTS = (512, 2048)
FAST_BATCHES = (1, 8)
FAST_CONTEXTS = (512,)
QUANTILES = (0.1, 0.5, 0.9)
DEFAULT_REPEATS = 30
DEFAULT_BUDGET_S = 120.0
MIN_REPEATS = 10


def tile_series(series: pd.DataFrame, *, n_series: int, context_length: int) -> pd.DataFrame:
    """``n_series`` copies of the last ``context_length`` steps, each with its own id."""
    tail = series.tail(context_length)
    if len(tail) < context_length:
        raise ValueError(
            f"The series has {len(tail)} steps, fewer than the requested context {context_length}."
        )
    copies = [tail.assign(**{ID_COL: f"s{index:05d}"}) for index in range(n_series)]
    return pd.concat(copies, ignore_index=True)


def time_predict(
    pipeline: Chronos2Pipeline,
    frame: pd.DataFrame,
    *,
    horizon: int,
    repeats: int,
    budget_s: float = float("inf"),
) -> list[float]:
    """Seconds of each of up to ``repeats`` timed calls, after one discarded warm-up.

    Stops early once ``budget_s`` seconds of timed calls have accumulated, but never before
    ``MIN_REPEATS`` calls (or ``repeats`` if that is smaller), so a slow cell such as 128 series
    at 2,048 hours on a CPU does not take an hour while a fast one still gets a full sample.
    """

    def call() -> None:
        pipeline.predict_df(
            frame,
            id_column=ID_COL,
            timestamp_column=TIMESTAMP_COL,
            target=LOAD,
            prediction_length=horizon,
            quantile_levels=list(QUANTILES),
            freq=FREQ,
        )

    def sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    call()
    sync()
    times: list[float] = []
    floor = min(MIN_REPEATS, repeats)
    while len(times) < repeats:
        start = time.perf_counter()
        call()
        sync()
        times.append(time.perf_counter() - start)
        if len(times) >= floor and sum(times) >= budget_s:
            break
    return times


def summarise_times(times: Sequence[float]) -> dict[str, float]:
    """Location, spread and tails of a sample of call durations, in seconds."""
    if len(times) == 0:
        raise ValueError("summarise_times needs at least one timing.")
    sample = np.asarray(times, dtype=float)
    mean = float(sample.mean())
    std = float(sample.std(ddof=1)) if len(sample) > 1 else 0.0
    return {
        "seconds_median": float(np.median(sample)),
        "seconds_mean": mean,
        "seconds_std": std,
        "seconds_p05": float(np.percentile(sample, 5)),
        "seconds_p95": float(np.percentile(sample, 95)),
        "seconds_min": float(sample.min()),
        "seconds_max": float(sample.max()),
        "cv": std / mean if mean > 0 else float("nan"),
        "repeats": len(sample),
    }


def benchmark(
    *,
    device: str,
    models: Sequence[str],
    batches: Sequence[int],
    contexts: Sequence[int],
    horizon: int,
    repeats: int,
    threads: int,
    budget_s: float = DEFAULT_BUDGET_S,
    dtype: str = "float32",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summary rows, one per (model, context, batch), and the raw timing samples behind them."""
    set_seed()
    torch.set_num_threads(threads)
    cpu = cpu_model()
    print(f"CPU {cpu}, {cores_available()} cores available, torch threads {threads}", flush=True)
    raw = spain_energy_frame()[[ID_COL, TIMESTAMP_COL, LOAD]]
    series, _ = validate_and_prepare(raw, freq=FREQ, target=LOAD)
    rows = []
    samples = []
    for model_id in models:
        pipeline = load_chronos2(model_id, device=device, dtype=dtype)
        parameters = parameter_count(pipeline)
        for context_length in contexts:
            for batch in batches:
                frame = tile_series(series, n_series=batch, context_length=context_length)
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                times = time_predict(
                    pipeline, frame, horizon=horizon, repeats=repeats, budget_s=budget_s
                )
                stats = summarise_times(times)
                peak_vram = (
                    torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")
                )
                hardware = {
                    "model": model_id,
                    "params_m": parameters / 1e6,
                    "device": device,
                    "dtype": dtype,
                    "cpu": cpu,
                    "cores_available": cores_available(),
                    "torch_threads": threads,
                    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "",
                    "context_length": context_length,
                    "batch_size": batch,
                    "horizon": horizon,
                }
                rows.append(
                    {
                        **hardware,
                        **stats,
                        "series_per_second": batch / stats["seconds_median"],
                        "series_per_second_p05": batch / stats["seconds_p95"],  # slow calls
                        "series_per_second_p95": batch / stats["seconds_p05"],  # fast calls
                        "peak_vram_mb": peak_vram,
                    }
                )
                samples.extend(
                    {**hardware, "repeat": index, "seconds": seconds}
                    for index, seconds in enumerate(times)
                )
                median = stats["seconds_median"]
                print(
                    f"{model_id:<28} ctx {context_length:>5} batch {batch:>4}: "
                    f"{median * 1000:8.1f} ms (p5 {stats['seconds_p05'] * 1000:7.1f}, "
                    f"p95 {stats['seconds_p95'] * 1000:7.1f}, n={stats['repeats']:>2}) "
                    f"-> {batch / median:8.1f} series/s",
                    flush=True,
                )
        del pipeline
        if device == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows), pd.DataFrame(samples)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line, run the sweep and write the two CSV files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--models", nargs="+", default=[CHRONOS2, CHRONOS2_SMALL])
    parser.add_argument("--batches", nargs="+", type=int, default=None)
    parser.add_argument("--contexts", nargs="+", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"timed calls per cell after the warm-up (default {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET_S,
        help=f"seconds of timed calls per cell before stopping early, floor {MIN_REPEATS} "
        f"calls (default {DEFAULT_BUDGET_S:.0f})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="PyTorch CPU threads (default: every core available to the process)",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPES,
        default="float32",
        help="precision of the weights; bfloat16 or float16 only make sense on a GPU",
    )
    parser.add_argument("--fast", action="store_true", help="reduced sweep, for a smoke test")
    parser.add_argument("--name", default="bench_latency", help="writes results/<name>.csv")
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda but PyTorch cannot see a GPU.")
    batches = args.batches or (FAST_BATCHES if args.fast else DEFAULT_BATCHES)
    contexts = args.contexts or (FAST_CONTEXTS if args.fast else DEFAULT_CONTEXTS)

    results, samples = benchmark(
        device=args.device,
        models=args.models,
        batches=batches,
        contexts=contexts,
        horizon=args.horizon,
        repeats=3 if args.fast else args.repeats,
        threads=args.threads or cores_available(),
        budget_s=args.budget,
        dtype=args.dtype,
    )
    paths = get_paths()
    path = save_results(results, args.name, paths)
    samples_path = save_results(samples, f"{args.name}_samples", paths)
    print(f"\nSaved {path} and {samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
