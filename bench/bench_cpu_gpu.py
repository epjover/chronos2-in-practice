"""Latency and throughput of Chronos-2 against batch size, context length and checkpoint.

Article 5 needs two numbers an accuracy benchmark never reports: how long one ``predict`` call
takes, and how many series per second a machine sustains. Both depend on three things the caller
controls — how many series go in one call, how much context each carries, and which checkpoint is
loaded — so this script sweeps all three and writes one row per combination.

Two deliberate choices:

- **Real data, tiled.** The series are copies of the Spanish load series with distinct ids, not
  random noise. Throughput depends on shapes rather than values, but real data keeps the numbers
  honest if anyone later puts caching or early exit in the loop.
- **Median of repeats after a warm-up.** The first call pays one-off costs (CUDA context, kernel
  autotuning) and a shared machine produces outliers, so the median of a few timed runs after one
  discarded warm-up is the number a service would actually see.
- **Threads set explicitly, hardware recorded.** PyTorch's CPU thread count is not always the
  core count — a container can leave OpenMP at one thread while ``nproc`` says eight, and the
  first version of this benchmark measured exactly that: 3x slower than the machine could do.
  The script now sets ``torch.set_num_threads`` itself (``--threads``, default: every core the
  process may use) and writes the CPU model, the core count and the thread count into every row.

Usage::

    uv run python bench/bench_cpu_gpu.py --device cpu
    uv run python bench/bench_cpu_gpu.py --device cuda

Results go to ``results/bench_latency.csv``; ``--fast`` runs a reduced sweep for a smoke test.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import pandas as pd
import torch

from c2p.config import get_paths, set_seed
from c2p.data import spain_energy_frame, validate_and_prepare
from c2p.experiments import FREQ, LOAD
from c2p.models import CHRONOS2, CHRONOS2_SMALL, load_chronos2, parameter_count
from c2p.results import cores_available, cpu_model, save_results
from c2p.schema import ID_COL, TIMESTAMP_COL

DEFAULT_BATCHES = (1, 8, 32, 128)
DEFAULT_CONTEXTS = (512, 2048)
FAST_BATCHES = (1, 8)
FAST_CONTEXTS = (512,)
QUANTILES = (0.1, 0.5, 0.9)


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
    pipeline, frame: pd.DataFrame, *, horizon: int, repeats: int
) -> tuple[float, float]:
    """Median and fastest seconds over ``repeats`` timed calls, after one discarded warm-up."""

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

    call()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2], times[0]


def benchmark(
    *,
    device: str,
    models: Sequence[str],
    batches: Sequence[int],
    contexts: Sequence[int],
    horizon: int,
    repeats: int,
    threads: int,
) -> pd.DataFrame:
    """One row per (model, context, batch): latency, throughput and peak VRAM."""
    set_seed()
    torch.set_num_threads(threads)
    cpu = cpu_model()
    print(f"CPU {cpu}, {cores_available()} cores available, torch threads {threads}", flush=True)
    raw = spain_energy_frame()[[ID_COL, TIMESTAMP_COL, LOAD]]
    series, _ = validate_and_prepare(raw, freq=FREQ, target=LOAD)
    rows = []
    for model_id in models:
        pipeline = load_chronos2(model_id, device=device)
        parameters = parameter_count(pipeline)
        for context_length in contexts:
            for batch in batches:
                frame = tile_series(series, n_series=batch, context_length=context_length)
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                median, fastest = time_predict(pipeline, frame, horizon=horizon, repeats=repeats)
                peak_vram = (
                    torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")
                )
                rows.append(
                    {
                        "model": model_id,
                        "params_m": parameters / 1e6,
                        "device": device,
                        "cpu": cpu,
                        "cores_available": cores_available(),
                        "torch_threads": threads,
                        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "",
                        "context_length": context_length,
                        "batch_size": batch,
                        "horizon": horizon,
                        "seconds_median": median,
                        "seconds_min": fastest,
                        "series_per_second": batch / median,
                        "peak_vram_mb": peak_vram,
                        "repeats": repeats,
                    }
                )
                print(
                    f"{model_id:<28} ctx {context_length:>5} batch {batch:>4}: "
                    f"{median * 1000:8.1f} ms -> {batch / median:8.1f} series/s",
                    flush=True,
                )
        del pipeline
        if device == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--models", nargs="+", default=[CHRONOS2, CHRONOS2_SMALL])
    parser.add_argument("--batches", nargs="+", type=int, default=None)
    parser.add_argument("--contexts", nargs="+", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="PyTorch CPU threads (default: every core available to the process)",
    )
    parser.add_argument("--fast", action="store_true", help="reduced sweep, for a smoke test")
    parser.add_argument("--name", default="bench_latency", help="writes results/<name>.csv")
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda but PyTorch cannot see a GPU.")
    batches = args.batches or (FAST_BATCHES if args.fast else DEFAULT_BATCHES)
    contexts = args.contexts or (FAST_CONTEXTS if args.fast else DEFAULT_CONTEXTS)

    results = benchmark(
        device=args.device,
        models=args.models,
        batches=batches,
        contexts=contexts,
        horizon=args.horizon,
        repeats=1 if args.fast else args.repeats,
        threads=args.threads or cores_available(),
    )
    path = save_results(results, args.name, get_paths())
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
