"""Turn measured throughput and published rates into a cost per million forecasts, with figures.

A benchmark in milliseconds and a price list in dollars per hour are each useless on their own.
This joins them: the per-device benchmark files written by ``bench/bench_cpu_gpu.py`` (one run
per device, so ``results/bench_latency_cuda.csv`` and ``results/bench_latency_cpu.csv``) and
``data/reference/cloud_prices.csv`` (fetched from the AWS Price List API by
``scripts/fetch_cloud_prices.py``).

One honesty constraint shapes the output. Throughput measured on a local machine does **not**
transfer to a cloud instance (different cores, different clocks, different memory bandwidth),
so this script never pretends it does. It reports three things instead:

1. **Rates**: hourly and monthly cost of an always-on endpoint, which is what you actually pay
   whether or not the model is fast.
2. **Cost per million forecasts** at the throughput *we measured*, labelled with the machine it
   was measured on, and with each instance's ``vcpu`` count beside it. The CPU measurement comes
   from one machine with a fixed core count, so a row for an instance with fewer vCPUs is
   optimistic by roughly the ratio of cores; compare like with like. It is an order-of-magnitude
   figure, not a quote.
3. **The break-even ratio**: how many times faster a GPU instance must be than a CPU one before
   it is the cheaper way to serve the same volume. That number depends only on the published
   rates, so it is exact, and it is usually the number that decides the question.

It writes four tables and the two figures of article 5's cost section, so every number and
every pixel in that section comes from committed code rather than from a shell one-liner.

Usage::

    uv run python bench/bench_cpu_gpu.py --device cuda --name bench_latency_cuda
    uv run python bench/bench_cpu_gpu.py --device cpu  --name bench_latency_cpu
    uv run python scripts/fetch_cloud_prices.py
    uv run python scripts/cost_table.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from c2p.config import Paths, get_paths
from c2p.plots.production import plot_cost_per_million, plot_throughput
from c2p.plotting import save_figure, set_style
from c2p.results import load_results, save_results

HOURS_PER_MONTH = 730  # the convention AWS uses in its own pricing pages
FIGURE_CONTEXT = 2048  # the context length the articles backtest with
FIGURE_MODEL = "amazon/chronos-2"
# The CPU throughput behind these costs was measured on an 8-core machine, so the cost figure
# compares it only with 8-vCPU CPU instances; a 2-vCPU rate would be flattered by it.
FIGURE_INSTANCES = (
    "ml.c7i.2xlarge",
    "ml.m5.2xlarge",
    "ml.g4dn.xlarge",
    "ml.g6.xlarge",
    "ml.g5.xlarge",
    "ml.g5.2xlarge",
)
SERVERLESS_MEMORY_GB = 6  # the largest serverless size; Chronos-2 is 119.5M parameters
PRICES_FILE = "cloud_prices.csv"


def load_prices(paths: Paths) -> pd.DataFrame:
    """Read ``data/reference/cloud_prices.csv``; point at ``fetch_cloud_prices.py`` if absent."""
    path = paths.data_reference / PRICES_FILE
    if not path.is_file():
        raise SystemExit(f"{path} is missing. Run: uv run python scripts/fetch_cloud_prices.py")
    return pd.read_csv(path)


def measured_throughput(bench: pd.DataFrame) -> pd.DataFrame:
    """Best sustained throughput per (device, model, context), i.e. at the best batch size."""
    best = bench.loc[
        bench.groupby(["device", "model", "context_length"])["series_per_second"].idxmax()
    ]
    return best[
        [
            "device",
            "model",
            "context_length",
            "batch_size",
            "seconds_median",
            "series_per_second",
            "cpu",
            "torch_threads",
            "gpu",
        ]
    ].reset_index(drop=True)


def hosting_rates(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep the hourly hosting rates and add the monthly cost at ``HOURS_PER_MONTH``."""
    rates = prices[(prices["component"] == "Hosting") & (prices["unit"] == "Hrs")].copy()
    rates["usd_per_month"] = rates["price_usd"] * HOURS_PER_MONTH
    return rates[
        ["instance", "vcpu", "gpu", "memory", "price_usd", "usd_per_month", "region", "checked_on"]
    ].rename(columns={"price_usd": "usd_per_hour"})


def cost_per_million(rates: pd.DataFrame, throughput: pd.DataFrame) -> pd.DataFrame:
    """Cross rates with measured throughput, keeping the measurement's provenance attached."""
    rows = []
    for _, rate in rates.iterrows():
        # The offer file writes "N/A" for CPU instances, which pandas reads as NaN.
        accelerators = pd.to_numeric(rate["gpu"], errors="coerce")
        device = "cuda" if accelerators >= 1 else "cpu"
        matching = throughput[throughput["device"] == device]
        for _, measurement in matching.iterrows():
            per_hour = measurement["series_per_second"] * 3600
            rows.append(
                {
                    "instance": rate["instance"],
                    "vcpu": rate["vcpu"],
                    "usd_per_hour": rate["usd_per_hour"],
                    "usd_per_month": rate["usd_per_month"],
                    "device": device,
                    "model": measurement["model"],
                    "context_length": measurement["context_length"],
                    "batch_size": measurement["batch_size"],
                    "series_per_second": measurement["series_per_second"],
                    "forecasts_per_hour": per_hour,
                    "usd_per_million_forecasts": rate["usd_per_hour"] / per_hour * 1e6,
                    "measured_on": _measured_on(measurement),
                }
            )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["context_length", "model", "usd_per_million_forecasts"], ignore_index=True
    )


def _measured_on(measurement: pd.Series) -> str:
    """Name the hardware behind a throughput row: the GPU, else the CPU and its thread count."""
    gpu = measurement.get("gpu")
    if isinstance(gpu, str) and gpu:  # an empty CSV cell arrives as NaN, which is truthy
        return gpu
    return f"{measurement['cpu']} ({int(measurement['torch_threads'])} threads)"


def break_even(rates: pd.DataFrame, *, cpu_instance: str, gpu_instance: str) -> pd.DataFrame:
    """How many times faster the GPU instance must be before it is the cheaper option."""
    indexed = rates.set_index("instance")["usd_per_hour"]
    missing = [name for name in (cpu_instance, gpu_instance) if name not in indexed.index]
    if missing:
        raise SystemExit(
            f"{missing} are not in the price table. Available: {sorted(indexed.index)}"
        )
    ratio = float(indexed[gpu_instance] / indexed[cpu_instance])
    return pd.DataFrame(
        [
            {
                "cpu_instance": cpu_instance,
                "cpu_usd_per_hour": float(indexed[cpu_instance]),
                "gpu_instance": gpu_instance,
                "gpu_usd_per_hour": float(indexed[gpu_instance]),
                "speedup_to_break_even": ratio,
            }
        ]
    )


def serverless_costs(prices: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    """Cost per million single-series requests on serverless inference, at measured latency.

    Serverless bills per GB-second of *request duration*, so only the batch-size-1 latency
    matters and idle time is free. That is the whole argument for it, and also its limit: one
    request at a time per concurrent worker. SageMaker Serverless Inference is CPU-only, so the
    CPU measurements are the relevant ones.
    """
    per_gb_second = prices.loc[
        (prices["component"] == "ServerlessInference") & (prices["instance"] == "Mem-1GB"),
        "price_usd",
    ]
    if per_gb_second.empty:
        return pd.DataFrame()
    rate = float(per_gb_second.iloc[0])
    single = bench[(bench["batch_size"] == 1) & (bench["device"] == "cpu")]
    rows = []
    for _, measurement in single.iterrows():
        seconds = float(measurement["seconds_median"])
        usd = rate * SERVERLESS_MEMORY_GB * seconds
        rows.append(
            {
                "memory_gb": SERVERLESS_MEMORY_GB,
                "device": measurement["device"],
                "model": measurement["model"],
                "context_length": measurement["context_length"],
                "seconds_per_request": seconds,
                "usd_per_gb_second": rate,
                "usd_per_million_requests": usd * 1e6,
                "measured_on": f"{measurement['cpu']} ({measurement['torch_threads']} threads)",
            }
        )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Join measured throughput with published rates; write the cost tables and figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bench",
        nargs="+",
        default=["bench_latency_cuda", "bench_latency_cpu"],
        help="one results/<name>.csv per device, concatenated",
    )
    parser.add_argument("--cpu-instance", default="ml.c7i.2xlarge")
    parser.add_argument("--gpu-instance", default="ml.g5.xlarge")
    args = parser.parse_args(argv)

    paths = get_paths()
    set_style()
    prices = load_prices(paths)
    bench = pd.concat([load_results(name, paths) for name in args.bench], ignore_index=True)
    throughput = measured_throughput(bench)
    rates = hosting_rates(prices)

    costs = cost_per_million(rates, throughput)
    save_results(rates, "09_hosting_rates", paths)
    save_results(costs, "09_cost_per_million", paths)
    even = break_even(rates, cpu_instance=args.cpu_instance, gpu_instance=args.gpu_instance)
    save_results(even, "09_break_even", paths)
    serverless = serverless_costs(prices, bench)
    if not serverless.empty:
        save_results(serverless, "09_serverless_cost", paths)

    peak = {
        device: bench[
            (bench["device"] == device)
            & (bench["model"] == FIGURE_MODEL)
            & (bench["context_length"] == FIGURE_CONTEXT)
        ]["series_per_second"].max()
        for device in ("cuda", "cpu")
    }
    throughput_ax = plot_throughput(
        bench,
        context_length=FIGURE_CONTEXT,
        title=(
            f"At {FIGURE_CONTEXT:,} hours of context the GPU sustains {peak['cuda']:,.0f} "
            f"series/s against {peak['cpu']:,.0f} on CPU"
        ),
    )
    save_figure(throughput_ax.figure, "09_throughput", paths.figures)
    cost_ax = plot_cost_per_million(
        costs,
        model=FIGURE_MODEL,
        context_length=FIGURE_CONTEXT,
        instances=FIGURE_INSTANCES,
        break_even=even,
        title=(
            f"Cost per million day-ahead forecasts, {FIGURE_MODEL.rsplit('/', maxsplit=1)[-1]} at "
            f"{FIGURE_CONTEXT:,} hours of context"
        ),
    )
    save_figure(cost_ax.figure, "09_cost_per_million", paths.figures)

    checked = rates["checked_on"].iloc[0]
    print(f"Rates for {rates['region'].iloc[0]}, checked {checked}:")
    print(
        rates[["instance", "vcpu", "gpu", "usd_per_hour", "usd_per_month"]].to_string(index=False)
    )
    print(f"\nBreak-even: {args.gpu_instance} must be")
    print(
        f"  {even['speedup_to_break_even'].iloc[0]:.2f}x faster than {args.cpu_instance} "
        f"({even['gpu_usd_per_hour'].iloc[0]:.4f} vs {even['cpu_usd_per_hour'].iloc[0]:.4f} USD/h)"
    )
    print("\nCost per million forecasts at locally measured throughput:")
    columns = [
        "instance",
        "model",
        "context_length",
        "series_per_second",
        "usd_per_million_forecasts",
    ]
    print(costs[columns].round(3).to_string(index=False))
    if not serverless.empty:
        print("\nServerless, one series per request:")
        print(
            serverless[
                ["model", "context_length", "seconds_per_request", "usd_per_million_requests"]
            ]
            .round(3)
            .to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
