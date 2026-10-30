"""How stable are the latency numbers? Within-run spread and run-to-run drift, from the CSVs.

Article 5 quotes one throughput figure per cell. This script answers the two questions a
reader should ask of such a figure before trusting it:

1. **Within a run**: across the 30 timed calls of one cell, how far do the slow calls sit from
   the median? Reported as the 5th and 95th percentile of call duration relative to the median
   (``seconds_p05`` and ``seconds_p95`` from ``bench/bench_cpu_gpu.py``), and the coefficient
   of variation.
2. **Between runs**: the whole sweep was run twice on the same machine, minutes apart
   (``bench_latency_<device>.csv`` and ``bench_latency_<device>_rerun.csv``). How much does the
   median of each cell move between the two? That is the error bar on any single-run number,
   and it is the one a reader cannot see in a results table.

Writes ``results/bench_stability.csv`` (one row per cell with both answers) and renders
``results/figures/09_latency_spread.png`` (the within-run spread at 2,048 hours of context).

Usage::

    uv run python bench/summarise_stability.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from c2p.config import get_paths
from c2p.plots.production import plot_latency_spread
from c2p.plotting import save_figure
from c2p.results import load_results, save_results

KEYS = ["model", "device", "context_length", "batch_size"]
FIGURE_CONTEXT = 2048


def stability_table(runs: pd.DataFrame, reruns: pd.DataFrame | None) -> pd.DataFrame:
    """One row per cell: within-run spread, and run-to-run drift when a rerun is available."""
    table = runs[
        [*KEYS, "seconds_median", "seconds_mean", "seconds_p05", "seconds_p95", "cv", "repeats"]
    ].copy()
    table["p05_vs_median"] = table["seconds_p05"] / table["seconds_median"] - 1
    table["p95_vs_median"] = table["seconds_p95"] / table["seconds_median"] - 1
    table["mean_vs_median"] = table["seconds_mean"] / table["seconds_median"] - 1
    if reruns is not None:
        other = reruns[[*KEYS, "seconds_median"]].rename(columns={"seconds_median": "rerun_median"})
        table = table.merge(other, on=KEYS, how="left")
        table["rerun_vs_median"] = table["rerun_median"] / table["seconds_median"] - 1
    return table.sort_values(KEYS, ignore_index=True)


def describe(table: pd.DataFrame) -> pd.DataFrame:
    """Per device: the worst and typical within-run tail, and the largest run-to-run move."""
    rows = []
    for device, group in table.groupby("device", sort=True):
        row = {
            "device": device,
            "cells": len(group),
            "p95_vs_median_median": group["p95_vs_median"].median(),
            "p95_vs_median_max": group["p95_vs_median"].max(),
            "cv_median": group["cv"].median(),
            "cv_max": group["cv"].max(),
        }
        drift = group["rerun_vs_median"].dropna() if "rerun_vs_median" in group else pd.Series()
        if not drift.empty:
            row["rerun_abs_drift_median"] = drift.abs().median()
            row["rerun_abs_drift_max"] = drift.abs().max()
            worst = group.loc[drift.abs().idxmax()]
            row["rerun_worst_cell"] = (
                f"{worst['model'].split('/')[-1]} ctx {worst['context_length']} "
                f"batch {worst['batch_size']}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    parser.add_argument("--prefix", default="bench_latency")
    args = parser.parse_args(argv)
    paths = get_paths()

    runs, reruns = [], []
    for device in args.devices:
        runs.append(load_results(f"{args.prefix}_{device}", paths))
        try:
            reruns.append(load_results(f"{args.prefix}_{device}_rerun", paths))
        except FileNotFoundError:
            print(f"No rerun for {device}: run-to-run drift not available for it.")
    table = stability_table(
        pd.concat(runs, ignore_index=True),
        pd.concat(reruns, ignore_index=True) if reruns else None,
    )
    path = save_results(table, "bench_stability", paths)
    summary = describe(table)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:+.1%}"))

    bench = pd.concat(runs, ignore_index=True)
    ax = plot_latency_spread(
        bench,
        context_length=FIGURE_CONTEXT,
        title=(
            f"Spread of call duration around the median at {FIGURE_CONTEXT:,} hours of context "
            f"(5th to 95th percentile of {int(bench['repeats'].median())} calls)"
        ),
    )
    (figure_path,) = save_figure(ax.figure, "09_latency_spread", paths.figures)
    print(f"\nSaved {path} and {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
