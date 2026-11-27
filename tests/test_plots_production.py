"""Tests for ``c2p.plots.production``: the throughput curve and the cost bars of Part 5."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from c2p.plots.production import (
    device_label,
    plot_cost_per_million,
    plot_coverage_by_month,
    plot_latency_spread,
    plot_throughput,
)

BATCHES = (1, 8, 32, 128)
GPU = "NVIDIA GeForce RTX 3070"


def bench_frame() -> pd.DataFrame:
    rows = []
    for model, factor in (("amazon/chronos-2", 1.0), ("autogluon/chronos-2-small", 3.0)):
        for device, speed in (("cuda", 60.0), ("cpu", 1.0)):
            for context_length in (512, 2048):
                for batch in BATCHES:
                    rows.append(
                        {
                            "model": model,
                            "device": device,
                            "gpu": GPU if device == "cuda" else "",
                            "context_length": context_length,
                            "batch_size": batch,
                            "series_per_second": speed * factor * batch**0.5,
                            "seconds_median": batch / (speed * factor * batch**0.5),
                            "seconds_p05": 0.95 * batch / (speed * factor * batch**0.5),
                            "seconds_p95": 1.20 * batch / (speed * factor * batch**0.5),
                            "series_per_second_p05": speed * factor * batch**0.5 / 1.20,
                            "series_per_second_p95": speed * factor * batch**0.5 / 0.95,
                        }
                    )
    return pd.DataFrame(rows)


def cost_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "instance": instance,
                "vcpu": 8.0,
                "device": device,
                "model": "amazon/chronos-2",
                "context_length": 2048,
                "usd_per_million_forecasts": cost,
            }
            for instance, device, cost in (
                ("ml.g5.xlarge", "cuda", 1.62),
                ("ml.g4dn.xlarge", "cuda", 0.84),
                ("ml.c7i.2xlarge", "cpu", 29.01),
            )
        ]
    )


# --- throughput -----------------------------------------------------------------------------


def test_throughput_draws_a_spread_band_when_percentiles_are_present() -> None:
    ax = plot_throughput(bench_frame(), context_length=2048)
    bands = [c for c in ax.collections if type(c).__name__ == "FillBetweenPolyCollection"]
    assert len(bands) == 4  # one per (model, device)
    plt.close(ax.figure)


def test_throughput_without_percentiles_draws_no_band() -> None:
    bench = bench_frame().drop(columns=["series_per_second_p05", "series_per_second_p95"])
    ax = plot_throughput(bench, context_length=2048)
    bands = [c for c in ax.collections if type(c).__name__ == "FillBetweenPolyCollection"]
    assert bands == []
    plt.close(ax.figure)


# --- latency spread -------------------------------------------------------------------------


def test_latency_spread_marks_p95_relative_to_median() -> None:
    ax = plot_latency_spread(bench_frame(), context_length=2048, title="spread")
    scatters = [c for c in ax.collections if type(c).__name__ == "PathCollection"]
    assert len(scatters) == 4
    highs = np.concatenate([c.get_offsets()[:, 1] for c in scatters])
    assert np.allclose(highs, 20.0)  # p95 is 1.20x the median in the fixture
    assert ax.get_ylabel().startswith("Call duration")
    plt.close(ax.figure)


def test_latency_spread_requires_percentile_columns() -> None:
    with pytest.raises(ValueError, match="missing the columns"):
        plot_latency_spread(bench_frame().drop(columns=["seconds_p95"]), context_length=2048)


def test_throughput_draws_one_line_per_model_and_device():
    ax = plot_throughput(bench_frame(), context_length=2048)
    assert len(ax.lines) == 4
    labels = {text.get_text() for text in ax.get_legend().get_texts()}
    assert labels == {
        "chronos-2 on CPU",
        "chronos-2 on NVIDIA GeForce RTX 3070",
        "chronos-2-small on CPU",
        "chronos-2-small on NVIDIA GeForce RTX 3070",
    }


def test_throughput_plots_only_the_requested_context_length():
    bench = bench_frame()
    ax = plot_throughput(bench, context_length=512)
    drawn = sorted(float(value) for line in ax.lines for value in line.get_ydata())
    expected = sorted(bench.loc[bench["context_length"] == 512, "series_per_second"])
    assert drawn == pytest.approx(expected)


def test_throughput_uses_log_axes_and_ticks_the_measured_batch_sizes():
    ax = plot_throughput(bench_frame(), context_length=2048)
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert tuple(ax.get_xticks()) == BATCHES


def test_throughput_leaves_room_on_the_right_for_the_peak_label():
    ax = plot_throughput(bench_frame(), context_length=2048)
    assert ax.get_xlim()[1] > max(BATCHES)


def test_throughput_rejects_a_context_length_that_was_not_measured():
    with pytest.raises(ValueError, match="No rows with context_length=777"):
        plot_throughput(bench_frame(), context_length=777)


def test_throughput_rejects_a_frame_without_the_benchmark_columns():
    with pytest.raises(ValueError, match="missing the columns"):
        plot_throughput(bench_frame().drop(columns=["series_per_second"]), context_length=512)


def test_device_label_names_the_cpu_and_threads_or_the_gpu():
    assert device_label({"model": "amazon/chronos-2", "device": "cpu", "gpu": ""}) == (
        "chronos-2 on CPU"
    )
    row = {
        "model": "amazon/chronos-2",
        "device": "cpu",
        "gpu": "",
        "cpu": "12th Gen Intel(R) Core(TM) i9-12900K",
        "torch_threads": 8,
    }
    assert device_label(row) == "chronos-2 on Intel Core i9-12900K, 8 threads"
    assert device_label({"model": "a/b-small", "device": "cuda", "gpu": GPU}) == f"b-small on {GPU}"


# --- cost -----------------------------------------------------------------------------------


def test_cost_bars_put_the_cheapest_instance_at_the_top():
    ax = plot_cost_per_million(cost_frame(), model="amazon/chronos-2", context_length=2048)
    # barh counts upward from y=0, so the last label is the top bar.
    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert labels == ["ml.c7i.2xlarge", "ml.g5.xlarge", "ml.g4dn.xlarge"]


def test_cost_bars_can_be_restricted_to_a_comparable_set():
    ax = plot_cost_per_million(
        cost_frame(),
        model="amazon/chronos-2",
        context_length=2048,
        instances=["ml.g5.xlarge", "ml.c7i.2xlarge"],
    )
    assert [label.get_text() for label in ax.get_yticklabels()] == [
        "ml.c7i.2xlarge",
        "ml.g5.xlarge",
    ]


def test_cost_chart_annotates_the_rate_only_break_even():
    break_even = pd.DataFrame(
        [
            {
                "cpu_instance": "ml.c7i.xlarge",
                "gpu_instance": "ml.g5.xlarge",
                "speedup_to_break_even": 6.58,
            }
        ]
    )
    ax = plot_cost_per_million(
        cost_frame(), model="amazon/chronos-2", context_length=2048, break_even=break_even
    )
    texts = [text.get_text() for text in ax.texts]
    assert any("6.6x faster" in text for text in texts)


def test_cost_chart_rejects_an_empty_selection():
    with pytest.raises(ValueError, match="No rows for model="):
        plot_cost_per_million(cost_frame(), model="nope", context_length=2048)
    with pytest.raises(ValueError, match="No rows for model="):
        plot_cost_per_million(
            cost_frame(), model="amazon/chronos-2", context_length=2048, instances=["ml.absent"]
        )


def test_cost_chart_rejects_a_frame_without_the_cost_columns():
    with pytest.raises(ValueError, match="missing the columns"):
        plot_cost_per_million(
            cost_frame().drop(columns=["usd_per_million_forecasts"]),
            model="amazon/chronos-2",
            context_length=2048,
        )


# --- coverage by month ------------------------------------------------------------------------


def by_month_frame() -> pd.DataFrame:
    rows = []
    for model, level in (("zero-shot", 0.73), ("conformal_additive", 0.83)):
        for month in range(1, 13):
            rows.append(
                {
                    "model": model,
                    "period": f"2018-{month:02d}",
                    "coverage": level + 0.01 * (month % 3),
                    "n": 720,
                    "nominal": 0.8,
                }
            )
    return pd.DataFrame(rows)


def test_coverage_by_month_draws_one_line_per_model_plus_the_nominal_level():
    ax = plot_coverage_by_month(by_month_frame())
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == ["nominal 80%", "zero-shot", "conformal_additive"]
    assert len(ax.lines) == 3  # the nominal line and the two models
    assert [label.get_text() for label in ax.get_xticklabels()][:2] == ["2018-01", "2018-02"]


def test_coverage_by_month_rejects_missing_columns_and_empty_input():
    with pytest.raises(ValueError, match="missing the columns"):
        plot_coverage_by_month(by_month_frame().drop(columns=["nominal"]))
    with pytest.raises(ValueError, match="empty"):
        plot_coverage_by_month(by_month_frame().iloc[:0])
