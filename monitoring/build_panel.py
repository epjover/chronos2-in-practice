"""Replay 2018 as if it were production, and build the monitoring panel from the log.

Monitoring is hard to demonstrate honestly, because a convincing demo needs a year of
predictions made *before* their actuals existed. A rolling-origin backtest is exactly that, run
in fast-forward: at each cutoff the model sees only the past, the forecast is written to a
prediction log with no actual attached, and the actuals are joined afterwards, the same order of
events as a real deployment, with the waiting removed.

What it produces, all under ``results/``:

- ``09_monitoring_log.csv``: one row per forecast step, with model version and input hash.
- ``09_monitoring_rolling.csv``: rolling WAPE and 80% coverage per cutoff, with the alert flag.
- ``09_monitoring_drift.csv``: input drift between the first and last quarter of the year.
- ``results/figures/09_monitoring_panel.png``: the panel an operator would look at.

Usage::

    uv run python monitoring/build_panel.py            # 365 daily cutoffs, ~5 min on a GPU
    uv run python monitoring/build_panel.py --fast     # 12 cutoffs, for a smoke test
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from c2p.config import get_device, get_paths, set_seed
from c2p.data import spain_energy_frame, validate_and_prepare
from c2p.eval import iter_windows
from c2p.experiments import FREQ, LOAD, day_ahead_config
from c2p.models import CHRONOS2, Chronos2Forecaster, load_chronos2
from c2p.monitoring import (
    ALERT_COL,
    CUTOFF_COL,
    MODEL_VERSION_COL,
    input_drift,
    join_actuals,
    log_predictions,
    log_summary,
    retraining_alerts,
    rolling_metrics,
)
from c2p.plotting import (
    AXIS_COLOR,
    BLUE,
    INK_PRIMARY,
    INK_SECONDARY,
    ISSUE_COLOR,
    save_figure,
    set_style,
    style_context,
)
from c2p.results import load_results, save_results
from c2p.schema import ID_COL, TIMESTAMP_COL

MODEL_VERSION = f"{CHRONOS2}@2.3.2"
QUANTILES = (0.1, 0.5, 0.9)
ROLLING_WINDOW = 28  # cutoffs, i.e. four weeks of daily forecasts
WAPE_THRESHOLD = 0.06


def replay(*, fast: bool) -> pd.DataFrame:
    """Forecast every cutoff of 2018 and return the prediction log (no actuals attached)."""
    set_seed()
    paths = get_paths()
    raw = spain_energy_frame()[[ID_COL, TIMESTAMP_COL, LOAD]]
    series, _ = validate_and_prepare(raw, freq=FREQ, target=LOAD)
    config = day_ahead_config(fast, max_context=2048)
    forecaster = Chronos2Forecaster(load_chronos2(device=get_device()))

    rows = []
    for window in iter_windows(series, config):
        forecast = forecaster.predict(
            window.context,
            prediction_length=config.prediction_length,
            quantile_levels=list(config.quantile_levels),
            freq=config.freq,
            target=config.target,
        )
        rows.append(
            log_predictions(
                forecast,
                window.context,
                model_version=MODEL_VERSION,
                cutoff=window.cutoff,
                quantile_levels=QUANTILES,
                extra={"context_steps": len(window.context)},
            )
        )
        if window.index % 50 == 0:
            print(f"  cutoff {window.index + 1}/{config.n_windows}: {window.cutoff}", flush=True)
    log = pd.concat(rows, ignore_index=True)
    print("Saved", save_results(log, "09_monitoring_log", paths).name)
    return series


def panel(rolling: pd.DataFrame, *, threshold: float) -> plt.Figure:
    """Two stacked panels: rolling error with its alert threshold, and interval coverage."""
    nominal = float(rolling["nominal"].iloc[0])
    window = int(rolling["window"].iloc[0])
    with style_context():
        fig, axes = plt.subplots(
            2, 1, figsize=(8.0, 5.0), sharex=True, layout="constrained", height_ratios=(1.2, 1.0)
        )
        breaches = rolling[rolling[ALERT_COL]]
        axes[0].plot(rolling[CUTOFF_COL], rolling["wape"], color=BLUE, zorder=3)
        axes[0].axhline(
            threshold, color=ISSUE_COLOR, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2
        )
        if not breaches.empty:
            axes[0].plot(
                breaches[CUTOFF_COL],
                breaches["wape"],
                linestyle="none",
                marker="o",
                markersize=4.5,
                color=ISSUE_COLOR,
                zorder=4,
            )
        axes[0].annotate(
            f"retraining threshold, WAPE {threshold:.0%}",
            (0.012, threshold),
            xycoords=("axes fraction", "data"),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=9,
            color=ISSUE_COLOR,
            va="bottom",
        )
        axes[0].set_ylabel(f"Rolling WAPE ({window} cutoffs)")
        axes[0].set_title(
            f"{int(rolling[ALERT_COL].sum())} of {len(rolling)} cutoffs above the retraining "
            f"threshold, in {int(rolling['first_breach'].sum())} episodes",
            loc="left",
            color=INK_PRIMARY,
        )
        axes[1].plot(rolling[CUTOFF_COL], rolling["coverage"], color=BLUE, zorder=3)
        axes[1].axhline(
            nominal, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2
        )
        axes[1].annotate(
            f"nominal {nominal:.0%}",
            (0.012, nominal),
            xycoords=("axes fraction", "data"),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=9,
            color=INK_SECONDARY,
            va="bottom",
        )
        axes[1].set_ylabel(f"{nominal:.0%} interval coverage")
        axes[1].set_xlabel("Forecast origin (UTC)")
        for ax in axes:
            ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(AXIS_COLOR)
            ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="12 cutoffs instead of 365")
    parser.add_argument("--threshold", type=float, default=WAPE_THRESHOLD)
    args = parser.parse_args(argv)

    paths = get_paths()
    set_style()
    series = replay(fast=args.fast)

    log = load_results("09_monitoring_log", paths)
    print(log_summary(log))

    joined = join_actuals(log, series, target=LOAD)
    window = 4 if args.fast else ROLLING_WINDOW
    rolling = rolling_metrics(joined, window=window, quantile_levels=(0.1, 0.9), nominal=0.8)
    rolling = retraining_alerts(rolling, wape_threshold=args.threshold)
    print("Saved", save_results(rolling, "09_monitoring_rolling", paths).name)

    cutoffs = sorted(log[CUTOFF_COL].unique())
    split = cutoffs[len(cutoffs) // 4]
    reference = series[series[TIMESTAMP_COL] <= split]
    current = series[series[TIMESTAMP_COL] > cutoffs[-len(cutoffs) // 4]]
    drift = input_drift(reference, current, columns=[LOAD])
    print("Saved", save_results(drift, "09_monitoring_drift", paths).name)
    print(drift.round(4).to_string(index=False))

    rolling = load_results("09_monitoring_rolling", paths)
    (figure_path,) = save_figure(
        panel(rolling, threshold=args.threshold), "09_monitoring_panel", paths.figures
    )
    print(f"Saved {figure_path}")
    print(
        f"\nRolling WAPE over {window} cutoffs: "
        f"min {rolling['wape'].min():.2%}, median {rolling['wape'].median():.2%}, "
        f"max {rolling['wape'].max():.2%}"
    )
    print(
        f"Coverage of the 80% band: min {rolling['coverage'].min():.1%}, "
        f"median {rolling['coverage'].median():.1%}, max {rolling['coverage'].max():.1%}"
    )
    by_version = rolling.groupby(MODEL_VERSION_COL)["wape"].mean()
    print("Mean rolling WAPE per model version:", by_version.round(4).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
