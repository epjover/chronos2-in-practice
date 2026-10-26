"""Fix the one defect every article found: calibrate Chronos-2's intervals and re-score 2018.

Parts 2 to 5 all report the same thing — the 80% band covers about 73% of the hours on Spanish
load, and far fewer in winter — and every lever they pull targets the median. This script pulls
the one lever that targets the bands, with the method that carries a finite-sample guarantee
and needs no training: conformalized quantile regression (``c2p.conformal``).

Protocol, chosen so that nothing here could have seen 2018:

1. **Calibrate** on the second half of 2017 — the period article 4 used for early stopping —
   with the same day-ahead windows (24 h, cutoff 22:00 UTC, 2,048 h of context): 184 windows,
   4,416 hours of misses per quantile pair and horizon step.
2. **Evaluate** on article 2's 365 windows of 2018, zero-shot against the two calibrated
   variants (additive shift, scaled multiplier), through the same harness and the same metrics.
3. **Look by month**, because a yearly coverage of 80% can still be 60% every January.

Outputs under ``results/``: ``09_conformal_adjustments.csv`` (the learned deltas),
``09_conformal_metrics.csv`` (per window), ``09_conformal_summary.csv``,
``09_conformal_by_month.csv`` and ``figures/09_conformal_coverage.png``.

Usage::

    uv run python monitoring/calibrate_intervals.py          # ~2 min on a GPU
    C2P_FAST_MODE=1 uv run python monitoring/calibrate_intervals.py
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from c2p.config import get_device, get_paths, is_fast_mode, set_seed
from c2p.conformal import ConformalForecaster, calibrate, coverage_by_period
from c2p.data import spain_energy_frame, validate_and_prepare
from c2p.eval import MODEL_COL, backtest, evaluate, skill_table
from c2p.experiments import FREQ, LOAD, VALIDATION_END, day_ahead_config
from c2p.models import Chronos2Forecaster, load_chronos2
from c2p.plots.production import plot_coverage_by_month
from c2p.plotting import save_figure, set_style
from c2p.results import save_results
from c2p.schema import ID_COL, TIMESTAMP_COL

CONTEXT = 2048
CALIBRATION_WINDOWS = 184  # daily cutoffs from 2017-06-30 22:00 to 2017-12-30 22:00


def main() -> int:
    set_seed()
    set_style()
    fast = is_fast_mode(default=False)
    paths = get_paths(fast_mode=fast)
    raw = spain_energy_frame()[[ID_COL, TIMESTAMP_COL, LOAD]]
    series, _ = validate_and_prepare(raw, freq=FREQ, target=LOAD)
    test_config = day_ahead_config(fast, max_context=CONTEXT)
    base = Chronos2Forecaster(load_chronos2(device=get_device()), name="zero-shot")

    # 1. Calibrate on 2017-H2: the series is cut at the validation end, so the harness's
    #    "last horizon ends at the end of the data" rule places every window before 2018.
    calibration_series = series[series[TIMESTAMP_COL] <= VALIDATION_END]
    calibration_config = replace(test_config, n_windows=12 if fast else CALIBRATION_WINDOWS)
    calibration_frame = backtest(base, calibration_series, calibration_config)
    assert calibration_frame[TIMESTAMP_COL].max() <= VALIDATION_END
    print(
        f"Calibration: {calibration_config.n_windows} windows, "
        f"{calibration_frame['cutoff'].min()} to {calibration_frame['cutoff'].max()}"
    )
    calibrations = {
        mode: calibrate(calibration_frame, quantile_levels=test_config.quantile_levels, mode=mode)
        for mode in ("additive", "scaled")
    }
    adjustments = pd.concat(
        [c.table.assign(mode=mode) for mode, c in calibrations.items()], ignore_index=True
    )
    print("Saved", save_results(adjustments, "09_conformal_adjustments", paths).name)
    wide = calibrations["additive"].delta(0.1, 0.9)
    print(
        f"Additive shift for the 80% band: {wide.min():,.0f} MW at step {wide.idxmin()} "
        f"to {wide.max():,.0f} MW at step {wide.idxmax()}"
    )

    # 2. Evaluate on 2018, all three through the same harness.
    models = [base] + [
        ConformalForecaster(base, calibration, name=f"conformal_{mode}")
        for mode, calibration in calibrations.items()
    ]
    forecasts, metrics = [], []
    for model in models:
        frame = backtest(model, series, test_config)
        forecasts.append(frame)
        scored = evaluate(frame, series, test_config)
        metrics.append(scored)
        print(
            f"  {model.name:<20} WQL {scored['WQL'].mean():.4f}  "
            f"cov80 {scored['coverage_80'].mean():.1%}  cov98 {scored['coverage_98'].mean():.1%}"
        )
    metrics = pd.concat(metrics, ignore_index=True)
    print("Saved", save_results(metrics, "09_conformal_metrics", paths).name)

    summary = (
        metrics.groupby(MODEL_COL)
        .agg(
            WQL=("WQL", "mean"),
            MASE=("MASE", "mean"),
            coverage_80=("coverage_80", "mean"),
            coverage_98=("coverage_98", "mean"),
            n_windows=("WQL", "size"),
        )
        .reset_index()
    )
    skill = skill_table(metrics, baseline="zero-shot", metric="WQL")
    summary = summary.merge(
        skill[[MODEL_COL, "skill", "ci_low", "ci_high", "win_rate"]], on=MODEL_COL
    )
    print("Saved", save_results(summary, "09_conformal_summary", paths).name)

    # 3. By month: the diagnostic the yearly number hides.
    all_forecasts = pd.concat(forecasts, ignore_index=True)
    by_month = coverage_by_period(all_forecasts, lower=0.1, upper=0.9, period="M")
    # The first window's horizon starts at 23:00 UTC on 31 December 2017: one hour of a month
    # that is otherwise not in the test year. A month needs at least a week of hours to count.
    by_month = by_month[by_month["n"] >= 24 * 7].reset_index(drop=True)
    print("Saved", save_results(by_month, "09_conformal_by_month", paths).name)
    worst = by_month.loc[by_month.groupby(MODEL_COL)["coverage"].idxmin()]
    ax = plot_coverage_by_month(
        by_month,
        title=(
            "80% band, coverage by month of 2018: zero-shot against conformal calibration "
            "on 2017-H2"
        ),
    )
    (figure_path,) = save_figure(ax.figure, "09_conformal_coverage", paths.figures)
    print(f"Saved {figure_path}")

    print("\n" + summary.round(4).to_string(index=False))
    print("\nWorst month per model (80% band):")
    print(worst[[MODEL_COL, "period", "coverage"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
