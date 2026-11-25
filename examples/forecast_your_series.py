r"""Forecast your own CSV with Chronos-2: checklist, zero-shot quantiles, fan chart, backtest.

The notebooks all start from the Spanish electricity dataset. This script is the same pipeline
pointed at a file of yours, so the first forecast on your data is one command rather than a
read of seven articles:

    uv run python examples/forecast_your_series.py data.csv \\
        --id-column store --timestamp-column date --target sales --freq D --horizon 14

What happens, in the order of the series:

1. **Checklist** (article 1): columns are renamed to the ``item_id`` / ``timestamp`` schema,
   timestamps become naive UTC, sentinels become NaN, duplicates are resolved, every series is
   put on a regular grid, and the report says what was found. Nothing is imputed.
2. **Zero-shot forecast** (article 1): Chronos-2 forecasts ``--horizon`` steps for every series
   at the requested quantiles, and the forecast is written next to your CSV with a fan chart
   of the first series.
3. **Backtest** (article 2, optional): with ``--backtest N``, the last ``N`` horizons of your
   data are replayed as rolling-origin windows against a seasonal naive baseline, and the skill
   score with its bootstrap interval tells you whether the model beats the obvious baseline on
   your series. Needs at least ``N + 1`` horizons of history per series.

Known-future covariates (article 3) are supported through ``--known``: name the columns you
will genuinely have for the horizon; for the zero-shot forecast they are read from the rows of
your CSV that lie beyond the last target value, so append those rows (target left empty) before
running.

A fine-tuned checkpoint from article 4 drops in with ``--model path/to/checkpoint``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from c2p.baselines import SeasonalNaive
from c2p.config import get_device
from c2p.data import check_future_frame, validate_and_prepare
from c2p.eval import BacktestConfig, backtest, evaluate, skill_table
from c2p.models import CHRONOS2, Chronos2Forecaster, load_chronos2
from c2p.plotting import plot_fan, save_figure, set_style
from c2p.schema import ID_COL, TIMESTAMP_COL

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


def load_frame(
    path: Path,
    *,
    id_column: str | None,
    timestamp_column: str,
    target: str,
    known: Sequence[str],
    freq: str,
    source_tz: str | None,
    sentinels: Sequence[float],
) -> tuple[pd.DataFrame, object]:
    """Read the CSV, map it onto the ``item_id`` / ``timestamp`` schema and run the checklist."""
    raw = pd.read_csv(path)
    for column in [timestamp_column, target, *known]:
        if column not in raw.columns:
            raise SystemExit(
                f"Column {column!r} is not in {path.name}. Columns: {list(raw.columns)}"
            )
    if id_column is None:
        raw[ID_COL] = path.stem  # a single series: the file name is as good an id as any
    elif id_column not in raw.columns:
        raise SystemExit(f"Column {id_column!r} is not in {path.name}.")
    else:
        raw = raw.rename(columns={id_column: ID_COL})
    raw = raw.rename(columns={timestamp_column: TIMESTAMP_COL})
    raw[TIMESTAMP_COL] = pd.to_datetime(raw[TIMESTAMP_COL])
    frame = raw[[ID_COL, TIMESTAMP_COL, target, *known]]
    return validate_and_prepare(
        frame, freq=freq, target=target, source_tz=source_tz, sentinels=tuple(sentinels)
    )


def split_future(
    frame: pd.DataFrame, *, target: str, known: Sequence[str], horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Rows after each series' last observed target are the known-future covariates.

    Returns the history (up to the last observed target of each series) and, when the CSV has
    rows beyond it, exactly ``horizon`` future rows per series. Series whose future is shorter
    than the horizon (a trailing NaN or sentinel in the target shortens the history, which is a
    common way to get here) are reported and the run stops, because the model needs the full
    horizon of covariates for every series. With no future rows at all, ``None`` is returned and
    the caller falls back to past-only covariates.
    """
    if not known:
        return frame, None
    observed = frame[frame[target].notna()]
    last = observed.groupby(ID_COL)[TIMESTAMP_COL].max().rename("last_observed")
    merged = frame.merge(last, on=ID_COL)
    is_future = merged[TIMESTAMP_COL] > merged["last_observed"]
    history = merged[~is_future].drop(columns="last_observed")
    future = merged[is_future].drop(columns="last_observed")
    if future.empty:
        return history, None
    counts = future.groupby(ID_COL).size().reindex(history[ID_COL].unique(), fill_value=0)
    short = counts[counts < horizon]
    if not short.empty:
        listing = ", ".join(f"{item}: {n}" for item, n in short.items())
        raise SystemExit(
            f"--known needs {horizon} future rows per series but some have fewer ({listing}). "
            "Append the missing rows with the target left empty, check for trailing sentinels "
            "or NaNs in the target, or drop --known to use the covariates as past-only."
        )
    future = future.groupby(ID_COL, group_keys=False).head(horizon)
    return history, future[[ID_COL, TIMESTAMP_COL, *known]]


def main(argv: Sequence[str] | None = None) -> int:
    """Run checklist, zero-shot forecast, fan chart and optional backtest on the given CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--target", required=True, help="column to forecast")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--id-column", default=None, help="series id column; omit for one series")
    parser.add_argument(
        "--freq", required=True, help='pandas frequency of the grid: "h", "D", "W-MON", "15min"...'
    )
    parser.add_argument("--horizon", type=int, required=True, help="steps to forecast")
    parser.add_argument("--known", nargs="*", default=[], help="known-future covariate columns")
    parser.add_argument("--quantiles", nargs="*", type=float, default=list(DEFAULT_QUANTILES))
    parser.add_argument(
        "--source-tz",
        default=None,
        help='time zone of naive local timestamps, e.g. "Europe/Madrid"',
    )
    parser.add_argument(
        "--sentinels",
        nargs="*",
        type=float,
        default=[],
        help='values meaning "no reading", e.g. 0 -999',
    )
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        help="most recent steps per series fed to the model (default: the model's 8,192 cap)",
    )
    parser.add_argument("--model", default=CHRONOS2, help="Hub id or local checkpoint folder")
    parser.add_argument(
        "--backtest",
        type=int,
        default=0,
        metavar="N",
        help="also replay the last N horizons against a seasonal naive",
    )
    parser.add_argument(
        "--season-length",
        type=int,
        default=None,
        help="steps per season for the baseline and MASE "
        "(default: 24 for hourly, 7 for daily, else 1)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output folder (default: next to the CSV)"
    )
    args = parser.parse_args(argv)

    out = args.out or args.csv.parent
    out.mkdir(parents=True, exist_ok=True)
    frame, report = load_frame(
        args.csv,
        id_column=args.id_column,
        timestamp_column=args.timestamp_column,
        target=args.target,
        known=args.known,
        freq=args.freq,
        source_tz=args.source_tz,
        sentinels=args.sentinels,
    )
    print(report)
    history, future = split_future(
        frame, target=args.target, known=args.known, horizon=args.horizon
    )
    if args.known and future is None:
        print(
            "WARNING: --known was given but the CSV has no rows beyond the last target value, so "
            "the covariates will be used as past-only. Append the horizon rows to use their "
            "future.",
            file=sys.stderr,
        )

    if future is not None:  # the same checks predict_df would fail on, with a readable message
        check_future_frame(
            history, future, prediction_length=args.horizon, freq=args.freq, target=args.target
        )

    device = get_device()
    model = Chronos2Forecaster(
        load_chronos2(args.model, device=device), context_length=args.context
    )
    forecast = model.predict(
        history,
        prediction_length=args.horizon,
        quantile_levels=args.quantiles,
        freq=args.freq,
        target=args.target,
        future_covariates=future,
    )
    forecast_path = out / f"{args.csv.stem}_forecast.csv"
    forecast.to_csv(forecast_path, index=False)
    print(f"Saved {forecast_path} ({len(forecast):,} rows, {forecast[ID_COL].nunique()} series)")

    set_style()
    first = str(history[ID_COL].iloc[0])
    ax = plot_fan(
        history,
        forecast,
        target=args.target,
        item_id=first,
        freq=args.freq,
        intervals={
            round(args.quantiles[-1] - args.quantiles[0], 2): (
                args.quantiles[0],
                args.quantiles[-1],
            )
        }
        if len(args.quantiles) >= 2
        else {},
        title=f"{first}: next {args.horizon} steps, Chronos-2 zero-shot",
    )
    (figure_path,) = save_figure(ax.figure, f"{args.csv.stem}_fan", out)
    print(f"Saved {figure_path}")

    if args.backtest:
        season = args.season_length or {"h": 24, "D": 7}.get(args.freq, 1)
        config = BacktestConfig(
            prediction_length=args.horizon,
            n_windows=args.backtest,
            freq=args.freq,
            season_length=season,
            target=args.target,
            known_covariates=tuple(args.known),
            max_context=args.context,
        )
        models = [model, SeasonalNaive(season_length=season)]
        metrics = pd.concat(
            [evaluate(backtest(m, history, config), history, config) for m in models],
            ignore_index=True,
        )
        skill = skill_table(metrics, baseline="seasonal_naive", metric="WQL")
        metrics_path = out / f"{args.csv.stem}_backtest.csv"
        metrics.to_csv(metrics_path, index=False)
        print(f"\nBacktest over the last {args.backtest} horizons (saved {metrics_path.name}):")
        summary = metrics.groupby("model")[["WQL", "MASE", "coverage_80"]].mean().round(4)
        print(summary.to_string())
        row = skill[skill["model"] == model.name].iloc[0]
        print(
            f"\nChronos-2 vs seasonal naive: WQL skill {row['skill']:+.1%} "
            f"[{row['ci_low']:+.1%}, {row['ci_high']:+.1%}], wins {row['win_rate']:.0%} of windows"
        )
        if row["ci_low"] <= 0 <= row["ci_high"]:
            print(
                "The interval contains zero: on this data the model is not shown to beat the naive."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
