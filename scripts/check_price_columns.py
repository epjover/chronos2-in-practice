"""Identify the two price columns of the Spanish dataset against OMIE's published prices.

The Kaggle table has ``price actual`` and ``price day ahead`` and describes the second as a
forecast. Part 3 forecasts the first, so what each column is decides how to read the result.
OMIE, the Iberian market operator, publishes the hourly marginal price of the day-ahead market;
this script downloads its file for one year, puts it on the dataset's UTC grid and measures how
each column relates to it.

For 2018 it finds that ``price actual`` follows OMIE's day-ahead price (correlation 0.99) about
6 EUR/MWh above it, which is what a final price made of the day-ahead price plus system costs
looks like, and that ``price day ahead`` equals OMIE's price to the cent on about two thirds of
the days and holds other values on the rest.

Usage::

    uv run python scripts/check_price_columns.py    # 2018, into results/05_price_columns.csv

It reads one public file from www.omie.es and nothing else.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from collections.abc import Sequence

import numpy as np
import pandas as pd

from c2p.config import get_paths
from c2p.data import spain_energy_frame
from c2p.results import save_results
from c2p.schema import TIMESTAMP_COL

OMIE_URL = (
    "https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc"
    "&filename=marginalpdbc_{year}.zip"
)
ACTUAL = "price_actual"
DAY_AHEAD = "price_day_ahead_tso"
OMIE = "omie_day_ahead"
# Two prices quoted in EUR/MWh with two decimals are the same price within half a cent.
SAME_PRICE = 0.005
TIMEOUT_S = 120


def download(year: int) -> bytes:
    """The zip of OMIE's daily ``marginalpdbc`` files for ``year``."""
    with urllib.request.urlopen(OMIE_URL.format(year=year), timeout=TIMEOUT_S) as response:
        return response.read()


def parse_omie(archive: bytes) -> pd.DataFrame:
    """OMIE's Spanish day-ahead price on a naive UTC hourly grid.

    Each daily file has one line per hour, ``year;month;day;hour;price_PT;price_ES;``, where
    ``hour`` counts from 1 in Spanish local time (23 or 25 hours on the days the clocks
    change). The hour index is added to local midnight in UTC, so those days stay aligned.
    """
    rows = []
    with zipfile.ZipFile(io.BytesIO(archive)) as files:
        for name in files.namelist():
            for line in files.read(name).decode("latin-1").splitlines():
                parts = line.split(";")
                if len(parts) >= 6 and parts[0].isdigit():
                    rows.append((*(int(part) for part in parts[:4]), float(parts[5])))
    if not rows:
        raise ValueError("The archive has no hourly price lines.")
    frame = pd.DataFrame(rows, columns=["year", "month", "day", "hour", OMIE])
    frame = frame.drop_duplicates(["year", "month", "day", "hour"])
    local_midnight = pd.to_datetime(frame[["year", "month", "day"]]).dt.tz_localize("Europe/Madrid")
    utc = local_midnight.dt.tz_convert("UTC") + pd.to_timedelta(frame["hour"] - 1, unit="h")
    return pd.DataFrame(
        {TIMESTAMP_COL: utc.dt.tz_localize(None).to_numpy(), OMIE: frame[OMIE].to_numpy()}
    )


def compare(dataset: pd.DataFrame, omie: pd.DataFrame) -> pd.DataFrame:
    """One row per measure of how the dataset's two price columns relate to OMIE's price."""
    joined = dataset[[TIMESTAMP_COL, ACTUAL, DAY_AHEAD]].merge(omie, on=TIMESTAMP_COL, how="inner")
    joined = joined.dropna()
    if joined.empty:
        raise ValueError("The dataset and the OMIE file share no hours.")
    premium = joined[ACTUAL] - joined[OMIE]
    same = (joined[DAY_AHEAD] - joined[OMIE]).abs() < SAME_PRICE
    local_day = joined[TIMESTAMP_COL].dt.tz_localize("UTC").dt.tz_convert("Europe/Madrid").dt.date
    share_by_day = same.groupby(local_day.to_numpy()).mean()
    measures = {
        "hours_compared": len(joined),
        "actual_vs_omie_correlation": joined[ACTUAL].corr(joined[OMIE]),
        "actual_minus_omie_mean": premium.mean(),
        "actual_minus_omie_std": premium.std(),
        "actual_equal_to_omie_share": ((joined[ACTUAL] - joined[OMIE]).abs() < SAME_PRICE).mean(),
        "day_ahead_column_equal_to_omie_share": same.mean(),
        "day_ahead_column_vs_omie_correlation": joined[DAY_AHEAD].corr(joined[OMIE]),
        "days_compared": len(share_by_day),
        "days_day_ahead_column_equal_all_hours": int((share_by_day == 1.0).sum()),
        "days_day_ahead_column_equal_no_hour": int((share_by_day == 0.0).sum()),
    }
    return pd.DataFrame(
        {"measure": list(measures), "value": [float(np.round(v, 6)) for v in measures.values()]}
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Download OMIE's prices for one year and write the comparison to ``results/``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=2018)
    args = parser.parse_args(argv)

    print(f"Reading {OMIE_URL.format(year=args.year)}")
    table = compare(spain_energy_frame(), parse_omie(download(args.year)))
    table.insert(0, "year", args.year)
    path = save_results(table, "05_price_columns", get_paths())
    print(table.to_string(index=False))
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
