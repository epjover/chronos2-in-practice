"""Load the Spanish energy dataset and get series frames ready for Chronos-2.

Article 1 walks through :func:`validate_and_prepare`, a checklist that turns a messy table
into a clean *series frame* (see :mod:`c2p.schema`): timestamps in naive UTC, one row per
series and grid step, missing values as NaN. The module also builds future frames, adds
calendar covariates and corrupts clean data on purpose for the "messy data" demo.

The dataset is the CC0 Kaggle dataset
``nicholasjhana/energy-consumption-generation-prices-and-weather``: hourly load, prices and
generation for Spain (2015-2018) plus observed weather for five cities. It is downloaded on
demand to ``data/raw/spain_energy/`` and never committed.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import holidays
import kagglehub
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset

from c2p.config import get_paths
from c2p.schema import ID_COL, TARGET_COL, TIMESTAMP_COL

logger = logging.getLogger(__name__)

DuplicatePolicy = Literal["mean", "first", "last", "raise"]

# --------------------------------------------------------------------------------------------
# Download and raw files
# --------------------------------------------------------------------------------------------

SPAIN_ENERGY_HANDLE = "nicholasjhana/energy-consumption-generation-prices-and-weather/versions/1"
ENERGY_FILE = "energy_dataset.csv"
WEATHER_FILE = "weather_features.csv"
SPAIN_ENERGY_FILES: dict[str, str] = {
    ENERGY_FILE: "14cd13e020d87af0ab49db8e0f1e4d308320ec5b8d2dec23bab4d27c75c7733a",
    WEATHER_FILE: "3566c3c21179d4fc6691eec6d37a6cfe8fe6a4d9e097f95e4fda4a29dc683b18",
}
SPAIN_ITEM_ID = "ES"

_HASH_CHUNK_BYTES = 1 << 20


def default_data_dir() -> Path:
    """Folder that holds the Spanish dataset: ``data/raw/spain_energy`` in the repository."""
    return get_paths().data_raw / "spain_energy"


def download_spain_energy(dest: Path | None = None, *, force: bool = False) -> Path:
    """Download the Spanish energy dataset (if needed) and return the folder with its CSVs.

    Files are fetched with ``kagglehub`` (no credentials needed; ``KAGGLEHUB_CACHE`` is
    honoured), copied into ``dest`` and checked against :data:`SPAIN_ENERGY_FILES`. Calling it
    again is cheap: nothing is downloaded while both files are present, unless ``force``.

    Parameters
    ----------
    dest
        Target folder; defaults to :func:`default_data_dir`.
    force
        Download again even if the files are already there.

    Raises
    ------
    ValueError
        If a file does not match its expected SHA256 checksum.
    """
    folder = Path(dest) if dest is not None else default_data_dir()
    missing = [name for name in SPAIN_ENERGY_FILES if not (folder / name).is_file()]
    if force or missing:
        logger.info("Downloading %s (missing: %s, force=%s)", SPAIN_ENERGY_HANDLE, missing, force)
        source = Path(kagglehub.dataset_download(SPAIN_ENERGY_HANDLE, force_download=force))
        _copy_dataset_files(source, folder)
    else:
        logger.info("Spanish energy dataset already in %s", folder)
    _verify_checksums(folder)
    return folder


def _copy_dataset_files(source: Path, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in SPAIN_ENERGY_FILES:
        if not (source / name).is_file():
            found = sorted(path.name for path in source.iterdir())
            raise FileNotFoundError(f"{name} not found in the download at {source}: {found}.")
        shutil.copy2(source / name, folder / name)
        logger.info("Copied %s to %s", name, folder)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksums(folder: Path) -> None:
    for name, expected in SPAIN_ENERGY_FILES.items():
        actual = _sha256(folder / name)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {folder / name}: expected SHA256 {expected}, got {actual}. "
                "Delete the file or call download_spain_energy(force=True) to download it again."
            )


def load_energy_raw(data_dir: Path | None = None) -> pd.DataFrame:
    """Read ``energy_dataset.csv`` as published (downloaded on demand)."""
    return pd.read_csv(download_spain_energy(data_dir) / ENERGY_FILE)


def load_weather_raw(data_dir: Path | None = None) -> pd.DataFrame:
    """Read ``weather_features.csv`` as published (downloaded on demand)."""
    return pd.read_csv(download_spain_energy(data_dir) / WEATHER_FILE)


# --------------------------------------------------------------------------------------------
# Spanish series frames
# --------------------------------------------------------------------------------------------

_GENERATION_SOURCES = (
    "biomass",
    "fossil brown coal/lignite",
    "fossil coal-derived gas",
    "fossil gas",
    "fossil hard coal",
    "fossil oil",
    "fossil oil shale",
    "fossil peat",
    "geothermal",
    "hydro pumped storage aggregated",
    "hydro pumped storage consumption",
    "hydro run-of-river and poundage",
    "hydro water reservoir",
    "marine",
    "nuclear",
    "other",
    "other renewable",
    "solar",
    "waste",
    "wind offshore",
    "wind onshore",
)


def _snake_case(name: str) -> str:
    """``"fossil brown coal/lignite"`` -> ``"fossil_brown_coal_lignite"``."""
    return re.sub(r"[\s/-]+", "_", name.strip()).lower()


# Raw column -> series-frame column. "_tso" marks day-ahead values published the day before:
# the grid operator's (TSO) forecasts of load, solar and wind, and the day-ahead market price.
# Values for day D are known only from D-1, so they are known-future covariates for horizons
# that end within the next day (about 24 h from a midnight cutoff). For longer horizons (e.g.
# 168 h) later days were not yet published: use them as past-only covariates or lag them.
ENERGY_COLUMNS: dict[str, str] = {
    "total load actual": "load_actual",
    "total load forecast": "load_forecast_tso",
    "price actual": "price_actual",
    "price day ahead": "price_day_ahead_tso",
    "forecast solar day ahead": "solar_forecast_tso",
    "forecast wind onshore day ahead": "wind_onshore_forecast_tso",
    "forecast wind offshore eday ahead": "wind_offshore_forecast_tso",  # "eday" (sic) in the file
    **{f"generation {source}": f"gen_{_snake_case(source)}" for source in _GENERATION_SOURCES},
}
_ENERGY_TIME_COLUMN = "time"

KELVIN_OFFSET = 273.15
_WEATHER_TEMPERATURES = {"temp": "temp_c", "temp_min": "temp_min_c", "temp_max": "temp_max_c"}
_WEATHER_NUMERIC = (
    "pressure",
    "humidity",
    "wind_speed",
    "wind_deg",
    "rain_1h",
    "rain_3h",
    "snow_3h",
    "clouds_all",
)


def _offset_strings_to_naive_utc(values: pd.Series) -> pd.Series:
    """Parse ISO strings with UTC offsets (``+01:00``/``+02:00``) into naive UTC timestamps."""
    return pd.to_datetime(values, utc=True).dt.tz_localize(None)


def spain_energy_frame(data_dir: Path | None = None) -> pd.DataFrame:
    """Spanish national data as a wide series frame with ``item_id="ES"``.

    Columns are renamed with :data:`ENERGY_COLUMNS`, timestamps are naive UTC and all-NaN
    columns are dropped (two in the published file). Nothing else is cleaned: run
    :func:`validate_and_prepare` on the result (e.g. with ``freq="h", target="load_actual"``).
    """
    raw = load_energy_raw(data_dir)
    unknown = sorted(set(raw.columns) - set(ENERGY_COLUMNS) - {_ENERGY_TIME_COLUMN})
    if unknown:
        raise ValueError(
            f"Unexpected columns in {ENERGY_FILE}: {unknown}. Add them to ENERGY_COLUMNS."
        )
    ordered = [column for column in ENERGY_COLUMNS if column in raw.columns]
    frame = raw[ordered].rename(columns=ENERGY_COLUMNS)
    all_nan = frame.columns[frame.isna().all()].tolist()
    if all_nan:
        logger.info("Dropping all-NaN columns: %s", all_nan)
    frame = frame.drop(columns=all_nan)
    frame.insert(0, TIMESTAMP_COL, _offset_strings_to_naive_utc(raw[_ENERGY_TIME_COLUMN]))
    frame.insert(0, ID_COL, SPAIN_ITEM_ID)
    return frame


def spain_weather_frame(data_dir: Path | None = None) -> pd.DataFrame:
    """Observed weather for five Spanish cities as a long series frame (one series per city).

    ``item_id`` is the city name without the stray spaces of the raw file, timestamps are naive
    UTC and temperatures are converted from Kelvin to Celsius. Only numeric measurements and
    the categorical ``weather_main`` are kept (``weather_id``, ``weather_description`` and
    ``weather_icon`` are labels of the same information). Duplicated (city, timestamp) rows
    are kept on purpose: :func:`validate_and_prepare` resolves them.

    This is *observed* weather: using it as a known-future covariate leaks the future.
    """
    raw = load_weather_raw(data_dir)
    frame = pd.DataFrame(
        {
            ID_COL: raw["city_name"].str.strip(),
            TIMESTAMP_COL: _offset_strings_to_naive_utc(raw["dt_iso"]),
            **{new: raw[old] - KELVIN_OFFSET for old, new in _WEATHER_TEMPERATURES.items()},
            **{column: raw[column] for column in _WEATHER_NUMERIC},
            "weather_main": raw["weather_main"].astype("category"),
        }
    )
    return frame.sort_values([ID_COL, TIMESTAMP_COL], kind="stable", ignore_index=True)


# --------------------------------------------------------------------------------------------
# Data checklist
# --------------------------------------------------------------------------------------------


@dataclass
class DataReport:
    """What :func:`validate_and_prepare` found and changed."""

    n_series: int
    n_rows_in: int
    n_rows_out: int
    n_duplicates: int
    n_sentinels: int
    n_missing_timestamps: int
    n_missing_values: dict[str, int]
    n_trimmed: int
    short_series: list[str]
    freq: str
    notes: list[str] = field(default_factory=list)
    n_sentinels_by_column: dict[str, int] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """One row per metric (``metric``, ``value``), ready for ``results/*.csv``."""
        rows: list[tuple[str, object]] = [
            ("freq", self.freq),
            ("n_series", self.n_series),
            ("n_rows_in", self.n_rows_in),
            ("n_rows_out", self.n_rows_out),
            ("n_duplicates", self.n_duplicates),
            ("n_sentinels", self.n_sentinels),
            *((f"n_sentinels[{col}]", n) for col, n in self.n_sentinels_by_column.items()),
            ("n_missing_timestamps", self.n_missing_timestamps),
            ("n_trimmed", self.n_trimmed),
            *((f"n_missing_values[{col}]", n) for col, n in self.n_missing_values.items()),
            ("short_series", "; ".join(self.short_series)),
            ("notes", " | ".join(self.notes)),
        ]
        return pd.DataFrame(rows, columns=["metric", "value"])

    def __str__(self) -> str:
        missing = {col: n for col, n in self.n_missing_values.items() if n > 0}
        missing_text = ", ".join(f"{col}={n:,}" for col, n in missing.items()) or "none"
        sentinel_text = ", ".join(f"{col}={n:,}" for col, n in self.n_sentinels_by_column.items())
        lines = [
            f"DataReport (freq={self.freq!r}, {self.n_series:,} series)",
            f"  rows in -> out:              {self.n_rows_in:,} -> {self.n_rows_out:,}",
            f"  duplicated rows resolved:    {self.n_duplicates:,}",
            f"  sentinel values -> NaN:      {self.n_sentinels:,}"
            + (f" ({sentinel_text})" if sentinel_text else ""),
            f"  missing timestamps inserted: {self.n_missing_timestamps:,}",
            f"  rows trimmed (max_length):   {self.n_trimmed:,}",
            f"  missing values after prep:   {missing_text}",
            f"  short series:                {', '.join(self.short_series) or 'none'}",
        ]
        lines += [f"  note: {note}" for note in self.notes]
        return "\n".join(lines)


def validate_and_prepare(
    df: pd.DataFrame,
    *,
    freq: str,
    target: str | Sequence[str] = TARGET_COL,
    id_column: str = ID_COL,
    timestamp_column: str = TIMESTAMP_COL,
    source_tz: str | None = None,
    sentinels: Sequence[float] = (),
    sentinel_columns: Sequence[str] | None = None,
    duplicates: DuplicatePolicy = "mean",
    categorical: str | Sequence[str] = (),
    max_length: int | None = None,
    min_length: int | None = None,
) -> tuple[pd.DataFrame, DataReport]:
    """Run the data checklist and return a clean series frame plus a report.

    Steps, in order:

    1. Check the arguments and columns (targets numeric, other text columns listed in
       ``categorical``) and work on a copy with string ids: the input is never modified.
    2. Convert timestamps to timezone-naive UTC. Naive local times need ``source_tz``
       (DST is resolved per series; non-existent spring times are shifted forward).
    3. Replace ``sentinels`` (e.g. ``0`` or ``-999`` meaning "no reading") with NaN.
    4. Sort, then resolve duplicated (series, timestamp) rows with the ``duplicates`` policy:
       numeric columns are averaged (``"mean"``) and categorical ones take the first value.
    5. Reject timestamps off the series' ``freq`` grid (e.g. 10:30 among hourly rows at :00).
    6. Reindex every series to a regular grid: missing timestamps become rows of NaN.
    7. Keep the most recent ``max_length`` rows per series (Chronos-2 reads at most 8,192
       steps of context anyway).
    8. Flag series with fewer than ``min_length`` rows in ``DataReport.short_series``.
    9. Cast numeric columns to float64 and ``categorical`` columns to ``category``.

    What it deliberately does NOT do: normalise, difference, detrend or impute. Chronos-2
    standardises each series itself and masks NaN, so these steps would only lose
    information (and imputed values look like real observations).

    Parameters
    ----------
    df
        Long frame with an id column, a timestamp column and the target column(s).
    freq
        Pandas frequency of the grid, e.g. ``"h"``.
    target
        Target column name or names.
    source_tz
        IANA time zone of naive timestamps (e.g. ``"Europe/Madrid"``); naive timestamps
        without it are assumed to be UTC. The repeated autumn hour can only be told apart by
        position, so the rows of each series must be in time order (as recorded). Use it for
        sub-daily data only: daily or coarser dates are calendar labels, not instants.
    sentinels
        Exact values that mean "missing", e.g. ``[0]`` or ``[-999]``.
    sentinel_columns
        Columns searched for ``sentinels``. Default: every numeric non-key column, covariates
        included (a sentinel of ``0`` also blanks legitimate zeros there, so the report counts
        them per column); pass the targets to limit the search.
    duplicates
        ``"mean"``, ``"first"``, ``"last"`` or ``"raise"``.
    categorical
        Non-target column name or names to keep as ``category``.
    max_length, min_length
        Optional per-series length limits (trim, flag).

    Returns
    -------
    tuple[pd.DataFrame, DataReport]
        The series frame, sorted by id then timestamp with a fresh index, and the report.
    """
    targets, categorical = _as_list(target), _as_list(categorical)
    keys = [id_column, timestamp_column]
    notes: list[str] = []
    _check_arguments(
        targets=targets,
        sentinels=sentinels,
        duplicates=duplicates,
        max_length=max_length,
        min_length=min_length,
    )
    _check_columns(
        df, keys=keys, targets=targets, categorical=categorical, sentinel_columns=sentinel_columns
    )
    frame = _copy_with_string_ids(df, id_column=id_column)
    offset = to_offset(freq)

    frame[timestamp_column] = _to_naive_utc(
        frame[timestamp_column], ids=frame[id_column], source_tz=source_tz, notes=notes
    )
    frame, sentinel_counts = _replace_sentinels(
        frame, sentinels, columns=sentinel_columns, skip=[*keys, *categorical]
    )
    frame, n_duplicates = _resolve_duplicates(frame, duplicates, keys=keys, categorical=categorical)
    _reject_off_grid_timestamps(
        frame, offset, id_column=id_column, timestamp_column=timestamp_column, source_tz=source_tz
    )
    frame, n_missing_timestamps = _reindex_to_regular_grid(
        frame, offset, id_column=id_column, timestamp_column=timestamp_column
    )
    frame, n_trimmed = _keep_most_recent(frame, max_length, id_column=id_column)
    short_series = _find_short_series(frame, min_length, id_column=id_column)
    frame = _set_dtypes(frame, categorical=categorical, keys=keys)

    value_columns = [column for column in frame.columns if column not in keys]
    report = DataReport(
        n_series=frame[id_column].nunique(),
        n_rows_in=len(df),
        n_rows_out=len(frame),
        n_duplicates=n_duplicates,
        n_sentinels=sum(sentinel_counts.values()),
        n_missing_timestamps=n_missing_timestamps,
        n_missing_values={column: int(frame[column].isna().sum()) for column in value_columns},
        n_trimmed=n_trimmed,
        short_series=short_series,
        freq=freq,
        notes=notes,
        n_sentinels_by_column=sentinel_counts,
    )
    logger.info("validate_and_prepare:\n%s", report)
    return frame, report


def _as_list(names: str | Sequence[str]) -> list[str]:
    """A column name or a sequence of names -> list of names."""
    return [names] if isinstance(names, str) else list(names)


def _require_columns(df: pd.DataFrame, columns: Sequence[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{frame_name} is missing column(s) {missing}; available: {df.columns.tolist()}."
        )


def _is_numeric(values: pd.Series) -> bool:
    """Numbers, excluding booleans (``0 == False`` would make sentinels match flags)."""
    return pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool)


def _check_arguments(
    *,
    targets: list[str],
    sentinels: Sequence[float],
    duplicates: DuplicatePolicy,
    max_length: int | None,
    min_length: int | None,
) -> None:
    """Step 1a: reject invalid arguments before looking at the data."""
    if not targets:
        raise ValueError("At least one target column is required.")
    if duplicates not in (policies := get_args(DuplicatePolicy)):
        raise ValueError(f"duplicates must be one of {policies}, got {duplicates!r}.")
    if np.ndim(sentinels) != 1 or not all(_is_number(value) for value in sentinels):
        raise TypeError(
            f"sentinels must be a list of numbers, e.g. [0] or [-999]; got {sentinels!r}."
        )
    for name, length in {"max_length": max_length, "min_length": min_length}.items():
        if length is not None and length < 1:
            raise ValueError(f"{name} must be a positive integer or None, got {length}.")


def _check_columns(
    df: pd.DataFrame,
    *,
    keys: list[str],
    targets: list[str],
    categorical: list[str],
    sentinel_columns: Sequence[str] | None,
) -> None:
    """Step 1b: columns exist, targets are numeric and other text columns are declared."""
    if df.empty:
        raise ValueError("The input frame is empty.")
    roles = [*keys, *targets, *categorical]
    _require_columns(df, [*roles, *(sentinel_columns or ())], "The input frame")
    if repeated := sorted({column for column in roles if roles.count(column) > 1}):
        raise ValueError(
            f"Column(s) {repeated} are used twice among id, timestamp, target and categorical."
        )
    if not_numeric := [column for column in targets if not _is_numeric(df[column])]:
        raise TypeError(
            f"Target column(s) {not_numeric} must be numeric, got "
            f"{[str(df[column].dtype) for column in not_numeric]}. Convert them with pd.to_numeric."
        )
    other = [column for column in df.columns if column not in roles]
    if not_numeric := [column for column in other if not pd.api.types.is_numeric_dtype(df[column])]:
        raise TypeError(
            f"Column(s) {not_numeric} are not numeric: list them in categorical= or drop them."
        )
    searchable = [*targets, *other]
    if bad := [c for c in sentinel_columns or () if c not in searchable or not _is_numeric(df[c])]:
        raise ValueError(
            f"sentinel_columns must be numeric target or covariate columns; {bad} are not."
        )
    _check_keys_present(df, keys)


def _check_keys_present(df: pd.DataFrame, keys: list[str]) -> None:
    """Ids and timestamps must be filled in; blank strings count as missing."""
    for column in keys:
        missing = df[column].isna()
        if pd.api.types.is_string_dtype(df[column]):
            missing |= df[column].str.strip().eq("")
        if n_missing := int(missing.sum()):
            raise ValueError(
                f"{n_missing} row(s) have a missing {column!r}; drop or fix them first."
            )


def _copy_with_string_ids(df: pd.DataFrame, *, id_column: str) -> pd.DataFrame:
    """Step 1c: work on a copy (the input is never modified) with ``str`` ids."""
    frame = df.copy().reset_index(drop=True)
    frame[id_column] = frame[id_column].astype(str)
    return frame


# A time followed by a UTC offset: "00:00:00+01:00", "00:00-0500", "00:00:00+01" or "00:00Z".
# Requiring the time keeps date-only strings such as "01-02-2018" from looking like offsets.
_UTC_OFFSET_PATTERN = r"\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:Z|[+-]\d{2}(?::?\d{2})?)$"


def _to_naive_utc(
    timestamps: pd.Series, *, ids: pd.Series, source_tz: str | None, notes: list[str]
) -> pd.Series:
    """Step 2: any timestamp representation -> timezone-naive UTC."""
    if pd.api.types.is_string_dtype(timestamps):
        timestamps = _parse_timestamp_strings(timestamps)
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        raise TypeError(
            f"Timestamps must be datetimes or ISO strings, got dtype {timestamps.dtype}."
        )
    if timestamps.dt.tz is not None:
        if source_tz is not None:
            notes.append(
                f"source_tz={source_tz!r} ignored: the timestamps carry their own offsets."
            )
        return timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    if source_tz is None:
        notes.append("Timestamps are timezone-naive and no source_tz was given: assumed to be UTC.")
        return timestamps
    return _local_to_naive_utc(timestamps, ids=ids, tz=source_tz, notes=notes)


def _parse_timestamp_strings(values: pd.Series) -> pd.Series:
    has_offset = values.str.strip().str.contains(_UTC_OFFSET_PATTERN, regex=True)
    if has_offset.any() and not has_offset.all():
        examples = values[~has_offset].head(3).tolist()
        raise ValueError(
            f"{int((~has_offset).sum())} timestamp string(s) lack a UTC offset while others have "
            f"one, e.g. {examples}. Make the format consistent before preparing the data."
        )
    parsed = pd.to_datetime(values, utc=bool(has_offset.all()))
    if n_unparsed := int(parsed.isna().sum()):
        examples = values[parsed.isna()].head(3).tolist()
        raise ValueError(
            f"{n_unparsed} timestamp string(s) are not dates, e.g. {examples}; drop or fix them."
        )
    return parsed


def _local_to_naive_utc(
    timestamps: pd.Series, *, ids: pd.Series, tz: str, notes: list[str]
) -> pd.Series:
    """Localize naive local times series by series, in row order (DST inference needs it)."""
    localized = pd.concat(
        [
            _localize_series(series, tz=tz, item=item)
            for item, series in timestamps.groupby(ids, sort=False)
        ]
    ).reindex(timestamps.index)
    n_shifted = int((localized.dt.tz_localize(None) != timestamps).sum())
    if n_shifted:
        notes.append(
            f"{n_shifted} non-existent local time(s) in {tz} (spring DST gap) shifted forward."
        )
    return localized.dt.tz_convert("UTC").dt.tz_localize(None)


def _localize_series(timestamps: pd.Series, *, tz: str, item: str) -> pd.Series:
    """Attach ``tz`` to one series kept in row order.

    Times repeated in the autumn (e.g. 02:00-02:59 twice in Madrid) are told apart only by
    their position, so they are inferred from the row order, never from a sort by local time
    (which would interleave the two copies of every repeated sub-hourly time).
    """
    as_nat = timestamps.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    ambiguous = as_nat.isna()
    if not ambiguous.any():
        return as_nat
    examples = timestamps[ambiguous].head(3).astype(str).tolist()
    try:
        localized = timestamps.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    except ValueError as error:
        raise ValueError(
            f"Series {item!r}: cannot tell which local times in {tz} are before or after the "
            f"autumn DST change, e.g. {examples} ({error}). Keep the rows in the order they "
            "were recorded, use timestamps with UTC offsets (e.g. '2018-10-28 02:00:00+01:00') "
            "or tz-aware values, or drop these rows."
        ) from error
    if not localized.is_monotonic_increasing:
        raise ValueError(
            f"Series {item!r} has local times repeated by the autumn DST change in {tz}, e.g. "
            f"{examples}, but its rows are not in time order, so they cannot be told apart. "
            "Keep the rows in the order they were recorded, or use timestamps with UTC offsets."
        )
    return localized


def _replace_sentinels(
    frame: pd.DataFrame,
    sentinels: Sequence[float],
    *,
    columns: Sequence[str] | None,
    skip: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Step 3: exact sentinel matches -> NaN; return the count per affected column."""
    if len(sentinels) == 0:
        return frame, {}
    if columns is None:
        columns = [c for c in frame.columns if c not in skip and _is_numeric(frame[c])]
    columns = list(columns)
    is_sentinel = frame[columns].isin(list(sentinels))
    frame[columns] = frame[columns].mask(is_sentinel)
    counts = is_sentinel.sum()
    return frame, {column: int(n) for column, n in counts.items() if n > 0}


def _resolve_duplicates(
    frame: pd.DataFrame,
    policy: DuplicatePolicy,
    *,
    keys: list[str],
    categorical: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    """Step 4: sort and keep one row per (series, timestamp)."""
    frame = frame.sort_values(keys, kind="stable", ignore_index=True)
    is_repeat = frame.duplicated(keys)
    n_duplicates = int(is_repeat.sum())
    if n_duplicates == 0:
        return frame, 0
    if policy == "raise":
        examples = frame.loc[is_repeat, keys].drop_duplicates().head(5).to_dict("records")
        raise ValueError(
            f"{n_duplicates} duplicated {tuple(keys)} row(s), e.g. {examples}. Aggregate them "
            "first or pass duplicates='mean', 'first' or 'last'."
        )
    if policy in ("first", "last"):
        return frame.drop_duplicates(keys, keep=policy, ignore_index=True), n_duplicates
    in_group = frame.duplicated(keys, keep=False)
    how = {
        column: "mean" if column not in categorical and _is_numeric(frame[column]) else "first"
        for column in frame.columns
        if column not in keys
    }
    merged = frame[in_group].groupby(keys, sort=False, observed=True).agg(how).reset_index()
    resolved = pd.concat([frame[~in_group], merged[frame.columns]])
    return resolved.sort_values(keys, kind="stable", ignore_index=True), n_duplicates


def _series_grid(timestamps: pd.Series, offset: BaseOffset) -> pd.DatetimeIndex:
    """Regular grid from the first to the last timestamp of one series."""
    return pd.date_range(timestamps.min(), timestamps.max(), freq=offset)


def _reject_off_grid_timestamps(
    frame: pd.DataFrame,
    offset: BaseOffset,
    *,
    id_column: str,
    timestamp_column: str,
    source_tz: str | None,
) -> None:
    """Step 5: every timestamp must be a whole number of steps after its series' first one.

    Only the phase within a series matters: an hourly series at :30 past every hour (local
    hours in India, for example, are at :30 in UTC) is regular, a single 10:30 among rows
    at :00 is not.
    """
    on_series_grid = (
        frame.groupby(id_column, sort=False)[timestamp_column]
        .transform(lambda series: series.isin(_series_grid(series, offset)))
        .astype(bool)
    )
    if on_series_grid.all():
        return
    freq = offset.freqstr
    examples = frame.loc[~on_series_grid, [id_column, timestamp_column]].head(5).astype(str)
    hint = (
        f" Daily or coarser data should not get source_tz={source_tz!r}: its dates are "
        "calendar labels, and converting them to UTC moves them across DST changes."
        if source_tz is not None
        else ""
    )
    raise ValueError(
        f"{int((~on_series_grid).sum())} timestamp(s) are not on the regular {freq!r} grid of "
        f"their series (anchored at its first timestamp), e.g. {examples.to_dict('records')}. "
        f"Check that {freq!r} is the resolution the data was recorded at; if the series mixes "
        f"resolutions, aggregate it to {freq!r} first.{hint}"
    )


def _reindex_to_regular_grid(
    frame: pd.DataFrame, offset: BaseOffset, *, id_column: str, timestamp_column: str
) -> tuple[pd.DataFrame, int]:
    """Step 6: insert NaN rows for missing timestamps; return the number inserted."""
    grid = pd.concat(
        [
            pd.DataFrame({id_column: item, timestamp_column: _series_grid(series, offset)})
            for item, series in frame.groupby(id_column, sort=False)[timestamp_column]
        ],
        ignore_index=True,
    )
    reindexed = (
        frame.set_index([id_column, timestamp_column])
        .reindex(pd.MultiIndex.from_frame(grid))
        .reset_index()
    )
    return reindexed, len(reindexed) - len(frame)


def _keep_most_recent(
    frame: pd.DataFrame, max_length: int | None, *, id_column: str
) -> tuple[pd.DataFrame, int]:
    """Step 7: keep the last ``max_length`` rows of every series."""
    if max_length is None:
        return frame, 0
    kept = frame.groupby(id_column, sort=False).tail(max_length).reset_index(drop=True)
    return kept, len(frame) - len(kept)


def _find_short_series(frame: pd.DataFrame, min_length: int | None, *, id_column: str) -> list[str]:
    """Step 8: ids of series with fewer than ``min_length`` rows."""
    if min_length is None:
        return []
    lengths = frame.groupby(id_column, sort=True).size()
    return lengths.index[lengths < min_length].tolist()


def _set_dtypes(
    frame: pd.DataFrame, *, categorical: Sequence[str], keys: list[str]
) -> pd.DataFrame:
    """Step 9: float64 for numbers, ``category`` for categorical columns; sorted, fresh index."""
    for column in frame.columns:
        if column in keys:
            continue
        frame[column] = frame[column].astype("category" if column in categorical else "float64")
    return frame.sort_values(keys, ignore_index=True)


# --------------------------------------------------------------------------------------------
# Future frames
# --------------------------------------------------------------------------------------------


def make_future_frame(
    context: pd.DataFrame,
    *,
    prediction_length: int,
    freq: str,
    id_column: str = ID_COL,
    timestamp_column: str = TIMESTAMP_COL,
) -> pd.DataFrame:
    """``item_id`` and ``timestamp`` of the next ``prediction_length`` steps of every series."""
    if prediction_length < 1:
        raise ValueError(f"prediction_length must be positive, got {prediction_length}.")
    _require_columns(context, [id_column, timestamp_column], "The context frame")
    if context.empty:
        raise ValueError("The context frame is empty.")
    offset = to_offset(freq)
    last_timestamps = context.groupby(id_column, sort=False)[timestamp_column].max()
    return pd.concat(
        [
            pd.DataFrame(
                {
                    id_column: item,
                    timestamp_column: pd.date_range(
                        last + offset, periods=prediction_length, freq=offset
                    ),
                }
            )
            for item, last in last_timestamps.items()
        ],
        ignore_index=True,
    )


def check_future_frame(
    context: pd.DataFrame,
    future: pd.DataFrame,
    *,
    prediction_length: int,
    freq: str,
    id_column: str = ID_COL,
    timestamp_column: str = TIMESTAMP_COL,
    target: str | Sequence[str] = TARGET_COL,
) -> None:
    """Raise ``ValueError`` unless ``future`` is a valid known-covariate frame for ``context``.

    Valid means (as ``Chronos2Pipeline.predict_df`` expects): no target column, no column
    missing from the context, the same series ids (same values and type), and exactly
    ``prediction_length`` rows per series on the grid steps right after each series' last
    context timestamp. One check is stricter than the library: no NaN. ``predict_df`` accepts
    a NaN here and reads it as "not known at this step" (see ``c2p.baselines.KnownUntil``),
    which is rarely what a hole in a covariate feed means, so this check asks for every step.
    """
    _require_columns(context, [id_column, timestamp_column], "The context frame")
    _require_columns(future, [id_column, timestamp_column], "The future frame")
    if leaked := [column for column in _as_list(target) if column in future.columns]:
        raise ValueError(
            f"The future frame contains target column(s) {leaked}: the model would see the "
            "values it has to forecast. Keep only id, timestamp and known covariates."
        )
    if extra := [column for column in future.columns if column not in context.columns]:
        raise ValueError(
            f"The future frame has column(s) {extra} that the context frame lacks: a known "
            "covariate needs its past values in the context too (predict_df rejects it)."
        )
    _check_same_ids(context, future, id_column)
    rows = future.groupby(id_column).size()
    wrong = rows[rows != prediction_length]
    if not wrong.empty:
        raise ValueError(
            f"Each series needs exactly prediction_length={prediction_length} future rows; "
            f"got {wrong.head(5).to_dict()}."
        )
    _check_future_timestamps(context, future, prediction_length, freq, id_column, timestamp_column)
    if nan_columns := future.columns[future.isna().any()].tolist():
        raise ValueError(
            f"The future frame has NaN in {nan_columns}: known covariates must be known for "
            "every forecast step."
        )


def _check_same_ids(context: pd.DataFrame, future: pd.DataFrame, id_column: str) -> None:
    """Ids are compared as they are: ``"1"`` and ``1`` are different series for Chronos-2."""
    context_ids, future_ids = set(context[id_column]), set(future[id_column])
    if context_ids == future_ids:
        return
    type_hint = ""
    if {str(item) for item in context_ids} == {str(item) for item in future_ids}:
        type_hint = (
            f" The ids differ only in type (context {context[id_column].dtype}, future "
            f"{future[id_column].dtype}): use the same type in both frames."
        )
    raise ValueError(
        "Series differ between context and future frames: only in context "
        f"{sorted(context_ids - future_ids, key=str)[:5]}, only in future "
        f"{sorted(future_ids - context_ids, key=str)[:5]}.{type_hint}"
    )


def _check_future_timestamps(
    context: pd.DataFrame,
    future: pd.DataFrame,
    prediction_length: int,
    freq: str,
    id_column: str,
    timestamp_column: str,
) -> None:
    expected = make_future_frame(
        context,
        prediction_length=prediction_length,
        freq=freq,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    actual_by_id = {
        item: pd.DatetimeIndex(timestamps).sort_values()
        for item, timestamps in future.groupby(id_column)[timestamp_column]
    }
    for item, expected_ts in expected.groupby(id_column)[timestamp_column]:
        actual = actual_by_id[item]
        if not actual.equals(pd.DatetimeIndex(expected_ts)):
            raise ValueError(
                f"Series {item!r}: future timestamps must be the {prediction_length} {freq!r} "
                f"steps after the last context timestamp, {expected_ts.iloc[0]} to "
                f"{expected_ts.iloc[-1]}; got {actual[0]} to {actual[-1]}."
            )


# --------------------------------------------------------------------------------------------
# Calendar covariates
# --------------------------------------------------------------------------------------------

CALENDAR_COLUMNS: tuple[str, ...] = ("hour", "day_of_week", "month", "is_weekend", "is_holiday")


def add_calendar_features(
    df: pd.DataFrame,
    *,
    timestamp_column: str = TIMESTAMP_COL,
    tz: str = "Europe/Madrid",
    country: str = "ES",
) -> pd.DataFrame:
    """Return a copy with float calendar columns computed in local time.

    Adds :data:`CALENDAR_COLUMNS`: ``hour``, ``day_of_week`` (Monday=0), ``month``,
    ``is_weekend`` and ``is_holiday`` (national holidays of ``country``). Demand follows the
    local clock, so features use ``tz`` while ``timestamp`` stays naive UTC. The calendar is
    known in advance, so it works on future frames too (a valid known-future covariate).
    """
    _require_columns(df, [timestamp_column], "The frame")
    if existing := [column for column in CALENDAR_COLUMNS if column in df.columns]:
        raise ValueError(f"Calendar column(s) {existing} already exist; drop them first.")
    timestamps = df[timestamp_column]
    if not pd.api.types.is_datetime64_dtype(timestamps) or timestamps.isna().any() or df.empty:
        raise ValueError(
            f"{timestamp_column!r} must be non-empty timezone-naive UTC datetimes without NaT, "
            f"got dtype {timestamps.dtype}; run validate_and_prepare first."
        )
    local = timestamps.dt.tz_localize("UTC").dt.tz_convert(tz)
    local_dates = local.dt.tz_localize(None).dt.normalize()
    years = range(local.dt.year.min(), local.dt.year.max() + 1)
    holiday_dates = pd.to_datetime(list(holidays.country_holidays(country, years=years)))

    out = df.copy()
    out["hour"] = local.dt.hour.astype("float64")
    out["day_of_week"] = local.dt.day_of_week.astype("float64")
    out["month"] = local.dt.month.astype("float64")
    out["is_weekend"] = (local.dt.day_of_week >= 5).astype("float64")
    out["is_holiday"] = local_dates.isin(holiday_dates).astype("float64")
    return out


# --------------------------------------------------------------------------------------------
# Corruption helpers for the "messy data" demo
# --------------------------------------------------------------------------------------------


def drop_rows(
    df: pd.DataFrame,
    *,
    n_gaps: int,
    gap_length: int,
    rng: np.random.Generator,
    column: str = TARGET_COL,
    protect_last: int = 0,
    id_column: str = ID_COL,
    timestamp_column: str = TIMESTAMP_COL,
) -> pd.DataFrame:
    """Return a copy without ``n_gaps`` runs of ``gap_length`` time steps in every series.

    Runs are consecutive in time (whatever the row order) and never include the first step
    or the last ``max(protect_last, 1)`` steps of a series. The removed rows are exactly those
    :func:`mask_values` blanks for an identically seeded ``rng``, so "rows dropped" and
    "values missing" can be compared like for like. ``column`` is only validated (it keeps
    both signatures interchangeable). The original index is kept, so
    ``df.index.difference(result.index)`` lists the removed rows.
    """
    _require_columns(df, [column], "The frame")
    positions = _gap_positions(
        df,
        n_gaps=n_gaps,
        gap_length=gap_length,
        rng=rng,
        protect_last=protect_last,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    keep = np.ones(len(df), dtype=bool)
    keep[positions] = False
    return df.iloc[keep].copy()


def mask_values(
    df: pd.DataFrame,
    *,
    n_gaps: int,
    gap_length: int,
    rng: np.random.Generator,
    column: str = TARGET_COL,
    fill: float = np.nan,
    protect_last: int = 0,
    id_column: str = ID_COL,
    timestamp_column: str = TIMESTAMP_COL,
) -> pd.DataFrame:
    """Return a copy where ``column`` is set to ``fill`` (NaN, or a sentinel such as 0) in gaps.

    Gaps are placed exactly where :func:`drop_rows` removes rows for an identically seeded
    ``rng``: runs of ``gap_length`` consecutive time steps, away from both ends.
    """
    _require_columns(df, [column], "The frame")
    positions = _gap_positions(
        df,
        n_gaps=n_gaps,
        gap_length=gap_length,
        rng=rng,
        protect_last=protect_last,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    out = df.copy()
    out[column] = out[column].astype("float64")
    out.iloc[positions, out.columns.get_loc(column)] = fill
    return out


def _gap_positions(
    df: pd.DataFrame,
    *,
    n_gaps: int,
    gap_length: int,
    rng: np.random.Generator,
    protect_last: int,
    id_column: str,
    timestamp_column: str,
) -> np.ndarray:
    """Row positions of the gaps of every series; gap offsets count steps in time order."""
    _require_columns(df, [id_column, timestamp_column], "The frame")
    if n_gaps < 0 or gap_length < 1 or protect_last < 0:
        raise ValueError(
            "Need n_gaps >= 0, gap_length >= 1 and protect_last >= 0, got "
            f"{n_gaps}, {gap_length}, {protect_last}."
        )
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        raise TypeError(
            f"{timestamp_column!r} must hold datetimes to place gaps in time order, got dtype "
            f"{df[timestamp_column].dtype}; run validate_and_prepare first."
        )
    times = df[timestamp_column].to_numpy()
    per_series = [
        _in_time_order(rows, times)[_gap_offsets(len(rows), n_gaps, gap_length, protect_last, rng)]
        for rows in df.groupby(id_column, sort=False).indices.values()
    ]
    return np.concatenate(per_series) if per_series else np.array([], dtype=int)


def _in_time_order(rows: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Row positions of one series reordered by timestamp (ties keep the frame order)."""
    return rows[np.argsort(times[rows], kind="stable")]


def _gap_offsets(
    n_rows: int, n_gaps: int, gap_length: int, protect_last: int, rng: np.random.Generator
) -> np.ndarray:
    """Offsets of ``n_gaps`` non-touching interior gaps within one series of ``n_rows``.

    Gaps never include the first row nor the last ``max(protect_last, 1)`` rows, so a dropped
    gap is always a hole inside the series rather than a shorter series. Choosing distinct
    "slack" values and spacing them by ``gap_length`` keeps at least one row between gaps.
    """
    first, end = 1, n_rows - max(protect_last, 1)
    n_choices = (end - first) - n_gaps * gap_length + 1
    if n_choices < n_gaps:
        raise ValueError(
            f"A series of {n_rows} rows cannot hold {n_gaps} gaps of {gap_length} rows "
            f"(protect_last={protect_last})."
        )
    slack = np.sort(rng.choice(n_choices, size=n_gaps, replace=False))
    starts = first + slack + np.arange(n_gaps) * gap_length
    return (starts[:, None] + np.arange(gap_length)).ravel()


def inject_spike(
    df: pd.DataFrame,
    *,
    at: int | pd.Timestamp,
    factor: float,
    column: str = TARGET_COL,
    timestamp_column: str = TIMESTAMP_COL,
) -> pd.DataFrame:
    """Return a copy with ``column`` multiplied by ``factor`` in one row.

    ``at`` is a row position (negative counts from the end) or a timestamp that must match
    exactly one row.
    """
    _require_columns(df, [column], "The frame")
    position = _spike_position(df, at, timestamp_column)
    out = df.copy()
    out[column] = out[column].astype("float64")
    location = out.columns.get_loc(column)
    value = out.iloc[position, location]
    if pd.isna(value):
        raise ValueError(f"{column!r} is NaN at row {position}; a spike needs an observed value.")
    out.iloc[position, location] = value * factor
    return out


def _spike_position(df: pd.DataFrame, at: int | pd.Timestamp, timestamp_column: str) -> int:
    if isinstance(at, int | np.integer) and not isinstance(at, bool):
        if not -len(df) <= at < len(df):
            raise IndexError(f"Row position {at} is outside a frame of {len(df)} rows.")
        return int(at) % len(df)
    _require_columns(df, [timestamp_column], "The frame")
    matches = np.flatnonzero(df[timestamp_column] == pd.Timestamp(at))
    if len(matches) != 1:
        raise ValueError(
            f"Timestamp {at} matches {len(matches)} rows; it must match exactly one "
            "(filter the frame to a single series first)."
        )
    return int(matches[0])
