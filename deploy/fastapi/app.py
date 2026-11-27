"""Chronos-2 forecasting service: a small FastAPI app that serves probabilistic forecasts on CPU.

Run it locally from the repository root::

    uv run --group deploy uvicorn app:app --app-dir deploy/fastapi --port 8000

or build the Docker image described in ``deploy/fastapi/README.md``.

Endpoints:

- ``GET /health``: model id, device and library version once the model is loaded.
- ``POST /forecast``: quantile forecasts for up to ``MAX_SERIES`` series per request.

Every request goes through ``c2p.data.validate_and_prepare`` before it reaches the model:
timestamps become naive UTC, missing timestamps become NaN rows on a regular grid and only the
most recent ``MAX_CONTEXT`` steps of each series are kept. The response carries a SHA-256 hash
of that canonical model input, so a monitoring log can tie every forecast to exactly what the
model saw.

Environment variables (all optional):

- ``MODEL_ID``: Hugging Face id or local folder of a Chronos-2 model (default ``amazon/chronos-2``;
  a fine-tuned checkpoint works too).
- ``DEVICE``: ``cpu`` (default) or ``cuda``.
- ``MAX_CONTEXT``: most recent steps per series fed to the model (1..8192, default 8192).
- ``LOG_LEVEL``: Python logging level (default ``INFO``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib.metadata import version
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

from c2p.data import DataReport, validate_and_prepare
from c2p.models import CHRONOS2, Chronos2Forecaster, load_chronos2
from c2p.schema import ID_COL, PREDICTION_COL, TARGET_COL, TIMESTAMP_COL, Forecaster, quantile_col

logger = logging.getLogger("c2p.api")

# Request limits. Grid sizes count the timestamps inserted for gaps, so a request cannot make
# the server build a huge grid from two far-apart points.
MAX_SERIES = 1_000
MAX_POINTS_PER_SERIES = 32_768  # grid steps per series accepted; older ones are trimmed
MAX_TOTAL_POINTS = 2_000_000  # grid steps per request, all series together
MAX_PREDICTION_LENGTH = 1_024  # Chronos-2's native maximum horizon
MAX_QUANTILES = 99
MAX_ID_LENGTH = 200
MODEL_CONTEXT_LENGTH = 8_192  # Chronos-2 reads at most this many past steps
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
HASH_FORMAT_VERSION = 1


# --------------------------------------------------------------------------------------------
# Settings and model loading
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Service configuration, read once from environment variables by :meth:`from_env`."""

    model_id: str = CHRONOS2
    device: Literal["cpu", "cuda"] = "cpu"
    max_context: int = MODEL_CONTEXT_LENGTH
    log_level: str = "INFO"
    torch_threads: int = 0  # 0 = every core available to the process

    def __post_init__(self) -> None:
        if self.torch_threads < 0:
            raise ValueError(f"TORCH_NUM_THREADS must be >= 0, got {self.torch_threads}.")
        if self.torch_threads == 0:
            try:
                object.__setattr__(self, "torch_threads", len(os.sched_getaffinity(0)))
            except AttributeError:
                object.__setattr__(self, "torch_threads", os.cpu_count() or 1)
        if not self.model_id.strip():
            raise ValueError("MODEL_ID must not be empty.")
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"DEVICE must be 'cpu' or 'cuda', got {self.device!r}.")
        if not 1 <= self.max_context <= MODEL_CONTEXT_LENGTH:
            raise ValueError(
                f"MAX_CONTEXT must be in 1..{MODEL_CONTEXT_LENGTH}, got {self.max_context}."
            )
        if not isinstance(logging.getLevelName(self.log_level), int):
            raise ValueError(f"LOG_LEVEL={self.log_level!r} is not a logging level.")

    @classmethod
    def from_env(cls) -> Settings:
        """Build the settings from ``MODEL_ID``, ``DEVICE``, ``MAX_CONTEXT`` and ``LOG_LEVEL``."""
        max_context = os.environ.get("MAX_CONTEXT", str(MODEL_CONTEXT_LENGTH)).strip()
        if not max_context.isdigit():
            raise ValueError(f"MAX_CONTEXT must be a positive integer, got {max_context!r}.")
        threads = os.environ.get("TORCH_NUM_THREADS", "0").strip()
        if not threads.isdigit():
            raise ValueError(f"TORCH_NUM_THREADS must be a non-negative integer, got {threads!r}.")
        return cls(
            model_id=os.environ.get("MODEL_ID", CHRONOS2).strip(),
            device=os.environ.get("DEVICE", "cpu").strip().lower(),  # type: ignore[arg-type]
            max_context=int(max_context),
            log_level=os.environ.get("LOG_LEVEL", "INFO").strip().upper(),
            torch_threads=int(threads),
        )


@dataclass(frozen=True)
class LoadedModel:
    """The forecaster loaded at startup and the settings it was loaded with."""

    forecaster: Forecaster
    settings: Settings


ForecasterLoader = Callable[[Settings], Forecaster]


def load_chronos2_forecaster(settings: Settings) -> Forecaster:
    """Default loader: Chronos-2 (or a fine-tuned checkpoint) on ``settings.device``.

    Sets PyTorch's CPU thread count explicitly (``TORCH_NUM_THREADS``, default: every core the
    process may use). Containers can leave OpenMP at one thread, which makes CPU inference
    several times slower than the hardware allows and is invisible from outside.
    """
    import torch

    torch.set_num_threads(settings.torch_threads)
    logger.info("PyTorch CPU threads: %d", torch.get_num_threads())
    pipeline = load_chronos2(settings.model_id, device=settings.device)
    return Chronos2Forecaster(pipeline, name=settings.model_id)


# --------------------------------------------------------------------------------------------
# Request and response schemas
# --------------------------------------------------------------------------------------------

QuantileLevel = Annotated[float, Field(gt=0.0, lt=1.0)]


class SeriesInput(BaseModel):
    """One series: equally long ``timestamps`` and ``values``; ``null`` marks a missing value.

    Timestamps may carry UTC offsets (``2026-03-29T02:00:00+02:00``) or be naive, in which case
    they are read as UTC. Missing timestamps are allowed: they become missing values.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: Annotated[str, Field(min_length=1, max_length=MAX_ID_LENGTH)]
    timestamps: Annotated[list[datetime], Field(min_length=1, max_length=MAX_POINTS_PER_SERIES)]
    values: Annotated[
        list[FiniteFloat | None], Field(min_length=1, max_length=MAX_POINTS_PER_SERIES)
    ]

    @model_validator(mode="after")
    def _check_series(self) -> SeriesInput:
        name = repr(self.item_id)
        if not self.item_id.strip():
            raise ValueError("item_id must not be blank.")
        if len(self.timestamps) != len(self.values):
            raise ValueError(
                f"Series {name} has {len(self.timestamps)} timestamps but {len(self.values)} "
                "values; send one value (or null) per timestamp."
            )
        if all(value is None for value in self.values):
            raise ValueError(f"Series {name} has no observed values; send at least one number.")
        if len({_is_aware(timestamp) for timestamp in self.timestamps}) > 1:
            raise ValueError(
                f"Series {name} mixes timestamps with and without UTC offsets; use one style."
            )
        if len(set(self.timestamps)) != len(self.timestamps):
            raise ValueError(f"Series {name} has duplicated timestamps; send each instant once.")
        return self


class ForecastRequest(BaseModel):
    """Series to forecast and how: horizon, grid frequency and quantile levels."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "series": [
                        {
                            "item_id": "ES",
                            "timestamps": [
                                "2018-12-31T20:00:00Z",
                                "2018-12-31T21:00:00Z",
                                "2018-12-31T23:00:00Z",
                            ],
                            "values": [27510.0, 25934.0, None],
                        }
                    ],
                    "prediction_length": 24,
                    "freq": "h",
                    "quantile_levels": [0.1, 0.5, 0.9],
                }
            ]
        },
    )

    series: Annotated[list[SeriesInput], Field(min_length=1, max_length=MAX_SERIES)]
    prediction_length: Annotated[int, Field(ge=1, le=MAX_PREDICTION_LENGTH)]
    freq: Annotated[str, Field(min_length=1, max_length=20, examples=["h", "15min", "D"])]
    quantile_levels: Annotated[
        list[QuantileLevel], Field(min_length=1, max_length=MAX_QUANTILES)
    ] = list(DEFAULT_QUANTILES)

    @field_validator("freq")
    @classmethod
    def _check_freq(cls, value: str) -> str:
        try:
            return to_offset(value).freqstr
        except ValueError as error:
            raise ValueError(
                f"freq={value!r} is not a pandas frequency string such as 'h', '15min' or 'D'."
            ) from error

    @field_validator("quantile_levels")
    @classmethod
    def _sort_levels(cls, levels: list[float]) -> list[float]:
        if len(set(levels)) != len(levels):
            raise ValueError(f"quantile_levels has duplicates: {levels}.")
        return sorted(levels)

    @model_validator(mode="after")
    def _check_request(self) -> ForecastRequest:
        ids = [series.item_id for series in self.series]
        if len(set(ids)) != len(ids):
            repeated = sorted({item for item in ids if ids.count(item) > 1})[:5]
            raise ValueError(f"item_id values must be unique; repeated: {repeated}.")
        if len({_is_aware(series.timestamps[0]) for series in self.series}) > 1:
            raise ValueError(
                "Some series have timestamps with UTC offsets and others without; use one style."
            )
        _check_grid_sizes(self.series, to_offset(self.freq))
        return self


class SeriesForecast(BaseModel):
    """Forecast of one series: UTC timestamps, point forecast (median) and quantiles."""

    item_id: str
    timestamps: list[datetime]
    predictions: list[float]
    quantiles: dict[str, list[float]]


class InputSummary(BaseModel):
    """What the data checklist did to the request before forecasting."""

    n_series: int
    n_points: int
    n_missing_timestamps: int
    n_missing_values: int
    n_trimmed: int
    notes: list[str]


class ForecastResponse(BaseModel):
    """Forecasts plus the metadata a monitoring log needs."""

    model_id: str
    input_hash: str
    latency_ms: float
    freq: str
    prediction_length: int
    quantile_levels: list[float]
    input_summary: InputSummary
    forecasts: list[SeriesForecast]


class HealthResponse(BaseModel):
    """Liveness and identity of the loaded model."""

    status: Literal["ok"]
    model_id: str
    device: str
    chronos_version: str


def _is_aware(timestamp: datetime) -> bool:
    return timestamp.utcoffset() is not None


def _check_grid_sizes(series: Sequence[SeriesInput], offset: BaseOffset) -> None:
    """Bound the size of the regular grid the request expands to (gaps included)."""
    sizes = {item.item_id: _grid_size(item.timestamps, offset) for item in series}
    too_long = {item_id: size for item_id, size in sizes.items() if size > MAX_POINTS_PER_SERIES}
    if too_long:
        examples = dict(list(too_long.items())[:3])
        raise ValueError(
            f"Series span more than {MAX_POINTS_PER_SERIES:,} steps of {offset.freqstr!r} "
            f"(gaps included), e.g. {examples}. Send only recent history: the model reads at "
            f"most {MODEL_CONTEXT_LENGTH:,} steps."
        )
    if (total := sum(sizes.values())) > MAX_TOTAL_POINTS:
        raise ValueError(
            f"The request spans {total:,} grid steps in total (gaps included); the limit is "
            f"{MAX_TOTAL_POINTS:,}. Split it into several requests."
        )


def _grid_size(timestamps: Sequence[datetime], offset: BaseOffset) -> int:
    """Approximate number of grid steps from the first to the last timestamp.

    Exact for fixed frequencies such as ``h``; based on the first step for calendar ones such
    as ``MS``.
    """
    first, last = pd.Timestamp(min(timestamps)), pd.Timestamp(max(timestamps))
    step = (first + offset) - first
    if step <= timedelta(0):
        raise ValueError(f"freq={offset.freqstr!r} does not move time forward.")
    return int((last - first) // step) + 1


# --------------------------------------------------------------------------------------------
# Forecasting steps
# --------------------------------------------------------------------------------------------


class InvalidInputError(ValueError):
    """The request is well formed but its data cannot be put on a regular grid."""


def request_to_frame(request: ForecastRequest) -> pd.DataFrame:
    """Flatten the request into a long frame (``item_id``, ``timestamp``, ``target``).

    Timestamps with UTC offsets become a UTC-aware column and naive ones stay naive;
    ``validate_and_prepare`` turns both into naive UTC. ``null`` values become NaN.
    """
    lengths = [len(series.timestamps) for series in request.series]
    ids = np.repeat([series.item_id for series in request.series], lengths)
    timestamps = [timestamp for series in request.series for timestamp in series.timestamps]
    values = [value for series in request.series for value in series.values]  # noqa: PD011
    aware = _is_aware(request.series[0].timestamps[0])
    return pd.DataFrame(
        {
            ID_COL: pd.array(ids, dtype="str"),
            TIMESTAMP_COL: pd.to_datetime(timestamps, utc=aware),
            TARGET_COL: np.array(values, dtype="float64"),  # None -> NaN
        }
    )


def prepare_context(
    request: ForecastRequest, *, max_context: int
) -> tuple[pd.DataFrame, DataReport]:
    """Run the data checklist of Part 1 on the request; raise ``InvalidInputError`` on bad data."""
    try:
        return validate_and_prepare(
            request_to_frame(request),
            freq=request.freq,
            target=TARGET_COL,
            duplicates="raise",
            max_length=max_context,
        )
    except (TypeError, ValueError) as error:
        raise InvalidInputError(str(error)) from error


def input_hash(
    context: pd.DataFrame, *, prediction_length: int, freq: str, quantile_levels: Sequence[float]
) -> str:
    """SHA-256 of the canonical model input: what the model sees, not the request bytes.

    The digest covers a JSON header (sorted keys: format version, ``freq``,
    ``prediction_length``, sorted ``quantile_levels`` and the id and length of every series in
    id order), then all timestamps as little-endian int64 microseconds since the Unix epoch
    (UTC) and all values as little-endian float64 with NaN for missing values. Requests that
    differ only in JSON formatting, series or quantile order, UTC offsets, ``null`` versus an
    absent timestamp, or history older than the context window get the same hash. The model id
    is not part of the hash: log it next to the hash.
    """
    lengths = context.groupby(ID_COL, sort=False).size()  # context is sorted by id
    header = {
        "version": HASH_FORMAT_VERSION,
        "freq": freq,
        "prediction_length": prediction_length,
        "quantile_levels": sorted(float(level) for level in quantile_levels),
        "series": [[str(item_id), int(n)] for item_id, n in lengths.items()],
    }
    timestamps = context[TIMESTAMP_COL].dt.as_unit("us").to_numpy().astype("<i8")
    values = context[TARGET_COL].to_numpy(dtype="<f8") + 0.0  # + 0.0 turns -0.0 into 0.0
    values[np.isnan(values)] = np.nan  # a single NaN bit pattern
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    digest.update(timestamps.tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def summarize_input(context: pd.DataFrame, report: DataReport) -> InputSummary:
    """Describe the validated input (series, rows, span, gaps) for the response envelope."""
    return InputSummary(
        n_series=report.n_series,
        n_points=len(context),
        n_missing_timestamps=report.n_missing_timestamps,
        n_missing_values=report.n_missing_values.get(TARGET_COL, 0),
        n_trimmed=report.n_trimmed,
        notes=report.notes,
    )


def to_series_forecasts(
    forecast: pd.DataFrame, *, item_order: Sequence[str], quantile_levels: Sequence[float]
) -> list[SeriesForecast]:
    """Split a forecast frame into one response entry per series, in request order."""
    blocks = dict(list(forecast.groupby(ID_COL, sort=False)))
    return [
        SeriesForecast(
            item_id=item_id,
            timestamps=[
                timestamp.to_pydatetime()
                for timestamp in blocks[item_id][TIMESTAMP_COL].dt.tz_localize("UTC")
            ],
            predictions=blocks[item_id][PREDICTION_COL].tolist(),
            quantiles={
                quantile_col(level): blocks[item_id][quantile_col(level)].tolist()
                for level in quantile_levels
            },
        )
        for item_id in item_order
    ]


# --------------------------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------------------------


async def get_model(request: Request) -> LoadedModel:
    """FastAPI dependency: the model loaded at startup (503 while it is not available).

    Async so that FastAPI resolves it on the event loop instead of a worker thread.
    """
    model: LoadedModel | None = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="The model is not loaded.")
    return model


async def validation_error_handler(request: Request, error: RequestValidationError) -> JSONResponse:
    """422 with the type, location and message of each error, without echoing the input.

    FastAPI's default handler echoes the offending input, which fails to render for NaN or
    Infinity (not valid JSON, so the client would get a 500) and can copy a whole oversized
    request into the response.
    """
    details = [
        {key: item[key] for key in ("type", "loc", "msg") if key in item} for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(details)})


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(level)
    # The per-request data report is returned in the response; keep it out of the server log.
    logging.getLogger("c2p.data").setLevel(logging.WARNING)


def create_app(
    settings: Settings | None = None,
    *,
    load_forecaster: ForecasterLoader = load_chronos2_forecaster,
) -> FastAPI:
    """Build the app. The model is loaded once at startup by ``load_forecaster(settings)``.

    Tests pass a fake ``load_forecaster`` so they run without downloading the model.
    """
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _configure_logging(settings.log_level)
        started = time.perf_counter()
        app.state.model = LoadedModel(forecaster=load_forecaster(settings), settings=settings)
        logger.info(
            "Model %s ready on %s in %.1f s.",
            settings.model_id,
            settings.device,
            time.perf_counter() - started,
        )
        yield
        app.state.model = None

    app = FastAPI(
        title="Chronos-2 forecasting service",
        version="0.1.0",
        summary="Probabilistic forecasts from Chronos-2, on a regular grid with gaps as NaN.",
        lifespan=lifespan,
    )
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    @app.get("/health")
    async def health(model: Annotated[LoadedModel, Depends(get_model)]) -> HealthResponse:
        # Async on purpose: it answers on the event loop even while every worker thread is busy
        # with forecasts, so health checks do not time out under load.
        return HealthResponse(
            status="ok",
            model_id=model.settings.model_id,
            device=model.settings.device,
            chronos_version=version("chronos-forecasting"),
        )

    @app.post("/forecast")
    def forecast(
        request: ForecastRequest, model: Annotated[LoadedModel, Depends(get_model)]
    ) -> ForecastResponse:
        # A plain `def` endpoint runs in a worker thread, so inference does not block /health.
        started = time.perf_counter()
        try:
            context, report = prepare_context(request, max_context=model.settings.max_context)
        except InvalidInputError as error:
            raise HTTPException(
                status_code=422,
                detail=[{"type": "invalid_series", "loc": ["body", "series"], "msg": str(error)}],
            ) from error
        digest = input_hash(
            context,
            prediction_length=request.prediction_length,
            freq=request.freq,
            quantile_levels=request.quantile_levels,
        )
        result = model.forecaster.predict(
            context,
            prediction_length=request.prediction_length,
            quantile_levels=request.quantile_levels,
            freq=request.freq,
            target=TARGET_COL,
        )
        forecasts = to_series_forecasts(
            result,
            item_order=[series.item_id for series in request.series],
            quantile_levels=request.quantile_levels,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "forecast input_hash=%s model_id=%s n_series=%d n_points=%d prediction_length=%d "
            "latency_ms=%.1f",
            digest,
            model.settings.model_id,
            report.n_series,
            len(context),
            request.prediction_length,
            latency_ms,
        )
        return ForecastResponse(
            model_id=model.settings.model_id,
            input_hash=digest,
            latency_ms=round(latency_ms, 3),
            freq=request.freq,
            prediction_length=request.prediction_length,
            quantile_levels=request.quantile_levels,
            input_summary=summarize_input(context, report),
            forecasts=forecasts,
        )

    return app


app = create_app()
