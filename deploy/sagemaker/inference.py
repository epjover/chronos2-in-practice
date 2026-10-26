"""SageMaker inference handler for Chronos-2: the same contract as the FastAPI service.

SageMaker's PyTorch serving container imports this file and calls four functions around the
model. Keeping the request and response shapes identical to ``deploy/fastapi/app.py`` is the
point: the client code, the tests and the monitoring log do not change when a forecast moves
from a container you run to an endpoint AWS runs, so you can compare the two on latency and cost
rather than on how much rewriting each one cost you.

What the container calls, in order:

- :func:`model_fn` once per worker, with the folder the model artifact was unpacked into.
- :func:`input_fn` per request, to turn the request body into a frame.
- :func:`predict_fn` per request.
- :func:`output_fn` per request, to serialise the result.

Only the error handling differs from the FastAPI app: SageMaker turns an exception into a 400
with the message, so the validation here raises :class:`ValueError` with the same wording rather
than building an HTTP response.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import pandas as pd

from c2p.data import validate_and_prepare
from c2p.models import CHRONOS2, Chronos2Forecaster, load_chronos2
from c2p.monitoring import context_hash
from c2p.schema import ID_COL, PREDICTION_COL, TIMESTAMP_COL, quantile_col

JSON_CONTENT_TYPE = "application/json"
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
MAX_CONTEXT = int(os.environ.get("MAX_CONTEXT", "8192"))
MAX_SERIES = int(os.environ.get("MAX_SERIES", "1000"))
MAX_PREDICTION_LENGTH = 1024
TARGET = "value"

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def model_fn(model_dir: str, context: Any = None) -> Chronos2Forecaster:
    """Load the model once per worker.

    ``model_dir`` holds whatever was in the model artifact. If a checkpoint was packaged, it is
    loaded from disk; otherwise ``MODEL_ID`` (default ``amazon/chronos-2``) is pulled from the
    Hub, which requires the endpoint to have network access.
    """
    del context
    local = os.path.join(model_dir, "chronos-2")
    model_id = local if os.path.isdir(local) else os.environ.get("MODEL_ID", CHRONOS2)
    device = os.environ.get("DEVICE", "cpu")
    logger.info("Loading %s on %s", model_id, device)
    return Chronos2Forecaster(load_chronos2(model_id, device=device))


def input_fn(body: str | bytes, content_type: str = JSON_CONTENT_TYPE) -> dict[str, Any]:
    """Parse and validate the request body; the frame it returns is model-ready."""
    if content_type.split(";")[0].strip() != JSON_CONTENT_TYPE:
        raise ValueError(f"Unsupported content type {content_type!r}; send {JSON_CONTENT_TYPE}.")
    payload = json.loads(body)
    series = payload.get("series")
    if not isinstance(series, list) or not series:
        raise ValueError("'series' must be a non-empty list of {item_id, timestamps, values}.")
    if len(series) > MAX_SERIES:
        raise ValueError(f"{len(series)} series exceeds the limit of {MAX_SERIES} per request.")
    horizon = int(payload.get("prediction_length", 24))
    if not 1 <= horizon <= MAX_PREDICTION_LENGTH:
        raise ValueError(f"prediction_length must be 1..{MAX_PREDICTION_LENGTH}, got {horizon}.")
    quantile_levels = [float(level) for level in payload.get("quantile_levels", DEFAULT_QUANTILES)]
    if not all(0.0 < level < 1.0 for level in quantile_levels):
        raise ValueError(f"quantile_levels must lie in (0, 1), got {quantile_levels}.")

    rows = []
    for entry in series:
        timestamps, values = entry.get("timestamps"), entry.get("values")
        if not isinstance(timestamps, list) or not isinstance(values, list):
            raise ValueError("Each series needs 'timestamps' and 'values' lists.")
        if len(timestamps) != len(values):
            raise ValueError(
                f"Series {entry.get('item_id')!r} has {len(timestamps)} timestamps and "
                f"{len(values)} values."
            )
        rows.append(
            pd.DataFrame(
                {
                    ID_COL: str(entry.get("item_id", "series")),
                    TIMESTAMP_COL: pd.to_datetime(timestamps, utc=True, format="ISO8601"),
                    TARGET: pd.to_numeric(pd.Series(values), errors="coerce"),
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    freq = payload.get("freq", "h")
    # The same checklist the notebooks use: naive UTC, regular grid, gaps as NaN.
    prepared, report = validate_and_prepare(frame, freq=freq, target=TARGET)
    prepared = prepared.groupby(ID_COL, group_keys=False).tail(MAX_CONTEXT)
    return {
        "context": prepared,
        "prediction_length": horizon,
        "quantile_levels": quantile_levels,
        "freq": freq,
        "notes": list(report.notes),
    }


def predict_fn(request: dict[str, Any], model: Chronos2Forecaster) -> dict[str, Any]:
    """Forecast, and record how long the model itself took."""
    started = time.perf_counter()
    forecast = model.predict(
        request["context"],
        prediction_length=request["prediction_length"],
        quantile_levels=request["quantile_levels"],
        freq=request["freq"],
        target=TARGET,
    )
    return {
        "forecast": forecast,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "input_hash": context_hash(request["context"]),
        "quantile_levels": request["quantile_levels"],
        "notes": request["notes"],
    }


def output_fn(prediction: dict[str, Any], accept: str = JSON_CONTENT_TYPE) -> tuple[str, str]:
    """Serialise one forecast per series, in the FastAPI service's response shape."""
    if accept.split(";")[0].strip() not in (JSON_CONTENT_TYPE, "*/*"):
        raise ValueError(f"Unsupported accept type {accept!r}; ask for {JSON_CONTENT_TYPE}.")
    forecast = prediction["forecast"]
    levels = prediction["quantile_levels"]
    forecasts = []
    for item_id, group in forecast.groupby(ID_COL, sort=False):
        group = group.sort_values(TIMESTAMP_COL)
        forecasts.append(
            {
                "item_id": str(item_id),
                "timestamps": [stamp.isoformat() + "Z" for stamp in group[TIMESTAMP_COL]],
                "predictions": group[PREDICTION_COL].round(4).tolist(),
                "quantiles": {
                    str(level): group[quantile_col(level)].round(4).tolist() for level in levels
                },
            }
        )
    body = {
        "input_hash": prediction["input_hash"],
        "latency_ms": round(prediction["latency_ms"], 2),
        "quantile_levels": levels,
        "notes": prediction["notes"],
        "forecasts": forecasts,
    }
    return json.dumps(body), JSON_CONTENT_TYPE
