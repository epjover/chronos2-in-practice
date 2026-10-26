"""Tests for the FastAPI forecasting service in ``deploy/fastapi/app.py``.

Fast tests replace Chronos-2 with a deterministic fake forecaster (injected through
``create_app(load_forecaster=...)``), so they need no model download. The slow test serves the
real model on CPU. The deploy dependencies come from the ``deploy`` group
(``uv sync --group deploy``); without them the module is skipped.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterator, Sequence
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset

from c2p.schema import (
    ID_COL,
    PREDICTION_COL,
    TARGET_COL,
    TARGET_NAME_COL,
    TIMESTAMP_COL,
    quantile_col,
)

pytest.importorskip("fastapi", reason="install the deploy group: uv sync --group deploy")
pytest.importorskip("httpx", reason="install the deploy group: uv sync --group deploy")

from fastapi.testclient import TestClient

APP_PATH = Path(__file__).resolve().parents[1] / "deploy" / "fastapi" / "app.py"
START = pd.Timestamp("2026-01-05 00:00")  # naive UTC


def _load_app_module() -> ModuleType:
    """Import ``deploy/fastapi/app.py`` under a unique name (it is not a package)."""
    spec = importlib.util.spec_from_file_location("c2p_forecast_api", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # pydantic resolves the annotations through sys.modules
    spec.loader.exec_module(module)
    return module


api = _load_app_module()


class FakeForecaster:
    """Stand-in for Chronos-2: repeats the last observed value, quantiles at fixed offsets.

    It records every context it receives so tests can check what the model would have seen.
    """

    name = "fake"

    def __init__(self) -> None:
        self.contexts: list[pd.DataFrame] = []

    def predict(
        self,
        context: pd.DataFrame,
        *,
        prediction_length: int,
        quantile_levels: Sequence[float],
        freq: str,
        target: str | Sequence[str] = TARGET_COL,
        future_covariates: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        assert isinstance(target, str)
        self.contexts.append(context.copy())
        offset = to_offset(freq)
        blocks = []
        for item_id, block in context.groupby(ID_COL, sort=False):
            last_value = float(block[target].dropna().iloc[-1])
            frame = pd.DataFrame(
                {
                    ID_COL: item_id,
                    TIMESTAMP_COL: pd.date_range(
                        block[TIMESTAMP_COL].iloc[-1] + offset, periods=prediction_length, freq=freq
                    ),
                    TARGET_NAME_COL: target,
                    PREDICTION_COL: last_value,
                }
            )
            for level in quantile_levels:
                frame[quantile_col(level)] = last_value + (level - 0.5) * 10
            blocks.append(frame)
        return pd.concat(blocks, ignore_index=True)


# --------------------------------------------------------------------------------------------
# Helpers and fixtures
# --------------------------------------------------------------------------------------------


def hourly_series(
    item_id: str,
    n: int = 48,
    *,
    start: pd.Timestamp = START,
    drop: Sequence[int] = (),
    nulls: Sequence[int] = (),
    tz: str = "UTC",
    level: float = 100.0,
) -> dict[str, Any]:
    """A JSON-ready hourly series; ``drop`` removes rows, ``nulls`` sends ``null`` values."""
    timestamps = pd.date_range(start.tz_localize("UTC"), periods=n, freq="h").tz_convert(tz)
    values = [level + float(np.sin(2 * np.pi * i / 24)) for i in range(n)]
    keep = [i for i in range(n) if i not in set(drop)]
    return {
        "item_id": item_id,
        "timestamps": [timestamps[i].isoformat() for i in keep],
        "values": [None if i in set(nulls) else values[i] for i in keep],
    }


def forecast_request(
    series: Sequence[dict[str, Any]],
    *,
    prediction_length: int = 6,
    freq: str = "h",
    quantile_levels: Sequence[float] = (0.1, 0.5, 0.9),
) -> dict[str, Any]:
    return {
        "series": list(series),
        "prediction_length": prediction_length,
        "freq": freq,
        "quantile_levels": list(quantile_levels),
    }


@pytest.fixture
def fake() -> FakeForecaster:
    return FakeForecaster()


def make_client(fake: FakeForecaster, **settings: Any) -> TestClient:
    app = api.create_app(
        api.Settings(model_id="test/fake-model", **settings), load_forecaster=lambda _: fake
    )
    return TestClient(app)


@pytest.fixture
def client(fake: FakeForecaster) -> Iterator[TestClient]:
    with make_client(fake) as test_client:  # the context manager runs the lifespan
        yield test_client


def post_ok(client: TestClient, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def hash_of(client: TestClient, payload: dict[str, Any]) -> str:
    return post_ok(client, payload)["input_hash"]


# --------------------------------------------------------------------------------------------
# Health and lifecycle
# --------------------------------------------------------------------------------------------


def test_health_reports_model_device_and_version(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_id": "test/fake-model",
        "device": "cpu",
        "chronos_version": version("chronos-forecasting"),
    }


def test_returns_503_while_the_model_is_not_loaded(fake: FakeForecaster) -> None:
    client = make_client(fake)  # no `with`: the lifespan (model loading) does not run
    assert client.get("/health").status_code == 503
    assert client.post("/forecast", json=forecast_request([hourly_series("a")])).status_code == 503


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_ID", "/models/finetuned")
    monkeypatch.setenv("DEVICE", "CPU")
    monkeypatch.setenv("MAX_CONTEXT", "2048")
    settings = api.Settings.from_env()
    assert (settings.model_id, settings.device, settings.max_context) == (
        "/models/finetuned",
        "cpu",
        2048,
    )


@pytest.mark.parametrize(
    ("variable", "value"),
    [("DEVICE", "gpu"), ("MAX_CONTEXT", "0"), ("MAX_CONTEXT", "9000"), ("MAX_CONTEXT", "lots")],
)
def test_settings_reject_bad_environment(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=variable):
        api.Settings.from_env()


# --------------------------------------------------------------------------------------------
# Forecast responses
# --------------------------------------------------------------------------------------------


def test_forecast_response_schema(client: TestClient) -> None:
    payload = forecast_request(
        [hourly_series("b", level=200.0), hourly_series("a")],
        prediction_length=6,
        quantile_levels=[0.9, 0.1, 0.5],
    )
    body = post_ok(client, payload)

    assert body["model_id"] == "test/fake-model"
    assert len(body["input_hash"]) == 64
    int(body["input_hash"], 16)
    assert body["latency_ms"] >= 0
    assert body["quantile_levels"] == [0.1, 0.5, 0.9]  # sorted
    assert (body["freq"], body["prediction_length"]) == ("h", 6)
    assert [entry["item_id"] for entry in body["forecasts"]] == ["b", "a"]  # request order
    last_input = pd.Timestamp(payload["series"][0]["timestamps"][-1])
    for entry in body["forecasts"]:
        assert set(entry) == {"item_id", "timestamps", "predictions", "quantiles"}
        assert len(entry["timestamps"]) == len(entry["predictions"]) == 6
        assert all(timestamp.endswith("Z") for timestamp in entry["timestamps"])  # UTC
        assert pd.Timestamp(entry["timestamps"][0]) == last_input + pd.Timedelta(hours=1)
        assert list(entry["quantiles"]) == ["0.1", "0.5", "0.9"]
        assert all(len(values) == 6 for values in entry["quantiles"].values())
    b_last = payload["series"][0]["values"][-1]
    assert body["forecasts"][0]["predictions"] == pytest.approx([b_last] * 6)


def test_missing_timestamps_and_nulls_become_nan_rows(
    client: TestClient, fake: FakeForecaster
) -> None:
    series = hourly_series("a", n=48, drop=[10, 11, 12], nulls=[20])
    body = post_ok(client, forecast_request([series]))

    context = fake.contexts[-1]
    assert len(context) == 48
    expected = pd.date_range(START, periods=48, freq="h")
    assert context[TIMESTAMP_COL].tolist() == expected.tolist()
    assert context[TARGET_COL].isna().to_numpy().nonzero()[0].tolist() == [10, 11, 12, 20]
    summary = body["input_summary"]
    assert summary["n_points"] == 48
    assert summary["n_missing_timestamps"] == 3
    assert summary["n_missing_values"] == 4


def test_utc_offsets_become_naive_utc(client: TestClient, fake: FakeForecaster) -> None:
    series = hourly_series("a", n=24, tz="Europe/Madrid")  # "+01:00" offsets
    assert series["timestamps"][0].endswith("+01:00")
    body = post_ok(client, forecast_request([series]))

    context = fake.contexts[-1]
    assert context[TIMESTAMP_COL].dt.tz is None
    assert context[TIMESTAMP_COL].iloc[0] == START
    assert pd.Timestamp(body["forecasts"][0]["timestamps"][0]) == pd.Timestamp(
        "2026-01-06 00:00", tz="UTC"
    )


def test_naive_timestamps_are_read_as_utc(client: TestClient, fake: FakeForecaster) -> None:
    series = hourly_series("a", n=24)
    series["timestamps"] = [timestamp.removesuffix("+00:00") for timestamp in series["timestamps"]]
    body = post_ok(client, forecast_request([series]))

    assert fake.contexts[-1][TIMESTAMP_COL].iloc[0] == START
    assert any("assumed to be UTC" in note for note in body["input_summary"]["notes"])


def test_long_series_keep_the_most_recent_steps(fake: FakeForecaster) -> None:
    with make_client(fake, max_context=24) as client:
        body = post_ok(client, forecast_request([hourly_series("a", n=48)]))

    context = fake.contexts[-1]
    assert len(context) == 24
    assert context[TIMESTAMP_COL].iloc[0] == START + pd.Timedelta(hours=24)
    assert body["input_summary"]["n_trimmed"] == 24


# --------------------------------------------------------------------------------------------
# Input hash
# --------------------------------------------------------------------------------------------


def test_hash_is_stable_across_equivalent_requests(client: TestClient) -> None:
    a, b = hourly_series("a", nulls=[5]), hourly_series("b", level=50.0)
    reference = hash_of(client, forecast_request([a, b]))

    equivalent = {
        "same request again": forecast_request([a, b]),
        "series order": forecast_request([b, a]),
        "quantile order": forecast_request([a, b], quantile_levels=[0.9, 0.5, 0.1]),
        "UTC offsets": forecast_request(
            [hourly_series("a", nulls=[5], tz="Asia/Kolkata"), b]  # "+05:30" offsets
        ),
        "absent row instead of null": forecast_request([hourly_series("a", drop=[5]), b]),
    }
    for case, payload in equivalent.items():
        assert hash_of(client, payload) == reference, case


def test_hash_changes_with_the_model_input(client: TestClient) -> None:
    a, b = hourly_series("a"), hourly_series("b", level=50.0)
    reference = hash_of(client, forecast_request([a, b]))

    changed_value = json.loads(json.dumps(a))
    changed_value["values"][-1] += 0.001
    different = {
        "a value": forecast_request([changed_value, b]),
        "an item id": forecast_request([{**a, "item_id": "c"}, b]),
        "a missing value": forecast_request([hourly_series("a", nulls=[5]), b]),
        "the horizon": forecast_request([a, b], prediction_length=7),
        "the quantile levels": forecast_request([a, b], quantile_levels=[0.1, 0.5]),
    }
    for case, payload in different.items():
        assert hash_of(client, payload) != reference, case


def test_hash_follows_the_documented_canonical_form(client: TestClient) -> None:
    payload = forecast_request(
        [
            {
                "item_id": "a",
                "timestamps": ["2026-01-05T00:00:00Z", "2026-01-05T02:00:00Z"],
                "values": [1.5, -0.0],
            }
        ],
        prediction_length=2,
        quantile_levels=[0.5, 0.1],
    )
    header = {
        "version": 1,
        "freq": "h",
        "prediction_length": 2,
        "quantile_levels": [0.1, 0.5],
        "series": [["a", 3]],
    }
    start_us = int(pd.Timestamp("2026-01-05").value // 1000)
    hour_us = 3_600_000_000
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    digest.update(np.array([start_us + i * hour_us for i in range(3)], dtype="<i8").tobytes())
    digest.update(np.array([1.5, np.nan, 0.0], dtype="<f8").tobytes())

    assert hash_of(client, payload) == digest.hexdigest()


# --------------------------------------------------------------------------------------------
# Validation errors
# --------------------------------------------------------------------------------------------


def _with(series: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {**series, **changes}


BASE = hourly_series("a", n=24)
INVALID_REQUESTS: dict[str, tuple[dict[str, Any], str]] = {
    "no series": (forecast_request([]), "at least 1 item"),
    "length mismatch": (forecast_request([_with(BASE, values=BASE["values"][:-1])]), "values"),
    "all values null": (forecast_request([_with(BASE, values=[None] * 24)]), "no observed"),
    "blank item id": (forecast_request([_with(BASE, item_id="  ")]), "blank"),
    "duplicated item ids": (forecast_request([BASE, BASE]), "unique"),
    "duplicated timestamps": (
        forecast_request([_with(BASE, timestamps=[BASE["timestamps"][0]] * 24)]),
        "duplicated timestamps",
    ),
    "mixed offsets in a series": (
        forecast_request(
            [_with(BASE, timestamps=[BASE["timestamps"][0][:-6], *BASE["timestamps"][1:]])]
        ),
        "mixes",
    ),
    "mixed offsets across series": (
        forecast_request(
            [
                BASE,
                _with(
                    BASE,
                    item_id="b",
                    timestamps=[timestamp[:-6] for timestamp in BASE["timestamps"]],
                ),
            ]
        ),
        "one style",
    ),
    "timestamp off the grid": (
        forecast_request(
            [_with(BASE, timestamps=[*BASE["timestamps"][:-1], "2026-01-06T10:30:00+00:00"])]
        ),
        "not on the regular",
    ),
    "series too long": (
        forecast_request(
            [
                {
                    "item_id": "a",
                    "timestamps": ["2020-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
                    "values": [1.0, 2.0],
                }
            ]
        ),
        "steps of 'h'",
    ),
    "horizon zero": (forecast_request([BASE], prediction_length=0), "greater than or equal"),
    "horizon too long": (forecast_request([BASE], prediction_length=1025), "less than or equal"),
    "unknown freq": (forecast_request([BASE], freq="hourly"), "frequency"),
    "uppercase H": (forecast_request([BASE], freq="H"), "frequency"),
    "quantile 0": (forecast_request([BASE], quantile_levels=[0.0, 0.5]), "greater than 0"),
    "quantile 1": (forecast_request([BASE], quantile_levels=[0.5, 1.0]), "less than 1"),
    "no quantiles": (forecast_request([BASE], quantile_levels=[]), "at least 1 item"),
    "duplicated quantiles": (
        forecast_request([BASE], quantile_levels=[0.5, 0.5]),
        "duplicates",
    ),
    "unknown field": ({**forecast_request([BASE]), "horizon": 3}, "Extra inputs"),
}


@pytest.mark.parametrize("case", list(INVALID_REQUESTS))
def test_invalid_requests_get_422(client: TestClient, case: str) -> None:
    payload, message = INVALID_REQUESTS[case]
    response = client.post("/forecast", json=payload)
    assert response.status_code == 422, response.text
    errors = response.json()["detail"]
    assert all(set(error) == {"type", "loc", "msg"} for error in errors)  # input not echoed
    assert message in " ".join(error["msg"] for error in errors)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_values_get_422(client: TestClient, token: str) -> None:
    body = json.dumps(forecast_request([BASE])).replace(
        json.dumps(BASE["values"][3]), token, 1
    )  # the JSON extensions Python's parser accepts
    response = client.post("/forecast", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422, response.text
    assert "finite number" in response.text


def test_too_many_series_get_422(client: TestClient) -> None:
    series = [
        {"item_id": f"s{i}", "timestamps": ["2026-01-05T00:00:00Z"], "values": [1.0]}
        for i in range(api.MAX_SERIES + 1)
    ]
    response = client.post("/forecast", json=forecast_request(series))
    assert response.status_code == 422
    assert f"at most {api.MAX_SERIES}" in response.text
    assert len(response.content) < 1_000  # the oversized request is not echoed back


# --------------------------------------------------------------------------------------------
# Real model (slow)
# --------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_real_chronos2_on_cpu() -> None:
    app = api.create_app(api.Settings())  # downloads amazon/chronos-2 on first use
    payload = forecast_request(
        [hourly_series("a", n=168, drop=[100, 101]), hourly_series("b", n=168, level=20.0)],
        prediction_length=24,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    with TestClient(app) as client:
        assert client.get("/health").json()["model_id"] == "amazon/chronos-2"
        body = post_ok(client, payload)

    assert body["input_summary"]["n_missing_timestamps"] == 2
    for entry, level in zip(body["forecasts"], (100.0, 20.0), strict=True):
        low, median, high = (np.array(entry["quantiles"][key]) for key in ("0.1", "0.5", "0.9"))
        assert len(median) == 24
        assert np.all(low <= median) and np.all(median <= high)
        assert entry["predictions"] == pytest.approx(median.tolist())
        assert abs(median.mean() - level) < 2.0  # a daily sine of amplitude 1 around `level`


def test_settings_read_torch_threads_and_default_to_the_available_cores(monkeypatch):
    monkeypatch.delenv("TORCH_NUM_THREADS", raising=False)
    assert api.Settings.from_env().torch_threads >= 1
    monkeypatch.setenv("TORCH_NUM_THREADS", "3")
    assert api.Settings.from_env().torch_threads == 3
    monkeypatch.setenv("TORCH_NUM_THREADS", "many")
    with pytest.raises(ValueError, match="TORCH_NUM_THREADS"):
        api.Settings.from_env()
