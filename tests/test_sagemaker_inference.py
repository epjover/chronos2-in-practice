"""Tests for the SageMaker handler in ``deploy/sagemaker/inference.py``.

The handler's job is to speak the same JSON as the FastAPI service while the container around it
is completely different, so these tests check the contract rather than the forecast: what the
four entry points accept, what they reject, and that the data checklist still runs between the
request and the model. A deterministic fake stands in for Chronos-2, so none of this downloads
weights.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from c2p.schema import ID_COL, PREDICTION_COL, TARGET_NAME_COL, TIMESTAMP_COL, quantile_col

HANDLER_PATH = Path(__file__).resolve().parents[1] / "deploy" / "sagemaker" / "inference.py"
HOURS = 336
START = datetime(2018, 12, 1, tzinfo=UTC)


def load_handler() -> ModuleType:
    """Import ``inference.py`` by path: it lives outside the installed package."""
    spec = importlib.util.spec_from_file_location("sagemaker_inference", HANDLER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


inference = load_handler()


class FakeForecaster:
    """Returns the last observed value at every horizon step, with a fixed-width band."""

    def predict(self, context, *, prediction_length, quantile_levels, freq, target):
        frames = []
        for item_id, group in context.groupby(ID_COL, sort=False):
            group = group.sort_values(TIMESTAMP_COL)
            last_value = float(group[target].ffill().iloc[-1])
            stamps = pd.date_range(
                group[TIMESTAMP_COL].iloc[-1] + pd.tseries.frequencies.to_offset(freq),
                periods=prediction_length,
                freq=freq,
            )
            frame = pd.DataFrame(
                {
                    ID_COL: item_id,
                    TIMESTAMP_COL: stamps,
                    TARGET_NAME_COL: target,
                    PREDICTION_COL: last_value,
                }
            )
            for level in quantile_levels:
                frame[quantile_col(level)] = last_value + (level - 0.5) * 100.0
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


def payload(
    *, n_series: int = 1, hours: int = HOURS, skip: set[int] | None = None, **overrides
) -> dict:
    skip = skip or set()
    series = [
        {
            "item_id": f"s{index}",
            "timestamps": [
                (START + timedelta(hours=step)).isoformat()
                for step in range(hours)
                if step not in skip
            ],
            "values": [
                25000 + 5000 * math.sin(2 * math.pi * step / 24)
                for step in range(hours)
                if step not in skip
            ],
        }
        for index in range(n_series)
    ]
    return {
        "series": series,
        "prediction_length": 24,
        "freq": "h",
        "quantile_levels": [0.1, 0.5, 0.9],
        **overrides,
    }


def round_trip(body: dict) -> dict:
    request = inference.input_fn(json.dumps(body))
    prediction = inference.predict_fn(request, FakeForecaster())
    text, content_type = inference.output_fn(prediction)
    assert content_type == inference.JSON_CONTENT_TYPE
    return json.loads(text)


# --- the happy path -------------------------------------------------------------------------


def test_round_trip_returns_one_forecast_per_series_in_the_service_shape():
    response = round_trip(payload(n_series=3))
    assert [f["item_id"] for f in response["forecasts"]] == ["s0", "s1", "s2"]
    first = response["forecasts"][0]
    assert len(first["timestamps"]) == 24
    assert len(first["predictions"]) == 24
    assert sorted(first["quantiles"]) == ["0.1", "0.5", "0.9"]
    assert first["timestamps"][0].endswith("Z")
    assert response["latency_ms"] >= 0.0
    assert len(response["input_hash"]) == 16


def test_the_forecast_starts_one_step_after_the_last_observation():
    response = round_trip(payload())
    expected = (START + timedelta(hours=HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert response["forecasts"][0]["timestamps"][0] == expected


def test_the_data_checklist_runs_between_the_request_and_the_model():
    request = inference.input_fn(json.dumps(payload(skip={10, 11})))
    # The two missing hours come back as NaN rows on the regular grid, not as a shorter series.
    assert len(request["context"]) == HOURS
    assert request["context"][inference.TARGET].isna().sum() == 2


def test_the_input_hash_is_stable_and_sensitive():
    body = payload()
    assert round_trip(body)["input_hash"] == round_trip(body)["input_hash"]
    nudged = payload()
    nudged["series"][0]["values"][0] += 1.0
    assert round_trip(nudged)["input_hash"] != round_trip(body)["input_hash"]


def test_older_history_beyond_max_context_is_trimmed(monkeypatch):
    monkeypatch.setattr(inference, "MAX_CONTEXT", 100)
    request = inference.input_fn(json.dumps(payload()))
    assert len(request["context"]) == 100


# --- rejections -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"series": []}, "non-empty list"),
        ({"prediction_length": 0}, "prediction_length"),
        ({"prediction_length": 2000}, "prediction_length"),
        ({"quantile_levels": [0.0, 0.5]}, r"\(0, 1\)"),
        ({"quantile_levels": [1.0]}, r"\(0, 1\)"),
    ],
)
def test_input_fn_rejects_bad_requests(overrides, match):
    with pytest.raises(ValueError, match=match):
        inference.input_fn(json.dumps(payload(**overrides)))


def test_input_fn_rejects_mismatched_timestamps_and_values():
    body = payload()
    body["series"][0]["values"].pop()
    with pytest.raises(ValueError, match="timestamps and"):
        inference.input_fn(json.dumps(body))


def test_input_fn_rejects_more_series_than_the_limit(monkeypatch):
    monkeypatch.setattr(inference, "MAX_SERIES", 2)
    with pytest.raises(ValueError, match="exceeds the limit"):
        inference.input_fn(json.dumps(payload(n_series=3)))


def test_input_fn_rejects_a_content_type_it_does_not_speak():
    with pytest.raises(ValueError, match="Unsupported content type"):
        inference.input_fn(json.dumps(payload()), "text/csv")


def test_output_fn_rejects_an_accept_type_it_cannot_produce():
    request = inference.input_fn(json.dumps(payload()))
    prediction = inference.predict_fn(request, FakeForecaster())
    with pytest.raises(ValueError, match="Unsupported accept type"):
        inference.output_fn(prediction, "text/csv")
    assert inference.output_fn(prediction, "*/*")[1] == inference.JSON_CONTENT_TYPE


# --- the model loader -----------------------------------------------------------------------


def test_model_fn_prefers_a_packaged_checkpoint_over_the_hub(monkeypatch, tmp_path):
    (tmp_path / "chronos-2").mkdir()
    seen = {}

    def fake_load(model_id, *, device):
        seen["model_id"] = model_id
        seen["device"] = device
        return object()

    monkeypatch.setattr(inference, "load_chronos2", fake_load)
    monkeypatch.setattr(inference, "Chronos2Forecaster", lambda pipeline: pipeline)
    inference.model_fn(str(tmp_path))
    assert seen["model_id"] == str(tmp_path / "chronos-2")
    assert seen["device"] == "cpu"


def test_model_fn_falls_back_to_the_hub_id_when_nothing_was_packaged(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        inference, "load_chronos2", lambda model_id, *, device: seen.update(model_id=model_id)
    )
    monkeypatch.setattr(inference, "Chronos2Forecaster", lambda pipeline: pipeline)
    inference.model_fn(str(tmp_path))
    assert seen["model_id"] == "amazon/chronos-2"
