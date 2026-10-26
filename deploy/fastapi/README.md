# Chronos-2 forecasting service (FastAPI + Docker, CPU)

A small HTTP service that serves probabilistic forecasts from Chronos-2 on CPU, used in article 5
of the series. Every request goes through the same data checklist as the notebooks
(`c2p.data.validate_and_prepare`): timestamps are converted to UTC, missing timestamps become
missing values on a regular grid, and only the most recent 8,192 steps of each series are kept.
Each response carries a SHA-256 hash of that canonical input, so a monitoring log can tie every
forecast to exactly what the model saw.

| File | Purpose |
| --- | --- |
| `app.py` | The FastAPI app: `GET /health`, `POST /forecast` |
| `Dockerfile` | Multi-stage CPU image built with uv from the project lockfile |
| `Dockerfile.dockerignore` | Allowlist for the build context (only what the image needs) |
| `smoke_test.py` | End-to-end check and latency benchmark against a running service |
| `../../tests/test_api.py` | Unit tests with a fake model (fast) and the real model (`-m slow`) |

All commands run from the repository root.

## Run without Docker

```bash
uv sync --group deploy                      # adds fastapi, uvicorn and httpx
uv run --group deploy uvicorn app:app --app-dir deploy/fastapi --port 8000
```

The first start downloads `amazon/chronos-2` (about 480 MB) into the Hugging Face cache.

## Build and run the Docker image

```bash
docker build -f deploy/fastapi/Dockerfile -t chronos2-api:spike .
docker run -d -p 8000:8000 --name c2p-spike chronos2-api:spike
docker ps --filter name=c2p-spike           # STATUS shows (healthy) once the model is loaded
docker logs -f c2p-spike                    # one log line per forecast, with its input hash
```

The image installs CPU-only PyTorch from `uv.lock` (`uv sync --frozen --extra cpu`), runs as a
non-root user and contains the model weights, downloaded at build time, so the container starts
offline (`HF_HUB_OFFLINE=1`). Build arguments: `MODEL_ID` (Hub model baked into the image,
default `amazon/chronos-2`), `PYTHON_IMAGE` and `UV_IMAGE` (base images).

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_ID` | `amazon/chronos-2` | Hub id or local folder of a Chronos-2 model |
| `DEVICE` | `cpu` | `cpu` or `cuda` (the image ships CPU-only PyTorch) |
| `MAX_CONTEXT` | `8192` | Most recent steps per series fed to the model (1 to 8192) |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `TORCH_NUM_THREADS` | all cores available | PyTorch CPU threads, set explicitly at start-up (a container can otherwise leave OpenMP at one thread); lower it when several containers share a host |

To serve a fine-tuned model, mount its folder read-only and point `MODEL_ID` at it. LoRA adapters
load on top of `amazon/chronos-2`, which is already in the image:

```bash
docker run -d -p 8000:8000 --name c2p-spike \
  -v "$PWD/models/chronos-2-lora:/models/chronos-2-lora:ro" \
  -e MODEL_ID=/models/chronos-2-lora chronos2-api:spike
```

## Test

```bash
uv run --group deploy python deploy/fastapi/smoke_test.py --wait 300            # end to end
uv run --group deploy python deploy/fastapi/smoke_test.py --bench 1 100        # p50/p95 latency
uv run --group deploy pytest tests/test_api.py                                  # fake model
uv run --group deploy pytest -m slow tests/test_api.py                          # real model, CPU
```

The smoke test sends one week of hourly synthetic data with six missing hours and one `null`,
checks the response and prints the forecast quantiles and the latency.

## API

`GET /health` answers once the model is loaded (503 before):

```json
{"status": "ok", "model_id": "amazon/chronos-2", "device": "cpu", "chronos_version": "2.3.2"}
```

`POST /forecast` takes one or more series. Timestamps may carry UTC offsets or be naive (read as
UTC); `null` marks a missing value and missing timestamps are allowed. `quantile_levels`
defaults to `[0.1, 0.5, 0.9]`.

```json
{
  "series": [
    {
      "item_id": "ES",
      "timestamps": ["2018-12-31T20:00:00+01:00", "2018-12-31T21:00:00+01:00", "2018-12-31T23:00:00+01:00"],
      "values": [27510.0, 25934.0, null]
    }
  ],
  "prediction_length": 2,
  "freq": "h",
  "quantile_levels": [0.1, 0.5, 0.9]
}
```

The response lists the forecasts in request order, with UTC timestamps, the point forecast
(`predictions`, the median) and one list per quantile level, plus what the data checklist did:

```json
{
  "model_id": "amazon/chronos-2",
  "input_hash": "3f1c...e9a0",
  "latency_ms": 61.4,
  "freq": "h",
  "prediction_length": 2,
  "quantile_levels": [0.1, 0.5, 0.9],
  "input_summary": {"n_series": 1, "n_points": 4, "n_missing_timestamps": 1,
                    "n_missing_values": 2, "n_trimmed": 0, "notes": []},
  "forecasts": [
    {
      "item_id": "ES",
      "timestamps": ["2018-12-31T23:00:00Z", "2019-01-01T00:00:00Z"],
      "predictions": [24310.2, 23655.8],
      "quantiles": {"0.1": [23011.5, 22140.9], "0.5": [24310.2, 23655.8], "0.9": [25502.7, 25120.3]}
    }
  ]
}
```

(Numbers are illustrative.) `latency_ms` is measured inside the server, from the parsed request to
the finished forecast. Interactive documentation is served at `http://localhost:8000/docs`.

### Validation and limits

Bad input gets a 422 with the type, location and message of each problem (the input is not
echoed back): mismatched lengths, all-`null` series, duplicated item ids or timestamps,
timestamps off the `freq` grid (e.g. 10:30 among hourly rows), mixed naive and offset
timestamps, `NaN`/`Infinity` values, quantile levels outside (0, 1), unknown fields, and these
limits: at most 1,000 series per request, a horizon of 1 to 1,024 steps, at most 32,768 grid
steps per series and 2,000,000 per request (gaps included). Older steps beyond `MAX_CONTEXT` are
trimmed, and the response reports how many (`n_trimmed`). Limit the request body size in the
reverse proxy in front of the service as well.

### Input hash

`input_hash` is the SHA-256 of the canonical model input, not of the request bytes: a JSON header
(format version, `freq`, `prediction_length`, sorted `quantile_levels`, id and length of every
series in id order), then all timestamps as little-endian int64 microseconds since the Unix epoch
(UTC) and all values as little-endian float64 with NaN for missing values. Requests that differ
only in formatting, series or quantile order, UTC offsets, `null` versus an absent timestamp, or
history older than the context window get the same hash. The model id is not part of the hash;
it is logged next to it.

## Clean up

```bash
docker rm -f c2p-spike                     # stop and remove the container
docker image rm chronos2-api:spike         # remove the image
docker builder prune                       # optional: drop the build cache (uv and layers)
```

The image is local only: nothing here pushes it to a registry or creates cloud resources.
