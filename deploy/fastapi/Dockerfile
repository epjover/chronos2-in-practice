# syntax=docker/dockerfile:1
#
# CPU-only image of the Chronos-2 forecasting service (deploy/fastapi/app.py).
# Build from the repository root (the build context must contain pyproject.toml and uv.lock):
#
#   docker build -f deploy/fastapi/Dockerfile -t chronos2-api:spike .
#   docker run -d -p 8000:8000 --name c2p-spike chronos2-api:spike
#
# The model weights are downloaded at build time, so the container starts offline.
# deploy/fastapi/Dockerfile.dockerignore keeps the build context to the files listed there.

ARG PYTHON_IMAGE=python:3.11-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.18
ARG MODEL_ID=amazon/chronos-2

FROM ${UV_IMAGE} AS uv

# ---- 1. Third-party dependencies from the lockfile (CPU PyTorch), without the project ----------
FROM ${PYTHON_IMAGE} AS deps
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /src
# --no-default-groups: the project's default groups (dev, notebooks) are not needed to serve.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --no-default-groups --group deploy --extra cpu

# ---- 2. Model weights (rebuilt only when MODEL_ID or the dependencies change) ------------------
FROM deps AS weights
ARG MODEL_ID
ENV HF_HOME=/opt/hf
RUN /opt/venv/bin/python -c "import sys; from huggingface_hub import snapshot_download; \
snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors'])" "${MODEL_ID}"

# ---- 3. The c2p package, installed into the same environment (not editable) --------------------
FROM deps AS builder
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group deploy --extra cpu --no-editable

# ---- 4. Runtime: Python, the environment, the weights and app.py; no uv, runs as non-root ------
FROM ${PYTHON_IMAGE} AS runtime
ARG MODEL_ID
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app
ENV PATH=/opt/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODEL_ID=${MODEL_ID} \
    DEVICE=cpu
# Weights first: this layer stays cached when only the code changes.
COPY --from=weights /opt/hf /opt/hf
COPY --from=builder /opt/venv /opt/venv
COPY deploy/fastapi/app.py /app/app.py
WORKDIR /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
