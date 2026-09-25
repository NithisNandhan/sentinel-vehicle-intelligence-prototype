FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/models YOLO_CONFIG_DIR=/models/ultralytics \
    HF_HOME=/models/huggingface XDG_CACHE_HOME=/models/cache
WORKDIR /app
COPY edge/pyproject.toml ./
COPY edge/sentinel_edge/__init__.py ./sentinel_edge/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip pip install -e .
RUN mkdir -p /models /data
VOLUME ["/models", "/data"]

FROM base AS offline
COPY edge/sentinel_edge ./sentinel_edge
CMD ["python", "-m", "sentinel_edge", "simulate"]

FROM base AS live-deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/models/cache/pip pip install -e '.[live]'

FROM live-deps AS live
COPY edge/sentinel_edge ./sentinel_edge
CMD ["python", "-m", "sentinel_edge", "live"]

