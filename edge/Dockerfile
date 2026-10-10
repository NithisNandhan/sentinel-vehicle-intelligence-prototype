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
RUN --mount=type=cache,target=/models/cache/pip pip install 'lap>=0.5.12,<1'
COPY edge/sentinel_edge ./sentinel_edge
CMD ["python", "-m", "sentinel_edge", "live"]

# CPU-only engineering runtime for recorded footage. CPU Torch is installed
# first so ultralytics does not pull the multi-gigabyte CUDA build.
FROM base AS live-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip pip install \
    --index-url https://download.pytorch.org/whl/cpu 'torch>=2.2,<3' 'torchvision>=0.17,<1'
RUN --mount=type=cache,target=/root/.cache/pip pip install -e '.[live-cpu]' 'lap>=0.5.12,<1'
COPY edge/sentinel_edge ./sentinel_edge
ENV SENTINEL_OCR_DEVICE=cpu
CMD ["python", "-m", "sentinel_edge.live_detection_viewer", "--help"]

