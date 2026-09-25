"""Environment configuration; secrets are never logged or given defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    central_api_url: str
    edge_id: str
    edge_api_key: str = field(repr=False)
    hmac_key: str = field(repr=False)
    db_path: Path
    model_cache_dir: Path
    sentinel_base_url: str
    sentinel_email: str = field(repr=False)
    sentinel_password: str = field(repr=False)
    sentinel_cookie: str = field(repr=False)
    camera_ids: tuple[str, ...]
    inference_fps: float
    ocr_min_confidence: float
    heartbeat_seconds: int
    simulator_camera_ids: tuple[str, ...]
    simulator_plate: str
    media_relay_base_url: str
    mediamtx_publish_url: str
    preview_base_url: str

    @classmethod
    def from_env(cls, *, live: bool) -> "Config":
        def csv(name: str) -> tuple[str, ...]:
            return tuple(x.strip() for x in os.getenv(name, "").split(",") if x.strip())

        cfg = cls(
            central_api_url=os.getenv("CENTRAL_API_URL", "http://localhost:8000").rstrip("/"),
            edge_id=os.getenv("EDGE_ID", "region-demo-01"),
            edge_api_key=os.getenv("EDGE_API_KEY", ""),
            hmac_key=os.getenv("WATCHLIST_HMAC_KEY", ""),
            db_path=Path(os.getenv("EDGE_DB_PATH", "./data/outbox.sqlite3")),
            model_cache_dir=Path(os.getenv("MODEL_CACHE_DIR", "./models")),
            sentinel_base_url=os.getenv("SENTINEL_BASE_URL", "https://cctv.corp8.cloud").rstrip("/"),
            sentinel_email=os.getenv("SENTINEL_EMAIL", ""),
            sentinel_password=os.getenv("SENTINEL_PASSWORD", ""),
            sentinel_cookie=os.getenv("SENTINEL_COOKIE", ""),
            camera_ids=csv("CAMERA_IDS"),
            inference_fps=float(os.getenv("INFERENCE_FPS", "5")),
            ocr_min_confidence=float(os.getenv("OCR_MIN_CONFIDENCE", "0.85")),
            heartbeat_seconds=int(os.getenv("HEARTBEAT_SECONDS", "30")),
            simulator_camera_ids=csv("SIMULATOR_CAMERA_IDS") or ("cam01", "cam04"),
            simulator_plate=os.getenv("SIMULATOR_PLATE", "GJ01AB1234"),
            media_relay_base_url=os.getenv("MEDIA_RELAY_BASE_URL", "").rstrip("/"),
            mediamtx_publish_url=os.getenv("MEDIAMTX_PUBLISH_URL", "rtsp://mediamtx:8554").rstrip("/"),
            preview_base_url=os.getenv("PREVIEW_BASE_URL", "http://localhost:8888").rstrip("/"),
        )
        missing = [name for name, value in (("EDGE_API_KEY", cfg.edge_api_key), ("WATCHLIST_HMAC_KEY", cfg.hmac_key)) if not value]
        if live:
            missing += [name for name, value in (("SENTINEL_EMAIL", cfg.sentinel_email), ("SENTINEL_PASSWORD", cfg.sentinel_password)) if not value]
        if missing:
            raise ValueError("Missing required environment variables: " + ", ".join(missing))
        if len(cfg.camera_ids) > 2 or len(cfg.simulator_camera_ids) != 2:
            raise ValueError("Configure at most two live cameras and exactly two simulator cameras")
        if not 0 < cfg.inference_fps <= 30 or not 0 < cfg.ocr_min_confidence <= 1 or cfg.heartbeat_seconds < 1:
            raise ValueError("Invalid inference FPS, OCR confidence or heartbeat interval")
        return cfg
