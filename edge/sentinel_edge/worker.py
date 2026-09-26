"""Concurrent camera workers, health reporting and outbox submission."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .api import CentralClient
from .cameras import Camera
from .config import Config
from .events import health, sighting, utc_now
from .outbox import Outbox
from .stream import read_camera
from .vision import VisionPipeline

log = logging.getLogger(__name__)


@dataclass
class _Stats:
    decoded: int = 0
    inferred: int = 0
    reconnects: int = 0
    last_frame_at: float = 0.0
    degraded: bool = False


def run_camera(config: Config, camera: Camera, outbox: Outbox, stop: threading.Event, hybrid_resources=None) -> None:
    camera_id = camera.camera_id
    vision = VisionPipeline(config.model_cache_dir / "yolov8n.pt",
                            ocr_min_confidence=config.ocr_min_confidence, hybrid_resources=hybrid_resources)
    stats = _Stats()
    lock = threading.Lock()
    heartbeat_stop = threading.Event()

    def heartbeat_loop() -> None:
        while not heartbeat_stop.wait(config.heartbeat_seconds):
            with lock:
                status = "DEGRADED" if stats.degraded or time.monotonic() - stats.last_frame_at > config.heartbeat_seconds * 2 else "ONLINE"
                event = health(edge_id=config.edge_id, camera_id=camera_id, status=status,
                               decoded_frames=stats.decoded, inference_frames=stats.inferred,
                               reconnects=stats.reconnects, outbox_depth=outbox.depth(),
                               source_mode="live")
                stats.decoded = 0
                stats.inferred = 0
                stats.reconnects = 0
            outbox.put("/v1/health", event)

    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True,
                                        name=f"health-{camera_id}")
    heartbeat_thread.start()
    last_pts: int | None = None
    last_inference_pts: int | None = None
    last_inference_wall: float | None = None
    try:
        for _, frame in read_camera(config, camera, stop=stop):
            if stop.is_set():
                break
            now = time.monotonic()
            with lock:
                if stats.last_frame_at and now - stats.last_frame_at > config.heartbeat_seconds * 2:
                    stats.reconnects += 1
                stats.decoded += 1
                stats.last_frame_at = now
                stats.degraded = False
                if frame.discontinuity:
                    stats.reconnects += 1
            if frame.discontinuity or (last_pts is not None and frame.pts_ms is not None and frame.pts_ms < last_pts):
                vision.reset()
                last_inference_pts = None
                last_inference_wall = None
            last_pts = frame.pts_ms
            due = ((now - last_inference_wall >= 1 / config.inference_fps)
                   if frame.pts_ms is None and last_inference_wall is not None else
                   (last_inference_pts is None or frame.pts_ms is None or
                    frame.pts_ms - last_inference_pts >= 1000 / config.inference_fps))
            if due:
                with lock:
                    stats.inferred += 1
                last_inference_pts = frame.pts_ms
                last_inference_wall = now
                try:
                    observations = vision.process(frame.image, pts_ms=frame.pts_ms)
                except Exception:
                    log.exception("AI processing failed for camera %s", camera_id)
                    observations = []
                    with lock:
                        stats.degraded = True
                for observation in observations:
                    event = sighting(edge_id=config.edge_id, camera_id=camera_id,
                                     plate=observation.plate, hmac_key=config.hmac_key,
                                     confidence=observation.confidence, source_mode="live",
                                     stream_pts_ms=frame.pts_ms if frame.pts_ms is not None and frame.pts_ms >= 0 else None,
                                     track_id=observation.track_id,
                                     observed_at=utc_now(), model_version=vision.model_version)
                    outbox.put("/v1/sightings", event)
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=5)


def run_live(config: Config, cameras: list[Camera], stop: threading.Event) -> None:
    if not (config.model_cache_dir / "yolov8n.pt").is_file():
        raise FileNotFoundError("Missing yolov8n.pt in MODEL_CACHE_DIR; run prepare-models first")
    outbox = Outbox(config.db_path)
    client = CentralClient(config.central_api_url, config.edge_api_key, outbox)
    hybrid_resources = None
    if config.plate_reader == 'hybrid-v1':
        from .hybrid import HybridResources
        hybrid_resources = HybridResources(config.model_cache_dir / 'experiments',
                                           alignment=config.hybrid_alignment,
                                           restoration=config.hybrid_restoration)
    delivery = client.start_retry_loop(stop)
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="camera")
    try:
        futures = [pool.submit(run_camera, config, camera, outbox, stop, hybrid_resources) for camera in cameras]
        for future in as_completed(futures):
            future.result()
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        delivery.join(timeout=5)
