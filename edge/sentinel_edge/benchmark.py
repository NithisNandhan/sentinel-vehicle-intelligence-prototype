"""Repeatable model-throughput smoke test using a bundled, non-Sentinel image."""

from __future__ import annotations

import json
import math
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable


def latency_summary(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("At least one latency sample is required")
    ordered = sorted(samples_ms)
    return {"median_ms": round(statistics.median(ordered), 2),
            "p95_ms": round(ordered[math.ceil(len(ordered) * 0.95) - 1], 2)}


def run_model_benchmark(model_path: Path, *, warmup: int = 3, repeats: int = 20) -> dict:
    """Measure model calls, never save images or print OCR text."""
    import cv2
    import torch
    import ultralytics

    from .vision import VisionPipeline

    fixture = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    image = cv2.imread(str(fixture))
    if image is None:
        raise RuntimeError("Bundled traffic fixture is unavailable")
    pipeline = VisionPipeline(model_path)

    def synchronise() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def measure(operation: Callable[[], object]) -> tuple[object, dict[str, float]]:
        for _ in range(warmup):
            operation()
            synchronise()
        samples: list[float] = []
        result: object = None
        for _ in range(repeats):
            synchronise()
            start = time.perf_counter()
            result = operation()
            synchronise()
            samples.append((time.perf_counter() - start) * 1000)
        return result, latency_summary(samples)

    def track() -> object:
        return pipeline.model.track(image, persist=True, tracker="bytetrack.yaml",
                                    classes=pipeline.VEHICLE_CLASSES, verbose=False, conf=0.25)[0]

    detection, yolo_ms = measure(track)
    plates, alpr_ms = measure(lambda: pipeline.alpr.predict(image))
    detector_session = pipeline.alpr.detector.detector.model
    ocr_session = pipeline.alpr.ocr.ocr_model.model
    predictor = getattr(pipeline.model, "predictor", None)
    report = {
        "fixture": "ultralytics/assets/bus.jpg (not Sentinel)",
        "warmup": warmup,
        "repeats": repeats,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "vehicle_boxes_in_last_run": len(detection.boxes),
        "plate_detections_in_last_run": len(plates),
        "yolo_tracking": yolo_ms,
        "fastalpr_full_image": alpr_ms,
        "yolo_device": str(getattr(predictor, "device", "unknown")),
        "plate_detector_providers": detector_session.get_providers(),
        "ocr_providers": ocr_session.get_providers(),
    }
    print(json.dumps(report, indent=2))
    return report


def run_stream_probe(config, *, camera_id: str = "probe-bus", seconds: int = 20,
                     emit: bool = True) -> dict:
    """Exercise local RTSP decode and live vision without central writes or recordings."""
    import cv2
    import ultralytics

    from .cameras import Camera
    from .stream import read_camera
    from .vision import VisionPipeline

    if not config.media_relay_base_url:
        raise ValueError("A local MEDIA_RELAY_BASE_URL is required for the stream probe")
    stop = threading.Event()
    timer = threading.Timer(seconds, stop.set)
    vision = VisionPipeline(config.model_cache_dir / "yolov8n.pt")
    fixture = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    if fixture is None:
        raise RuntimeError("Bundled traffic fixture is unavailable")
    warmup_start = time.perf_counter()
    vision.model.track(fixture, persist=True, tracker="bytetrack.yaml",
                       classes=vision.VEHICLE_CLASSES, verbose=False, conf=0.25)
    vision.alpr.predict(fixture)
    warmup_ms = round((time.perf_counter() - warmup_start) * 1000, 2)
    vision.reset()
    decoded = 0
    inferred = 0
    sightings = 0
    last_inference_pts: int | None = None
    last_inference_wall: float | None = None
    pts: list[int] = []
    latencies: list[float] = []
    stream_started_at = time.monotonic()
    first_frame_at: float | None = None
    timer.start()
    try:
        for _, frame in read_camera(config, Camera(camera_id, "Local traffic fixture"), stop=stop):
            if stop.is_set():
                break
            if first_frame_at is None:
                first_frame_at = time.monotonic()
            decoded += 1
            if frame.pts_ms is not None:
                pts.append(frame.pts_ms)
            now = time.monotonic()
            if frame.discontinuity:
                vision.reset()
                last_inference_pts = None
                last_inference_wall = None
            due = ((now - last_inference_wall >= 1 / config.inference_fps)
                   if frame.pts_ms is None and last_inference_wall is not None else
                   (last_inference_pts is None or frame.pts_ms is None or
                    frame.pts_ms - last_inference_pts >= 1000 / config.inference_fps))
            if not due:
                continue
            last_inference_pts = frame.pts_ms
            last_inference_wall = now
            started = time.perf_counter()
            sightings += len(vision.process(frame.image))
            latencies.append((time.perf_counter() - started) * 1000)
            inferred += 1
    finally:
        stop.set()
        timer.cancel()
    report = {
        "source": "local RTSP fixture (not Sentinel)",
        "camera_id": camera_id,
        "duration_seconds": seconds,
        "model_warmup_ms": warmup_ms,
        "stream_startup_ms": round((first_frame_at - stream_started_at) * 1000, 2) if first_frame_at else None,
        "decoded_frames": decoded,
        "inference_frames": inferred,
        "confirmed_plate_sightings": sightings,
        "first_pts_ms": pts[0] if pts else None,
        "last_pts_ms": pts[-1] if pts else None,
        "vision_process": latency_summary(latencies) if latencies else None,
    }
    if emit:
        print(json.dumps(report, indent=2))
    if not decoded or not inferred:
        raise RuntimeError("Local RTSP fixture did not produce inference frames")
    return report


def run_two_stream_probe(config, *, camera_ids: tuple[str, str], seconds: int = 20) -> dict:
    """Match the live worker's two independent vision pipelines on local feeds."""
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="probe-camera") as pool:
        futures = [pool.submit(run_stream_probe, config, camera_id=camera_id,
                               seconds=seconds, emit=False) for camera_id in camera_ids]
        cameras = [future.result() for future in futures]
    report = {"source": "two local RTSP fixtures (not Sentinel)", "cameras": cameras}
    print(json.dumps(report, indent=2))
    return report
