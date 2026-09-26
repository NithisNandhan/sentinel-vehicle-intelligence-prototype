"""Same-frame live comparison; only counts/timings leave memory, never media/text."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .benchmark import latency_summary
from .stream import read_camera
from .vision import VisionPipeline


def _camera_probe(config, camera, pipelines, seconds):
    stop = threading.Event()
    # Bound startup separately; measure the requested interval from first frame.
    timer = threading.Timer(seconds + 60, stop.set)
    probe_started = time.monotonic()
    last_sample_pts = None
    last_sample_wall = None
    decoded = sampled = discontinuities = 0
    latencies = {name: [] for name in pipelines}
    accepted = {name: set() for name in pipelines}
    errors = {name: 0 for name in pipelines}
    first_at = last_at = None
    timer.start()
    try:
        for _, frame in read_camera(config, camera, stop=stop):
            if stop.is_set():
                break
            now = time.monotonic()
            if first_at is None:
                first_at = now
                timer.cancel()
                timer = threading.Timer(seconds, stop.set)
                timer.start()
            last_at = now
            decoded += 1
            if frame.discontinuity:
                discontinuities += 1
                last_sample_pts = last_sample_wall = None
                for pipeline in pipelines.values():
                    pipeline.reset()
            due = (last_sample_pts is None or frame.pts_ms - last_sample_pts >= 1000 / config.inference_fps
                   if frame.pts_ms is not None else
                   last_sample_wall is None or now - last_sample_wall >= 1 / config.inference_fps)
            if not due:
                continue
            last_sample_pts, last_sample_wall = frame.pts_ms, now
            sampled += 1
            # Rotate order to reduce systematic model-order timing bias.
            order = list(pipelines.items())
            if sampled % 2:
                order.reverse()
            for name, pipeline in order:
                started = time.perf_counter()
                try:
                    observations = pipeline.process(frame.image, pts_ms=frame.pts_ms)
                    for observation in observations:
                        accepted[name].add(hashlib.sha256(observation.plate.encode()).digest())
                except Exception:
                    errors[name] += 1  # Never print third-party messages or OCR text.
                latencies[name].append((time.perf_counter() - started) * 1000)
    finally:
        stop.set()
        timer.cancel()
    methods = {}
    for name, pipeline in pipelines.items():
        methods[name] = {
            "detector_model": pipeline.detector_model_name,
            "ocr_model": pipeline.ocr_model_name,
            "plate_detector_providers": pipeline.alpr.detector.detector.model.get_providers(),
            "ocr_providers": pipeline.alpr.ocr.ocr_model.model.get_providers(),
            "unique_accepted_plate_identifiers": len(accepted[name]),
            "processing_errors": errors[name],
            "metrics": pipeline.metrics,
            "vision_latency": latency_summary(latencies[name]) if latencies[name] else None,
        }
    return {"camera_id": camera.camera_id, "decoded_frames": decoded, "sampled_frames": sampled,
            "startup_seconds": round(first_at - probe_started, 2) if first_at is not None else None,
            "observed_wall_seconds": round(last_at - first_at, 2) if first_at is not None else 0,
            "discontinuities": discontinuities, "methods": methods}


def check_probability_adapter(pipeline):
    """Real neural-model parity on an in-memory synthetic image, not accuracy."""
    import numpy as np
    cv2 = pipeline.cv2
    image = np.full((64, 320, 3), 255, dtype=np.uint8)
    cv2.putText(image, "GJ01AB1234", (8, 48), cv2.FONT_HERSHEY_SIMPLEX,
                1.25, (0, 0, 0), 2, cv2.LINE_AA)
    matrix = pipeline._probabilities(image)
    recognizer = pipeline.alpr.ocr.ocr_model
    cfg = recognizer.config
    public_image = image
    if cfg.image_color_mode == "grayscale":
        public_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif cfg.image_color_mode == "rgb":
        public_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = recognizer.run_one(public_image, return_confidence=True, remove_pad_char=False)
    greedy = "".join(cfg.alphabet[max(range(len(row)), key=row.__getitem__)] for row in matrix)
    score_delta = max(abs(max(row) - float(score)) for row, score in zip(matrix, result.char_probs))
    row_sum_error = max(abs(sum(row) - 1.0) for row in matrix)
    if greedy != result.plate or len(result.char_probs) != len(matrix) or score_delta > 1e-5 or row_sum_error > 1e-4:
        raise RuntimeError("Full-probability adapter failed public OCR parity")
    return {"synthetic_fixture_only": True, "public_api_parity": True,
            "slots": len(matrix), "maximum_score_difference": score_delta,
            "maximum_probability_row_sum_error": row_sum_error}


def run_live_benchmark(config, cameras, *, seconds=180):
    import av
    import cv2
    import ultralytics
    from pathlib import Path

    av.logging.set_level(av.logging.PANIC)
    # Initialize/warm models serially; each camera owns separate tracker state.
    prepared = []
    adapter_check = None
    fixture = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    for camera in cameras:
        pipelines = {
            "legacy": VisionPipeline(config.model_cache_dir / "yolov8n.pt", multi_frame=False,
                                     ocr_min_confidence=config.ocr_min_confidence),
            "upgraded_models_consensus": VisionPipeline(config.model_cache_dir / "yolov8n.pt", multi_frame=False,
                                     ocr_min_confidence=config.ocr_min_confidence,
                                     detector_model="yolo-v9-t-512-license-plate-end2end",
                                     ocr_model="cct-s-v2-global-model"),
            "temporal": VisionPipeline(config.model_cache_dir / "yolov8n.pt",
                                       ocr_min_confidence=config.ocr_min_confidence),
        }
        for pipeline in pipelines.values():
            if fixture is not None:
                pipeline.process(fixture, pts_ms=0)
            pipeline.reset()
            pipeline.metrics.clear()
        if adapter_check is None:
            adapter_check = check_probability_adapter(pipelines["temporal"])
        prepared.append((camera, pipelines))
    started = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_camera_probe, config, camera, pipelines, seconds)
                   for camera, pipelines in prepared]
        results = [future.result() for future in futures]
    report = {"source": "live Sentinel; same sampled frames; no recordings or central writes",
              "started_at_utc": started, "requested_seconds": seconds,
              "ground_truth_available": False, "score_is_calibrated": False,
              "accuracy": None, "probability_adapter_check": adapter_check, "cameras": results}
    print(json.dumps(report, indent=2))
    if any(not camera["sampled_frames"] for camera in results):
        raise RuntimeError("Live benchmark did not receive frames on both cameras")
    if any(method["processing_errors"] for camera in results for method in camera["methods"].values()):
        raise RuntimeError("Live benchmark recorded processing errors")
    return report
