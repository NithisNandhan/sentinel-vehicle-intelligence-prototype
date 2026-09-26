"""CLI entrypoints for live worker, local stream relay and offline simulator."""

from __future__ import annotations

import argparse
import logging
import os
import threading
from pathlib import Path

from .cameras import discover_live_pair
from .config import Config


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m sentinel_edge")
    parser.add_argument("mode", choices=("live", "relay", "gateway", "simulate", "prepare-models", "benchmark-models", "benchmark-live", "benchmark-degradation", "probe-stream", "probe-two-streams"))
    parser.add_argument("--once", action="store_true", help="Send offline events once and exit")
    parser.add_argument("--probe-camera-id", default="probe-bus", help="Local MediaMTX path used only for probe-stream")
    parser.add_argument("--probe-seconds", type=int, default=20, help="Observation duration for stream probes or live benchmark")
    parser.add_argument("--manifest", type=Path, help="Licensed, labelled benchmark manifest (never Sentinel recordings)")
    parser.add_argument("--report", type=Path, help="Optional aggregate-only generated benchmark report")
    parser.add_argument("--degradation-family", choices=("resolution", "blur", "motion", "noise", "darkness"), default="resolution")
    parser.add_argument("--benchmark-stage", choices=("both", "end_to_end", "oracle_plate_ocr"), default="both")
    parser.add_argument("--ocr-spacing-ms", type=int, default=300)
    parser.add_argument("--ocr-backend", choices=("cct", "parseq"), default="cct",
                        help="Degradation benchmark only; PARSeq needs isolated research dependencies")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = Config.from_env(live=args.mode in {"live", "relay", "gateway", "benchmark-live"})
    config.model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(config.model_cache_dir / "cache"))
    os.environ.setdefault("HF_HOME", str(config.model_cache_dir / "huggingface"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config.model_cache_dir / "ultralytics"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    try:
        if args.mode == "simulate":
            from .simulator import run_simulator
            run_simulator(config, stop, once=args.once)
        elif args.mode == "prepare-models":
            from .vision import VisionPipeline
            os.chdir(config.model_cache_dir)
            from ultralytics import YOLO
            YOLO("yolov8n.pt")
            VisionPipeline(config.model_cache_dir / "yolov8n.pt")
        elif args.mode == "benchmark-models":
            from .benchmark import run_model_benchmark
            run_model_benchmark(config.model_cache_dir / "yolov8n.pt")
        elif args.mode == "benchmark-degradation":
            if args.manifest is None or not 150 <= args.ocr_spacing_ms <= 1000:
                parser.error("benchmark-degradation needs --manifest and temporal spacing between 150 and 1000 ms")
            from .degradation_benchmark import run_degradation_benchmark
            stages = None if args.benchmark_stage == "both" else [args.benchmark_stage]
            run_degradation_benchmark(config, args.manifest, family=args.degradation_family, stages=stages,
                                      spacing_ms=args.ocr_spacing_ms, output_path=args.report, ocr_backend=args.ocr_backend)
        elif args.mode in {"probe-stream", "probe-two-streams"}:
            if not 1 <= args.probe_seconds <= 300:
                parser.error("--probe-seconds must be between 1 and 300")
            if args.mode == "probe-stream":
                from .benchmark import run_stream_probe
                run_stream_probe(config, camera_id=args.probe_camera_id, seconds=args.probe_seconds)
            else:
                from .benchmark import run_two_stream_probe
                run_two_stream_probe(config, camera_ids=(args.probe_camera_id,
                                                         args.probe_camera_id + "-2"),
                                     seconds=args.probe_seconds)
        elif args.mode == "benchmark-live":
            if not 1 <= args.probe_seconds <= 900:
                parser.error("--probe-seconds must be between 1 and 900")
            from .live_benchmark import run_live_benchmark
            cameras, _ = discover_live_pair(config)
            run_live_benchmark(config, cameras, seconds=args.probe_seconds)
        elif args.mode == "gateway":
            from .catalogue_gateway import run_gateway
            run_gateway(config)
        else:
            cameras, used_fallback = discover_live_pair(config)
            if used_fallback:
                logging.warning("Camera catalogue unavailable; using documented CAM01/CAM02 paths only")
            from .api import CentralClient
            from .outbox import Outbox
            try:
                CentralClient(config.central_api_url, config.edge_api_key, Outbox(config.db_path)).sync_cameras([
                    {"id": camera.camera_id, "name": camera.name, "source_mode": "live",
                     "status": "UNSCHEDULED", "preview_url":
                     f"{config.preview_base_url}/{camera.camera_id}/index.m3u8"}
                    for camera in cameras
                ])
            except OSError:
                logging.warning("Camera registry sync deferred; central API is unavailable")
            if args.mode == "relay":
                from .relay import run_relays
                run_relays(config, cameras, stop)
            else:
                from .worker import run_live
                run_live(config, cameras, stop)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
