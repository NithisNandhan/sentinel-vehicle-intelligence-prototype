"""CLI entrypoints for live worker, local stream relay and offline simulator."""

from __future__ import annotations

import argparse
import logging
import os
import threading

from .cameras import discover, select_two
from .config import Config


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m sentinel_edge")
    parser.add_argument("mode", choices=("live", "relay", "simulate", "prepare-models"))
    parser.add_argument("--once", action="store_true", help="Send offline events once and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = Config.from_env(live=args.mode in {"live", "relay"})
    config.model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(config.model_cache_dir / "cache"))
    os.environ.setdefault("HF_HOME", str(config.model_cache_dir / "huggingface"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config.model_cache_dir / "ultralytics"))
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
        else:
            catalogue = discover(config)
            cameras = select_two(catalogue, config.camera_ids)
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
