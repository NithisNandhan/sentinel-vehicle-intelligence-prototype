"""Deterministic offline metadata producer; no real media is used."""

from __future__ import annotations

import threading
from uuid import uuid5, NAMESPACE_URL

from .api import CentralClient
from .config import Config
from .events import health, sighting
from .outbox import Outbox


def seed_events(config: Config, outbox: Outbox) -> None:
    first, second = config.simulator_camera_ids
    route = [
        (first, "GJ05CD4321", 0.96, "normal"),
        (first, config.simulator_plate, 0.97, "watchlist-first"),
        (second, config.simulator_plate, 0.95, "watchlist-second"),
    ]
    for camera_id, plate, confidence, label in route:
        event = sighting(edge_id=config.edge_id, camera_id=camera_id, plate=plate,
                         hmac_key=config.hmac_key, confidence=confidence,
                         source_mode="synthetic", model_version="offline-simulator-1")
        # A restart replays the same IDs. The central API remains idempotent.
        event["event_id"] = str(uuid5(NAMESPACE_URL, f"sentinel-demo:{config.edge_id}:{label}"))
        outbox.put("/v1/sightings", event)
    for camera_id in config.simulator_camera_ids:
        event = health(edge_id=config.edge_id, camera_id=camera_id, status="ONLINE",
                       decoded_frames=0, inference_frames=0, reconnects=0,
                       outbox_depth=outbox.depth(), source_mode="synthetic")
        outbox.put("/v1/health", event)


def run_simulator(config: Config, stop: threading.Event, *, once: bool = False) -> None:
    outbox = Outbox(config.db_path)
    client = CentralClient(config.central_api_url, config.edge_api_key, outbox)
    seed_events(config, outbox)
    if once:
        client.flush_once()
        return
    delivery = client.start_retry_loop(stop)
    try:
        while not stop.wait(config.heartbeat_seconds):
            for camera_id in config.simulator_camera_ids:
                outbox.put("/v1/health", health(edge_id=config.edge_id, camera_id=camera_id,
                    status="ONLINE", decoded_frames=0, inference_frames=0, reconnects=0,
                    outbox_depth=outbox.depth(), source_mode="synthetic"))
    finally:
        stop.set()
        delivery.join(timeout=5)

