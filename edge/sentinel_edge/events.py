"""Wire format constructors. No plaintext plates or imagery leave these functions."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from .privacy import mask_plate, plate_hmac


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sighting(*, edge_id: str, camera_id: str, plate: str, hmac_key: str,
             confidence: float, source_mode: str, stream_pts_ms: int | None = None,
             track_id: str | None = None, observed_at: str | None = None,
             model_version: str | None = None) -> dict:
    # `confidence` is the model's uncalibrated recognition score, not a measured
    # probability that the complete identifier is correct; retain wire compatibility.
    if source_mode not in {"live", "synthetic"}:
        raise ValueError("Invalid source mode")
    if not 0 <= confidence <= 1:
        raise ValueError("Invalid confidence")
    return {
        "schema_version": "1.0",
        "event_id": str(uuid4()),
        "edge_id": edge_id,
        "camera_id": camera_id,
        "observed_at": observed_at or utc_now(),
        "stream_pts_ms": stream_pts_ms,
        "track_id": track_id,
        "masked_plate": mask_plate(plate),
        "plate_hmac": plate_hmac(plate, hmac_key),
        "confidence": confidence,
        "source_mode": source_mode,
        "model_version": model_version,
    }


def health(*, edge_id: str, camera_id: str, status: str, decoded_frames: int,
           inference_frames: int, reconnects: int, outbox_depth: int,
           source_mode: str) -> dict:
    return {
        "event_id": str(uuid4()),
        "edge_id": edge_id,
        "camera_id": camera_id,
        "observed_at": utc_now(),
        "status": status,
        "decoded_frames": decoded_frames,
        "inference_frames": inference_frames,
        "reconnects": reconnects,
        "outbox_depth": outbox_depth,
        "source_mode": source_mode,
    }

