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
             model_version: str | None = None, session_id: str | None = None,
             time_basis: str | None = None, event_id: str | None = None,
             processed_at: str | None = None) -> dict:
    # `confidence` is the model's uncalibrated recognition score, not a measured
    # probability that the complete identifier is correct; retain wire compatibility.
    if source_mode not in {"live", "synthetic", "recorded"}:
        raise ValueError("Invalid source mode")
    if not 0 <= confidence <= 1:
        raise ValueError("Invalid confidence")
    if source_mode == "recorded":
        # Downloaded footage must stay traceable to one run and one video time,
        # and must say whether observed_at is the recording's clock or ours.
        if not session_id or stream_pts_ms is None or time_basis not in {"source_clock", "processing_time"}:
            raise ValueError("Recorded sightings need a session, video PTS and time basis")
    elif session_id is not None or time_basis is not None or processed_at is not None:
        raise ValueError("Session and time basis apply to recorded sightings only")
    event = {
        "schema_version": "1.0",
        "event_id": event_id or str(uuid4()),
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
    if source_mode == "recorded":
        # observed_at may be the recording's own clock; always keep ours too.
        event.update(session_id=session_id, time_basis=time_basis, processed_at=processed_at or utc_now())
    return event


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

