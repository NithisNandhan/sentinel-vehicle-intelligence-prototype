"""Wire format constructors. No plaintext plates or imagery leave these functions."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from .privacy import mask_plate, normalize_plate, plate_hmac


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



# ---------------------------------------------------------------------------
# Result-state events (schema 2.0, POST /v1/recognition-results). One event per
# vehicle passage. Separate from sightings: an unreadable or partial passage has
# no fingerprint and must not enter the fingerprint-required sightings table.
# A confirmed passage still sends its ordinary 1.0 sighting (alerts come only
# from those); its result event links to it. See docs/API_EVENT_CONTRACT.md.
# ---------------------------------------------------------------------------

RESULT_SCHEMA_VERSION = "2.0"
RESULT_STATES = ("confirmed", "partial", "unreadable")
_PATTERN_MASK = __import__("re").compile(r"^[A-Z0-9?]{2}\**[A-Z0-9?]{2}$")


def recognition_result(*, edge_id: str, camera_id: str, passage_id: str, result_state: str, source_mode: str,
                       observed_at: str, last_observed_at: str, event_id: str, stream_pts_ms: int | None = None,
                       end_pts_ms: int | None = None, track_ids=(), plate: str | None = None,
                       hmac_key: str | None = None, masked_pattern: str | None = None,
                       plate_length: int | None = None, unknown_positions=(), reason: str | None = None,
                       review_required: bool = False, length_certain: bool = True,
                       evidence_ref: str | None = None, observations: int = 0, quality: dict | None = None,
                       layout: str = "unknown", format_status: str | None = None,
                       sighting_event_id: str | None = None, watchlist_lead: dict | None = None,
                       model_version: str | None = None, profile: str | None = None,
                       policy_version: str | None = None, session_id: str | None = None,
                       time_basis: str | None = None, processed_at: str | None = None) -> dict:
    if result_state not in RESULT_STATES:
        raise ValueError("Unknown result state")
    if source_mode not in {"live", "synthetic", "recorded"}:
        raise ValueError("Invalid source mode")
    if source_mode == "recorded" and (not session_id or stream_pts_ms is None
                                      or time_basis not in {"source_clock", "processing_time"}):
        raise ValueError("Recorded results need a session, video PTS and time basis")
    event = {
        "schema_version": RESULT_SCHEMA_VERSION, "event_type": "recognition_result",
        "event_id": event_id, "edge_id": edge_id, "camera_id": camera_id, "passage_id": passage_id,
        "result_state": result_state, "observed_at": observed_at, "last_observed_at": last_observed_at,
        "stream_pts_ms": stream_pts_ms, "end_pts_ms": end_pts_ms, "track_ids": [str(t) for t in track_ids][:16],
        "observations": int(observations), "layout": layout, "review_required": bool(review_required),
        "source_mode": source_mode, "model_version": model_version, "profile": profile,
        "policy_version": policy_version, "quality": _quality(quality),
    }
    if result_state == "confirmed":
        if not plate or not hmac_key or reason or unknown_positions or watchlist_lead:
            raise ValueError("A confirmed result needs the plate fingerprint and nothing uncertain")
        event.update(masked_plate=mask_plate(plate), plate_hmac=plate_hmac(plate, hmac_key),
                     plate_length=len(normalize_plate(plate)), format_status=format_status,
                     sighting_event_id=sighting_event_id)
    elif result_state == "partial":
        if plate or not masked_pattern or not _PATTERN_MASK.fullmatch(masked_pattern) or not plate_length:
            raise ValueError("A partial result carries only its masked pattern")
        if len(masked_pattern) != plate_length or any(not 0 <= i < plate_length for i in unknown_positions):
            raise ValueError("Partial positions must lie inside the agreed length")
        if not reason:
            raise ValueError("A partial result states why it is not confirmed")
        event.update(masked_plate=masked_pattern, plate_length=plate_length,
                     unknown_positions=sorted(set(int(i) for i in unknown_positions)), reason=reason,
                     length_certain=bool(length_certain), format_status=format_status,
                     evidence_ref=evidence_ref, watchlist_lead=watchlist_lead)
    else:
        if plate or masked_pattern or not reason or watchlist_lead:
            raise ValueError("An unreadable result carries a reason and no text")
        event.update(reason=reason, evidence_ref=evidence_ref)
    if source_mode == "recorded":
        event.update(session_id=session_id, time_basis=time_basis, processed_at=processed_at or utc_now())
    return event


_QUALITY_KEYS = ("height_px", "width_px", "sharpness", "contrast", "glare_fraction", "dark_fraction",
                 "mean_brightness", "angle_deg", "blurred", "low_contrast", "glare", "low_light", "tilted")


def _quality(values):
    if not values:
        return None
    return {key: values[key] for key in _QUALITY_KEYS if key in values}
