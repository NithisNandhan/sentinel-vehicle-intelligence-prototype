"""Operator-only, metadata-only gateway to the shared local recognition session.

This intentionally does not proxy images, hypotheses or arbitrary edge endpoints.
Do not return upstream exception bodies: they can contain authenticated URLs.
"""

import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request

from fastapi import APIRouter, Depends, HTTPException, Response, Query
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator, ValidationError

from .config import get_settings
from .security import check_operator_key
from .unified_contract import DiagnosticsSnapshot


router = APIRouter(prefix="/v1/recognition", tags=["recognition"],
                   dependencies=[Depends(check_operator_key)])
_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_PROFILE = re.compile(r"^[A-Za-z0-9_.+:-]{1,80}$")
_MASK = re.compile(r"^[A-Z0-9]{0,4}\*{2,16}[A-Z0-9]{0,2}$")
_STATES = {"reading", "uncertain", "model_accepted", "unreadable", "paused", "unavailable", "completed"}
# `processing` and `completed` exist only for a recording; `live` only for a camera.
_STATUSES = {"paused", "connecting", "live", "unavailable", "processing", "completed"}
_VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck", "vehicle"}
_SESSION = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_FAILURE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_UNAVAILABLE = "Recognition service unavailable"


class SessionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: StrictStr | None = Field(pattern=_ID.pattern)


class HeartbeatIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: StrictStr = Field(pattern=_ID.pattern)


class ControlIn(BaseModel):
    """Playback control for the selected recording. Cameras have no such controls."""
    model_config = ConfigDict(extra="forbid")
    action: Literal["pause", "resume", "restart", "seek"]
    position_ms: StrictInt | None = Field(default=None, ge=0, le=1_000_000_000)

    @model_validator(mode="after")
    def only_seek_takes_a_position(self):
        if (self.action == "seek") != (self.position_ms is not None):
            raise ValueError("position_ms is required for seek and not allowed otherwise")
        return self


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the edge credential to a redirected host or login page.
        return None


def unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail=_UNAVAILABLE)


def request_edge(method: str, path: str, payload: dict | None = None) -> dict:
    settings = get_settings()
    base = settings.recognition_gateway_url.rstrip("/")
    try:
        parsed = urllib.parse.urlsplit(base)
    except ValueError:
        raise unavailable() from None
    if (not base or parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise unavailable()
    request = urllib.request.Request(
        base + path, method=method,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"X-Edge-Key": settings.edge_api_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.build_opener(_NoRedirect()).open(
                request, timeout=3 if method == "GET" else 5) as response:
            if response.headers.get_content_type() != "application/json":
                raise unavailable()
            limit = 3 * 1024 * 1024 if path.startswith("/internal/diagnostics?") else 1_048_576
            raw = response.read(limit + 1)
            if len(raw) > limit:
                raise unavailable()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise unavailable()
            return value
    except urllib.error.HTTPError as error:
        if error.code == 404 and path.startswith("/internal/diagnostics?"):
            raise HTTPException(status_code=404, detail="Track evidence is no longer available") from None
        # A heartbeat conflict must remain visible, not silently switch cameras.
        if error.code == 409:
            detail = ("Playback control not available for the current session" if path.endswith("/control")
                      else "Recognition session camera conflict")
            raise HTTPException(status_code=409, detail=detail) from None
        raise unavailable() from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise unavailable() from None


def _count(value) -> int:
    return value if type(value) is int and 0 <= value <= 1_000_000_000 else 0


def _id(value):
    return value if isinstance(value, str) and _ID.fullmatch(value) else None


def _display(value, fallback):
    # Names are display text, not a place to transport a URL or control chars.
    if not isinstance(value, str) or not value or len(value) > 160 or "://" in value or any(ord(c) < 32 for c in value):
        return fallback
    return value


def _milliseconds(value):
    return value if type(value) is int and 0 <= value <= 1_000_000_000 else None


def status_metadata(value: dict) -> dict:
    """Construct a fresh allowlisted payload; never pass upstream objects through."""
    status = value.get("status")
    if not isinstance(status, str) or status not in _STATUSES:
        raise unavailable()
    # Downloaded footage is labelled recorded end to end; it is never "live".
    recorded = value.get("source_mode") == "recorded"
    if (recorded and status == "live") or (not recorded and status in {"processing", "completed"}):
        raise unavailable()
    state = value.get("recognition_state", status if status in _STATES else "reading")
    if not isinstance(state, str) or state not in _STATES:
        raise unavailable()
    result = {
        "schema_version": "1.0", "status": status,
        "camera_id": _id(value.get("camera_id")), "source_mode": "recorded" if recorded else "live",
        "recognition_state": state,
        "vehicle_boxes": _count(value.get("vehicle_boxes")),
        "decoded_frames": _count(value.get("decoded_frames")),
        "analysed_frames": _count(value.get("analysed_frames")),
        "plate_crops": _count(value.get("plate_crops")),
        "unique_model_accepted_unverified": _count(value.get("unique_model_accepted_unverified")),
        "model_profile": None, "tracks": [],
    }
    profile = value.get("model_profile")
    if isinstance(profile, str) and _PROFILE.fullmatch(profile):
        result["model_profile"] = profile
    device = value.get("inference_device")
    result["inference_device"] = device if isinstance(device, str) and device in {"cpu", "gpu"} else None
    failure = value.get("failure_code")
    if status == "unavailable" and isinstance(failure, str) and _FAILURE.fullmatch(failure):
        result["failure_code"] = failure  # A class name only, never a message.
    if recorded:
        session = value.get("session_id")
        basis, pacing = value.get("time_basis"), value.get("pacing")
        stale = value.get("stale_after_seconds")
        result.update({
            "session_id": session if isinstance(session, str) and _SESSION.fullmatch(session) else None,
            "recording_name": _display(value.get("recording_name"), result["camera_id"]),
            "video_position_ms": _milliseconds(value.get("video_position_ms")),
            "video_duration_ms": _milliseconds(value.get("video_duration_ms")),
            "time_basis": basis if isinstance(basis, str) and basis in {"source_clock", "processing_time"} else None,
            "pacing": pacing if isinstance(pacing, str) and pacing in {"paced", "unpaced"} else None,
            "stale_after_seconds": stale if type(stale) is int and 1 <= stale <= 600 else 30,
        })
    age = value.get("frame_age_seconds")
    if type(age) in (int, float) and math.isfinite(age) and age >= 0:
        result["frame_age_seconds"] = age
    resolution = value.get("resolution")
    if (isinstance(resolution, (list, tuple)) and len(resolution) == 2
            and all(type(v) is int and 0 < v <= 32768 for v in resolution)):
        result["resolution"] = list(resolution)
    tracks = value.get("tracks")
    if isinstance(tracks, list):
        for row in tracks[:32]:
            if not isinstance(row, dict) or not _id(row.get("track_id")):
                continue
            masked = row.get("masked_plate")
            track_state = row.get("recognition_state")
            result["tracks"].append({
                "track_id": _id(row.get("track_id")),
                "observations": _count(row.get("observations")),
                "masked_plate": masked if isinstance(masked, str) and _MASK.fullmatch(masked) else None,
                "recognition_state": track_state if isinstance(track_state, str) and track_state in _STATES else "uncertain",
                "candidate_count": min(_count(row.get("candidate_count")), 200),
            })
    # Vehicles seen in this run: masked text and counts only, never a crop or full reading.
    result["vehicles_seen"] = []
    seen = value.get("vehicles_seen")
    for row in (seen if isinstance(seen, list) else [])[:50]:
        if not isinstance(row, dict) or not _id(row.get("track_id")):
            continue
        masked, state, kind = row.get("masked_plate"), row.get("state"), row.get("vehicle_class")
        confidence, watch = row.get("confidence"), row.get("watchlist")
        accepted = state == "model_accepted"
        result["vehicles_seen"].append({
            "track_id": row["track_id"],
            "vehicle_class": kind if isinstance(kind, str) and kind in _VEHICLE_CLASSES else "vehicle",
            "confidence": round(confidence, 3) if type(confidence) in (int, float) and math.isfinite(confidence)
            and 0 <= confidence <= 1 else None,
            "first_ms": _milliseconds(row.get("first_ms")), "last_ms": _milliseconds(row.get("last_ms")),
            "observations": _count(row.get("observations")),
            "state": state if isinstance(state, str) and state in _STATES else "unreadable",
            "masked_plate": masked if isinstance(masked, str) and _MASK.fullmatch(masked) else None,
            # Only an accepted reading may carry a watchlist flag to the operator.
            "watchlist": watch if accepted and isinstance(watch, str) and watch in {"possible_match", "no_match"} else None,
        })
    delivery = value.get("delivery")
    if isinstance(delivery, dict):
        result["delivery"] = {"enabled": delivery.get("enabled") is True,
                              "outbox_depth": _count(delivery.get("outbox_depth"))}
    return result


@router.get("/status")
def status():
    return status_metadata(request_edge("GET", "/internal/status"))


@router.get("/cameras")
def cameras():
    value = request_edge("GET", "/internal/cameras")
    if type(value.get("available")) is not bool or not isinstance(value.get("cameras"), list):
        raise unavailable()
    scope = value.get("scope", "authorised_catalogue" if value["available"] else "unavailable")
    if not isinstance(scope, str) or scope not in {"authorised_catalogue", "configured_relays", "unavailable"}:
        raise unavailable()
    if (value["available"] and scope == "unavailable") or (not value["available"] and scope != "unavailable"):
        raise unavailable()
    rows = []
    for row in value["cameras"][:10000]:
        if not isinstance(row, dict) or not _id(row.get("id")):
            continue
        name = row.get("name")
        # Names are display text, not a place to transport a URL or control chars.
        if not isinstance(name, str) or len(name) > 160 or "://" in name or any(ord(c) < 32 for c in name):
            name = row["id"]
        rows.append({"id": row["id"], "name": name})
    return {"available": value["available"], "scope": scope, "cameras": rows if value["available"] else []}


@router.get("/recordings")
def recordings():
    """Permitted local recordings known to the edge. Names and durations only."""
    value = request_edge("GET", "/internal/cameras")
    rows = []
    listed = value.get("recordings")
    for row in (listed if isinstance(listed, list) else [])[:1000]:
        if not isinstance(row, dict) or not _id(row.get("id")):
            continue
        basis = row.get("time_basis")
        rows.append({"id": row["id"], "name": _display(row.get("name"), row["id"]),
                     "duration_ms": _milliseconds(row.get("duration_ms")),
                     "time_basis": basis if isinstance(basis, str) and basis in {"source_clock", "processing_time"}
                     else "processing_time"})
    return {"recordings": rows}


_BASE64 = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
_READING = re.compile(r"^[A-Z0-9]{4,12}$")


@router.get("/diagnostics", response_model=DiagnosticsSnapshot)
def diagnostics(response: Response, session_id: str = Query(pattern=_SESSION.pattern),
                track_id: str = Query(pattern=_ID.pattern)):
    if not get_settings().dashboard_recorded_evidence:
        raise HTTPException(status_code=404, detail="Recorded evidence view is not enabled")
    query = urllib.parse.urlencode({"session_id": session_id, "track_id": track_id})
    value = request_edge("GET", "/internal/diagnostics?" + query)
    if value.get("source_mode") != "recorded":
        raise HTTPException(status_code=404, detail="No recorded evidence available")
    if not _id(value.get("camera_id")):
        raise unavailable()
    if value.get("session_id") != session_id:
        raise HTTPException(status_code=409, detail="Recognition session changed")
    if not isinstance(value.get("track"), dict) or value["track"].get("track_id") != track_id:
        raise HTTPException(status_code=404, detail="Track evidence is no longer available")
    try:
        from .diagnostics import snapshot
        result = snapshot(value)
    except (ValidationError, ValueError, TypeError):
        raise unavailable() from None
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return result


def _image(value, limit):
    return value if isinstance(value, str) and 0 < len(value) <= limit and _BASE64.fullmatch(value) else None


@router.get("/evidence")
def evidence(response: Response, mode: Literal["original", "overlays"] = "overlays"):
    """Analysed frame and full unverified readings for a permitted recording.

    Opt-in (DASHBOARD_RECORDED_EVIDENCE) and recorded sources only. Nothing is
    stored: this passes the edge's current in-memory view to the operator.
    """
    if not get_settings().dashboard_recorded_evidence:
        raise HTTPException(status_code=404, detail="Recorded evidence view is not enabled")
    path = "/internal/evidence?mode=original" if mode == "original" else "/internal/evidence"
    value = request_edge("GET", path)
    if value.get("source_mode") != "recorded":
        raise HTTPException(status_code=404, detail="No recording is being analysed")
    response.headers["Cache-Control"] = "no-store, max-age=0"
    session = value.get("session_id")
    rows = []
    listed = value.get("vehicles")
    for row in (listed if isinstance(listed, list) else [])[:30]:
        if not isinstance(row, dict) or not _id(row.get("track_id")):
            continue
        state, kind, reading = row.get("state"), row.get("vehicle_class"), row.get("reading")
        confidence, watch = row.get("confidence"), row.get("watchlist")
        rows.append({
            "track_id": row["track_id"],
            "vehicle_class": kind if isinstance(kind, str) and kind in _VEHICLE_CLASSES else "vehicle",
            "confidence": round(confidence, 3) if type(confidence) in (int, float) and math.isfinite(confidence)
            and 0 <= confidence <= 1 else None,
            "first_ms": _milliseconds(row.get("first_ms")), "last_ms": _milliseconds(row.get("last_ms")),
            "observations": _count(row.get("observations")),
            "state": state if isinstance(state, str) and state in _STATES else "unreadable",
            "reading": reading if isinstance(reading, str) and _READING.fullmatch(reading) else None,
            "thumb": _image(row.get("thumb"), 80_000),
            "watchlist": watch if isinstance(watch, str) and watch in {"possible_match", "no_match"} else None,
        })
    return {"source_mode": "recorded",
            "session_id": session if isinstance(session, str) and _SESSION.fullmatch(session) else None,
            "video_position_ms": _milliseconds(value.get("video_position_ms")),
            "frame": _image(value.get("frame"), 900_000), "vehicles": rows}


@router.post("/control")
def control(body: ControlIn):
    return status_metadata(request_edge("POST", "/internal/control", body.model_dump(exclude_none=True)))


@router.post("/session")
def session(body: SessionIn):
    return status_metadata(request_edge("POST", "/internal/session", body.model_dump()))


@router.post("/heartbeat")
def heartbeat(body: HeartbeatIn):
    return status_metadata(request_edge("POST", "/internal/heartbeat", body.model_dump()))
