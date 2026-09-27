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

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from .config import get_settings
from .security import check_operator_key


router = APIRouter(prefix="/v1/recognition", tags=["recognition"],
                   dependencies=[Depends(check_operator_key)])
_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_PROFILE = re.compile(r"^[A-Za-z0-9_.+:-]{1,80}$")
_MASK = re.compile(r"^[A-Z0-9]{0,4}\*{2,16}[A-Z0-9]{0,2}$")
_STATES = {"reading", "uncertain", "model_accepted", "unreadable", "paused", "unavailable"}
_UNAVAILABLE = "Recognition service unavailable"


class SessionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: StrictStr | None = Field(pattern=_ID.pattern)


class HeartbeatIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: StrictStr = Field(pattern=_ID.pattern)


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
            raw = response.read(1_048_577)
            if len(raw) > 1_048_576:
                raise unavailable()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise unavailable()
            return value
    except urllib.error.HTTPError as error:
        # A heartbeat conflict must remain visible, not silently switch cameras.
        if error.code == 409:
            raise HTTPException(status_code=409, detail="Recognition session camera conflict") from None
        raise unavailable() from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise unavailable() from None


def _count(value) -> int:
    return value if type(value) is int and 0 <= value <= 1_000_000_000 else 0


def _id(value):
    return value if isinstance(value, str) and _ID.fullmatch(value) else None


def status_metadata(value: dict) -> dict:
    """Construct a fresh allowlisted payload; never pass upstream objects through."""
    status = value.get("status")
    if not isinstance(status, str) or status not in {"paused", "connecting", "live", "unavailable"}:
        raise unavailable()
    state = value.get("recognition_state", status if status in _STATES else "reading")
    if not isinstance(state, str) or state not in _STATES:
        raise unavailable()
    result = {
        "schema_version": "1.0", "status": status,
        "camera_id": _id(value.get("camera_id")), "source_mode": "live",
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


@router.post("/session")
def session(body: SessionIn):
    return status_metadata(request_edge("POST", "/internal/session", body.model_dump()))


@router.post("/heartbeat")
def heartbeat(body: HeartbeatIn):
    return status_metadata(request_edge("POST", "/internal/heartbeat", body.model_dump()))
