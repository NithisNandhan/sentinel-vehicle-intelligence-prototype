"""Bounded recorded-only evidence for the single dashboard.

This is a display adapter, not a recognition or acceptance stage. It never
changes native pixels, OCR evidence, thresholds, model state or delivery.
"""
import base64
import binascii
import json
import math
import re
import struct

IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
_PLATE = re.compile(r"^[A-Z0-9]{1,25}$")
_BASE64 = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
_STATES = {"reading", "uncertain", "model_accepted", "unreadable", "paused", "unavailable", "completed"}
_MODELS = {"ocr_model", "ocr_checkpoint_sha256", "ocr_device", "vehicle_model", "vehicle_device",
           "plate_model", "plate_device", "detector_profile", "detector_device", "plate_checkpoint_sha256"}
_STAGES = {"vehicle", "plate", "ocr", "reconstruction", "fusion"}
MAX_DIAGNOSTICS_BYTES = 3 * 1024 * 1024


class DiagnosticUnavailable(ValueError):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(detail)


def _integer(value, maximum=1_000_000_000):
    return value if type(value) is int and 0 <= value <= maximum else None


def _number(value, *, score=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    if score and not 0 <= value <= 1:
        return None
    return float(value)


def _label(value, limit=240):
    if (not isinstance(value, str) or len(value) > limit or "://" in value
            or "\\" in value or any(ord(c) < 32 for c in value)):
        return ""
    return value


def _plate(value):
    return value if isinstance(value, str) and _PLATE.fullmatch(value) else None


def _png(value, limit):
    if not isinstance(value, str) or not 0 < len(value) <= limit or not _BASE64.fullmatch(value):
        return None
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return value if 0 < width <= 8192 and 0 < height <= 8192 and width * height <= 4_194_304 else None


def _characters(value):
    result = []
    for row in (value if isinstance(value, list) else [])[:25]:
        if not isinstance(row, dict):
            continue
        char, score = row.get("character"), _number(row.get("score"), score=True)
        if not isinstance(char, str) or len(char) != 1 or not _PLATE.fullmatch(char) or score is None:
            continue
        alternatives = []
        for other in (row.get("alternatives") if isinstance(row.get("alternatives"), list) else [])[:3]:
            if not isinstance(other, dict):
                continue
            c, s = other.get("character"), _number(other.get("score"), score=True)
            if isinstance(c, str) and len(c) == 1 and _PLATE.fullmatch(c) and s is not None:
                alternatives.append({"character": c, "score": s})
        result.append({"character": char, "score": score, "alternatives": alternatives})
    return result


def _candidates(value):
    result = []
    for row in (value if isinstance(value, list) else [])[:200]:
        if not isinstance(row, dict):
            continue
        plate, score = _plate(row.get("plate")), _number(row.get("score"), score=True)
        if plate is not None and score is not None:
            result.append({"plate": plate, "score": score,
                           "minimum_character_score": _number(row.get("minimum_character_score"), score=True),
                           "log_score": _number(row.get("log_score")),
                           "observations": _integer(row.get("observations"))})
    return result


def _reconstruction(value):
    if not isinstance(value, dict):
        return None
    image = _png(value.get("image"), 160_000)
    reason = _label(value.get("reason"))
    if value.get("image") and image is None:
        reason = "Reconstruction image exceeds the display limit or is unavailable; original pixels were not resized"
    return {"status": _label(value.get("status"), 40) or "unavailable", "reason": reason,
            "image": image, "text": _plate(value.get("text")),
            "width": _integer(value.get("width"), 8192), "height": _integer(value.get("height"), 8192),
            "contributing_originals": _integer(value.get("contributing_originals"), 8) or 0,
            "elapsed_ms": _number(value.get("elapsed_ms")), "characters": _characters(value.get("characters")),
            "candidates": _candidates(value.get("candidates"))}


def diagnostic_view(snapshot, models, session_id, track_id, *, enabled):
    """Fresh allowlisted DTO for one requested track. No upstream dict escapes."""
    if not enabled or snapshot.get("source_mode") != "recorded":
        raise DiagnosticUnavailable(404, "Recorded diagnostics are not available")
    if (not isinstance(session_id, str) or not IDENTIFIER.fullmatch(session_id)
            or not isinstance(track_id, str) or not IDENTIFIER.fullmatch(track_id)):
        raise DiagnosticUnavailable(400, "Invalid diagnostic identifiers")
    if session_id != snapshot.get("session_id"):
        raise DiagnosticUnavailable(409, "The recognition session changed")
    camera_id = snapshot.get("camera_id")
    if not isinstance(camera_id, str) or not IDENTIFIER.fullmatch(camera_id):
        raise DiagnosticUnavailable(404, "No recorded session is available")
    debug = snapshot.get("debug")
    tracks = debug.get("tracks") if isinstance(debug, dict) else None
    selected = next((row for row in (tracks if isinstance(tracks, list) else [])[:512]
                     if isinstance(row, dict) and row.get("track_id") == track_id), None)
    if selected is None:
        raise DiagnosticUnavailable(404, "Track evidence is no longer available")
    reasons = [_label(item) for item in (selected.get("gate_reasons") if isinstance(selected.get("gate_reasons"), list) else [])[:24]]
    reasons = [item for item in reasons if item]
    frames, image_omitted = [], False
    for row in (selected.get("frames") if isinstance(selected.get("frames"), list) else [])[:8]:
        if not isinstance(row, dict):
            continue
        pts, width, height = _integer(row.get("pts_ms")), _integer(row.get("width"), 8192), _integer(row.get("height"), 8192)
        if pts is None or not width or not height:
            continue
        image = _png(row.get("image"), 80_000)
        image_omitted |= bool(row.get("image")) and image is None
        frames.append({"pts_ms": pts, "width": width, "height": height, "quality": _number(row.get("quality")),
                       "text": _plate(row.get("text")) or "", "characters": _characters(row.get("characters")), "image": image})
    if image_omitted:
        reasons.append("Some native crops exceed the display limit or are unavailable; original pixels were not resized")
    best = selected.get("best_guess") if isinstance(selected.get("best_guess"), dict) else {}
    box = selected.get("vehicle_box")
    if not isinstance(box, (list, tuple)) or len(box) != 4 or any(_integer(v, 65536) is None for v in box):
        box = []
    kind = selected.get("vehicle_class")
    if not ((type(kind) is int and 0 <= kind < 100) or kind in ("car", "motorcycle", "bicycle", "truck", "bus", "vehicle")):
        kind = "vehicle"
    masked = selected.get("masked_plate")
    if not isinstance(masked, str) or not re.fullmatch(r"[A-Z0-9*]{1,25}", masked):
        masked = None
    recognition_state = selected.get("recognition_state")
    track = {"track_id": track_id, "vehicle_box": list(box), "vehicle_class": kind,
             "seen_pts_ms": _integer(selected.get("seen_pts_ms")), "observations": _integer(selected.get("observations")) or 0,
             "recognition_state": recognition_state if isinstance(recognition_state, str) and recognition_state in _STATES else "unreadable",
             "masked_plate": masked, "best_reading": _plate(best.get("plate")), "model_score": _number(best.get("score"), score=True),
             "gate_reasons": reasons, "frames": frames, "candidates": _candidates(selected.get("candidates")),
             "reconstruction": _reconstruction(selected.get("reconstruction"))}
    model_view = {key: (_label(value, 128) or None) for key, value in models.items() if key in _MODELS}
    stages = debug.get("stage_ms") if isinstance(debug.get("stage_ms"), dict) else {}
    stage_view = {key: number for key, value in stages.items()
                  if key in _STAGES and (number := _number(value)) is not None and number >= 0}
    result = {"schema_version": "1.0", "source_mode": "recorded", "session_id": session_id,
              "camera_id": camera_id, "video_position_ms": _integer(snapshot.get("video_position_ms")),
              "models": model_view, "stage_ms": stage_view, "track": track}
    if len(json.dumps(result, allow_nan=False).encode()) > MAX_DIAGNOSTICS_BYTES:
        raise DiagnosticUnavailable(503, "Diagnostic display budget exceeded")
    return result
