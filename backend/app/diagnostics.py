"""Bounded, allowlisted recorded-only evidence; never forward worker dictionaries."""
import math
import re

from .unified_contract import DiagnosticsSnapshot

TEXT = re.compile(r"^[A-Z0-9?*]{0,25}$")
MODEL = re.compile(r"^[A-Za-z0-9_.+: -]{0,160}$")
BASE64 = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
MODEL_KEYS = {"ocr_model", "ocr_device", "ocr_checkpoint_sha256", "vehicle_model", "vehicle_device",
              "vehicle_model_sha256", "plate_model", "plate_model_sha256", "plate_device", "detector_device", "detector_profile"}
STAGES = {"vehicle", "plate", "ocr", "reconstruction", "fusion"}
STATES = {"reading", "uncertain", "model_accepted", "unreadable", "paused", "unavailable", "completed"}


def image(value, limit=80_000):
    return value if isinstance(value, str) and 0 < len(value) <= limit and BASE64.fullmatch(value) else None


def text(value):
    return value if isinstance(value, str) and TEXT.fullmatch(value) else None


def number(value):
    return value if type(value) in {int, float} and math.isfinite(value) else None


def characters(values):
    result = []
    for item in values[:25] if isinstance(values, list) else []:
        if not isinstance(item, dict) or not text(item.get("character")) or len(item["character"]) != 1 or number(item.get("score")) is None:
            continue
        alternatives = []
        choices = item.get("alternatives")
        for choice in choices[:5] if isinstance(choices, list) else []:
            if isinstance(choice, dict) and text(choice.get("character")) and len(choice["character"]) == 1 and number(choice.get("score")) is not None:
                alternatives.append({"character": choice["character"], "score": choice["score"]})
        result.append({"character": item["character"], "score": item["score"], "alternatives": alternatives})
    return result


def candidates(values):
    result = []
    for item in values[:200] if isinstance(values, list) else []:
        if not isinstance(item, dict) or not text(item.get("plate")) or number(item.get("score")) is None:
            continue
        result.append({"plate": item["plate"], "score": item["score"],
                       "minimum_character_score": number(item.get("minimum_character_score")),
                       "log_score": number(item.get("log_score")),
                       "observations": item.get("observations") if type(item.get("observations")) is int else None})
    return result


def snapshot(value):
    track = value.get("track")
    if not isinstance(track, dict):
        raise ValueError("Missing track")
    reasons = track.get("gate_reasons")
    reasons = [item for item in (reasons[:32] if isinstance(reasons, list) else [])
               if isinstance(item, str) and len(item) <= 240 and "://" not in item and not any(ord(c) < 32 for c in item)]
    frames = []
    source_frames = track.get("frames")
    for item in source_frames[:8] if isinstance(source_frames, list) else []:
        if not isinstance(item, dict):
            continue
        frames.append({"pts_ms": item.get("pts_ms"), "width": item.get("width"), "height": item.get("height"),
                       "quality": number(item.get("quality")), "text": text(item.get("text")) or "",
                       "characters": characters(item.get("characters")), "image": image(item.get("image"))})
    reconstruction = None
    raw = track.get("reconstruction")
    if isinstance(raw, dict):
        reconstruction = {key: raw.get(key) for key in ("width", "height", "contributing_originals", "elapsed_ms") if key in raw}
        reconstruction.update(status=raw.get("status", "not_run") if MODEL.fullmatch(str(raw.get("status", "not_run"))) else "unavailable",
                              reason=str(raw.get("reason", ""))[:240] if "://" not in str(raw.get("reason", "")) and not any(ord(c) < 32 for c in str(raw.get("reason", ""))) else "Unavailable",
                              image=image(raw.get("image"), 160_000), text=text(raw.get("text")),
                              characters=characters(raw.get("characters")), candidates=candidates(raw.get("candidates")))
    models = value.get("models")
    stages = value.get("stage_ms")
    return DiagnosticsSnapshot.model_validate({
        "schema_version": value.get("schema_version", "1.0"), "source_mode": value.get("source_mode"),
        "session_id": value.get("session_id"), "camera_id": value.get("camera_id"),
        "video_position_ms": value.get("video_position_ms"),
        "models": {key: item for key, item in (models.items() if isinstance(models, dict) else [])
                   if key in MODEL_KEYS and (item is None or isinstance(item, str) and MODEL.fullmatch(item))},
        "stage_ms": {key: item for key, item in (stages.items() if isinstance(stages, dict) else [])
                     if key in STAGES and number(item) is not None and item >= 0},
        "track": {"track_id": track.get("track_id"), "vehicle_box": track.get("vehicle_box", []),
                  "vehicle_class": track.get("vehicle_class") if (type(track.get("vehicle_class")) is int or track.get("vehicle_class") in {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}) else "vehicle", "seen_pts_ms": track.get("seen_pts_ms"),
                  "observations": track.get("observations", 0),
                  "recognition_state": track.get("recognition_state") if track.get("recognition_state") in STATES else "reading",
                  "masked_plate": text(track.get("masked_plate")), "best_reading": text(track.get("best_reading")),
                  "model_score": number(track.get("model_score")), "gate_reasons": reasons, "frames": frames,
                  "candidates": candidates(track.get("candidates")), "reconstruction": reconstruction},
    })
