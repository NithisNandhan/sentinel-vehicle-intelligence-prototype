"""Conservative same-image OCR suggestions, never additional temporal votes."""
from __future__ import annotations

import math

from .ctc_fusion import greedy_text, sequence_score
from .indian_ocr_finetune import edit_distance
from .ocr_crop_experiment import contrast_view, padded_view
from .temporal import indian_plate_format

POLICY = {"retry_minimum_character": .8, "maximum_original_log_gap": .5,
          "minimum_original_character": .2, "minimum_view_mean": .85,
          "minimum_view_character": .5,
          "maximum_edits": 1}


def reading_summary(rows, tokens):
    text = greedy_text(rows, tokens)
    score, peaks = sequence_score(rows, tokens, text)
    return {"reading": text, "log_score": score if math.isfinite(score) else None,
            "mean": sum(peaks) / len(peaks) if peaks else 0.,
            "minimum": min(peaks, default=0.)}


def needs_retry(summary):
    return (not indian_plate_format(summary["reading"])
            or summary["minimum"] < POLICY["retry_minimum_character"])


def choose_suggestion(native_rows, margin_rows, contrast_rows, tokens):
    """Require view agreement AND original-image support, with fixed thresholds.

    The model receives images only. No registration database, label, grammar
    correction or probability renormalisation supplies a character. The selected
    string is an unconfirmed proposal, not an accepted sighting.
    """
    native = reading_summary(native_rows, tokens)
    views = [reading_summary(rows, tokens) for rows in (margin_rows, contrast_rows)]
    result = {"baseline": native["reading"], "reading": native["reading"],
              "changed": False, "reason": "native_retained", "views": views,
              "independent_observations_added": 0}
    candidate = views[0]["reading"]
    if (not needs_retry(native) or candidate != views[1]["reading"]
            or candidate == native["reading"] or not indian_plate_format(candidate)
            or edit_distance(candidate, native["reading"]) > POLICY["maximum_edits"]
            or min(view["mean"] for view in views) < POLICY["minimum_view_mean"]
            or min(view["minimum"] for view in views) < POLICY["minimum_view_character"]):
        return result
    log_score, peaks = sequence_score(native_rows, tokens, candidate)
    if native["log_score"] is None or not math.isfinite(log_score):
        return result
    gap = native["log_score"] - log_score
    if (not math.isfinite(gap) or gap > POLICY["maximum_original_log_gap"]
            or min(peaks, default=0.) < POLICY["minimum_original_character"]):
        result["reason"] = "insufficient_original_support"
        return result
    return {**result, "reading": candidate, "changed": True,
            "reason": "views_agree_with_original_support", "original_log_gap": gap}


def refine(reader, image, *, native_rows=None):
    """Bounded two-view retry. Original posteriors are returned unchanged."""
    rows = reader.probabilities(image) if native_rows is None else native_rows
    summary = reading_summary(rows, reader.tokens)
    if not needs_retry(summary):
        return {"baseline": summary["reading"], "reading": summary["reading"],
                "changed": False, "reason": "native_strong", "ocr_calls": 1,
                "independent_observations_added": 0}
    margin = reader.probabilities(padded_view(image))
    contrast = reader.probabilities(contrast_view(image))
    return {**choose_suggestion(rows, margin, contrast, reader.tokens), "ocr_calls": 3}
