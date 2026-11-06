"""Vehicle passages: plate-level identity across tracker IDs and one final outcome each.

A passage is the evidence one physical vehicle leaves while it is visible. Its
outcome (result_state.py) is produced once, when the passage ends, so outcomes
are counted after association and deduplication, never once per tracker ID or
per frame. Pixels and plate text stay in edge memory.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import secrets

from .vision import track_consistent

# A tracker may drop and re-create a vehicle's ID within this video time.
REASSOCIATION_MS = 1500
# Mean absolute grey-level difference below which two crops are the same view.
NEAR_DUPLICATE_MAD = 2.0
SIGNATURE_SIZE = (48, 16)
MAX_EVIDENCE_CROPS = 3
# Tracks seen in fewer samples, with no plate evidence, are likely detector noise.
MIN_VEHICLE_SAMPLES = 2


def new_passage_id():
    return secrets.token_hex(8)


def signature(crop):
    """Small grey thumbnail used only to recognise repeated views of one plate."""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return cv2.resize(gray, SIGNATURE_SIZE, interpolation=cv2.INTER_AREA).astype(np.int16)


def near_duplicate(previous, current, threshold=NEAR_DUPLICATE_MAD):
    import numpy as np
    return previous is not None and float(np.abs(previous - current).mean()) < threshold


def reassociations(new_tracks, histories, current_ids, pts_ms, *, window_ms=REASSOCIATION_MS):
    """{new_track_id: lost_track_id} where the pairing is unambiguous both ways.

    A lost track qualifies when it is absent from this frame, was seen within the
    window, still holds plate evidence, and its last box is geometrically and
    class-consistent with the new box. Two candidates either way means no transfer:
    mixing two vehicles' evidence is worse than starting again.
    """
    options = {}
    for track_id, box, vehicle_class in new_tracks:
        options[track_id] = [key for key, history in histories.items()
                             if key not in current_ids and 0 < pts_ms - history.seen_pts <= window_ms
                             and history.reader.observation_count
                             and track_consistent(history.box, box, history.vehicle_class, vehicle_class)]
    claimed = Counter(key for choices in options.values() for key in choices)
    return {new: choices[0] for new, choices in options.items() if len(choices) == 1 and claimed[choices[0]] == 1}


def empty_reason(failures):
    """Why a passage produced no OCR evidence, from its plate-search failures."""
    if not failures:
        return 'no_plate_found'
    text = failures.most_common(1)[0][0].lower()
    if 'smaller than' in text or 'too small' in text:
        return 'plate_too_small'
    if 'not a plate' in text or 'non-plate' in text or 'shape cannot' in text or 'score too low' in text:
        return 'non_plate_rejected'
    return 'no_plate_found'


@dataclass(repr=False)
class PassageOutcome:
    passage_id: str
    track_ids: tuple
    vehicle_class: int
    first_pts: int
    last_pts: int
    result: object                     # RecognitionResult
    cause: str                          # expired, reset or end_of_source
    plate_detections: int = 0
    searches: int = 0
    vehicle_samples: int = 0
    first_box: tuple | None = None
    last_box: tuple | None = None
    evidence: list = field(default_factory=list)   # [{'pts_ms', 'crop', 'quality'}], best first
    frame_jpeg: bytes | None = None
    frame_pts: int | None = None
    indicators: dict | None = None
    confirmed_plate: str | None = None  # The plate a sighting was sent for, if any.
    revisited: bool = False             # Final evidence no longer supports an earlier confirmation.
    layouts: dict = field(default_factory=dict)
