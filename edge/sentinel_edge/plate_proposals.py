"""Multi-source plate proposal pool (SENTINEL_MULTI_SOURCE_PLATES; GPU profile only).

Sources, for one vehicle region at native resolution:
1. the existing vehicle-crop plate detector (`TargetedPlateSearch`), always;
2. the plate detector on overlapping native tiles of a large region (automatic_crops.windows);
3. a text-region detector (text_boundaries.TextBoundaryDetector), whose boxes are
   only *proposals*: generic text (shop signs, branding) is filtered by shape,
   size and position, and a text-only box ranks below any plate-detector box.

Proposals are deduplicated by overlap, ranked, and the top one is returned with a
safe margin so plate edges are not clipped. Alternatives are kept for diagnosis.
A per-frame call and time budget keeps the cost bounded. This changes where OCR
looks, never how a reading is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .automatic_crops import overlap, windows

PLATE_ASPECT = (1.2, 8.)          # Single-row ~4.5, two-row ~1.7 (width / height).
TEXT_REGION_AREA = (.002, .25)    # Fraction of the vehicle region a plate can occupy.
MARGIN = .06


@dataclass(frozen=True)
class PoolProposal:
    box: tuple
    sources: tuple
    score: float
    detector_confidence: float | None = None


def plate_like(box, region_shape):
    w, h = box[2] - box[0], box[3] - box[1]
    if w <= 0 or h <= 0:
        return False
    area = w * h / float(region_shape[0] * region_shape[1])
    centre_y = (box[1] + box[3]) / 2 / region_shape[0]
    return (PLATE_ASPECT[0] <= w / h <= PLATE_ASPECT[1] and TEXT_REGION_AREA[0] <= area <= TEXT_REGION_AREA[1]
            and centre_y >= .25)   # Plates sit in the lower three quarters of a vehicle box.


def with_margin(box, shape, margin=MARGIN):
    x1, y1, x2, y2 = box
    mx, my = math.ceil((x2 - x1) * margin), math.ceil((y2 - y1) * margin)
    return (max(0, x1 - mx), max(0, y1 - my), min(shape[1], x2 + mx), min(shape[0], y2 + my))


def rank(proposals, region_shape, *, top_k=3, merge_iou=.45):
    """Deduplicate and rank (box, source, confidence) proposals.

    Score: plate-detector support dominates; agreement between sources adds;
    a text-only proposal must be plate-like to be kept at all.
    """
    clusters = []
    for box, source, confidence in proposals:
        if source == 'text' and not plate_like(box, region_shape):
            continue
        for cluster in clusters:
            if overlap(box, cluster['box']) >= merge_iou:
                cluster['members'].append((box, source, confidence))
                break
        else:
            clusters.append({'box': box, 'members': [(box, source, confidence)]})
    ranked = []
    for cluster in clusters:
        sources = tuple(sorted({source for _, source, _ in cluster['members']}))
        confidences = [c for _, s, c in cluster['members'] if c is not None and s != 'text']
        plate_support = any(s != 'text' for s in sources)
        score = (2. if plate_support else 0.) + .5 * (len(sources) - 1) + (max(confidences) if confidences else 0.)
        # Union of agreeing boxes avoids clipping an edge one source cut off.
        members = [b for b, _, _ in cluster['members']]
        union = (min(b[0] for b in members), min(b[1] for b in members),
                 max(b[2] for b in members), max(b[3] for b in members))
        ranked.append(PoolProposal(union, sources, round(score, 4), max(confidences) if confidences else None))
    ranked.sort(key=lambda p: (-p.score, p.box))
    return ranked[:top_k]


class ProposalPool:
    """Drop-in for `TargetedPlateSearch.box(region)` that pools several sources."""
    def __init__(self, primary, *, tile_detector=None, text_detector=None, tile_size=640, min_tile_region=900,
                 max_calls_per_frame=24, frame_budget_ms=150., top_k=3, clock=time.monotonic):
        if not 1 <= max_calls_per_frame <= 256 or frame_budget_ms <= 0 or not 1 <= top_k <= 8:
            raise ValueError('Bounded proposal work required')
        self.primary, self.tile_detector, self.text_detector = primary, tile_detector, text_detector
        self.tile_size, self.min_tile_region = tile_size, min_tile_region
        self.max_calls, self.budget_ms, self.top_k, self.clock = max_calls_per_frame, frame_budget_ms, top_k, clock
        self.calls = 0
        self.started = None
        self.last_calls = self.last_fallback = 0
        self.last_confidence = None
        self.last_alternatives = ()
        self.skipped_for_budget = 0

    def begin_frame(self):
        self.calls, self.started = 0, self.clock()

    def _within_budget(self):
        if self.started is None:
            self.begin_frame()
        spent = (self.clock() - self.started) * 1000
        if self.calls >= self.max_calls or spent >= self.budget_ms:
            self.skipped_for_budget += 1
            return False
        return True

    def box(self, region):
        proposals = []
        primary = self.primary.box(region)
        self.last_calls = getattr(self.primary, 'last_calls', 1)
        self.last_fallback = getattr(self.primary, 'last_fallback', False)
        if primary is not None:
            proposals.append((tuple(primary), 'plate', getattr(self.primary, 'last_confidence', None)))
        h, w = region.shape[:2]
        if self.tile_detector is not None and max(h, w) >= self.min_tile_region:
            for x1, y1, x2, y2 in windows(region.shape, max(256, min(self.tile_size, 2048))):
                if not self._within_budget():
                    break
                self.calls += 1
                self.last_calls += 1
                for raw in self.tile_detector.boxes(region[y1:y2, x1:x2]):
                    proposals.append(((raw[0] + x1, raw[1] + y1, raw[2] + x1, raw[3] + y1), 'tile', None))
        if self.text_detector is not None and self._within_budget():
            self.calls += 1
            for polygon in self.text_detector.polygons(region):
                xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
                proposals.append(((min(xs), min(ys), max(xs), max(ys)), 'text', None))
        ranked = rank(proposals, region.shape, top_k=self.top_k)
        self.last_alternatives = tuple(ranked)
        if not ranked:
            self.last_confidence = None
            return None
        best = ranked[0]
        self.last_confidence = best.detector_confidence
        return with_margin(tuple(round(v) for v in best.box), region.shape)
