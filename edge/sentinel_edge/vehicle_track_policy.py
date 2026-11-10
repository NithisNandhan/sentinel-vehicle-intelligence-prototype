"""PTS-based qualification of supplied native vehicle tracks.

Qualification describes repeated detector evidence, not physical identity. Only
observed boxes are returned; this module neither tracks nor interpolates images.
"""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import math
from numbers import Integral, Real

from .vision import track_consistent


class VehicleTrackPolicy:
    """Keep unique passage IDs and at most ten scores per active native ID.

    Three distinct source timestamps spanning at least 150 ms and two scores
    >= .4 within the last ten observations qualify a passage. Qualification is
    sticky until passage end.
    ``hits`` is lifetime observations; ``high_confidence_hits`` is the rolling
    ten-observation count. ``records`` retains one summary per passage, with no
    unbounded box trail. All returned records are detached JSON-compatible data.
    ``vehicle_class`` smooths the last ten confidence-weighted class votes;
    ``observed_class`` is the current detector label. Class flicker alone never
    splits a native track, and neither label nor continuity proves identity.
    """

    MOTOR_VEHICLE_CLASSES = frozenset((2, 3, 5, 7))
    MAX_ABSENCE_MS = 1000

    def __init__(self, *, min_confirmation_span_ms=150):
        if not self._finite_number(min_confirmation_span_ms) or min_confirmation_span_ms < 0:
            raise ValueError('Minimum confirmation span must be finite and nonnegative')
        self.min_confirmation_span_ms = min_confirmation_span_ms
        self._active: dict[str, dict] = {}
        self._records: list[dict] = []
        self._scores: dict[str, deque] = {}
        self._class_votes: dict[str, deque] = {}
        self._last_pts_ms: Real | None = None
        self._finished = False
        self._counters: Counter = Counter()

    @property
    def records(self) -> list[dict]:
        return deepcopy(self._records)

    @property
    def counters(self) -> dict[str, int]:
        return dict(self._counters)

    @staticmethod
    def _finite_number(value) -> bool:
        if isinstance(value, bool) or not isinstance(value, Real):
            return False
        try:
            return math.isfinite(value)
        except (OverflowError, TypeError, ValueError):
            return False

    def _check_timestamp(self, pts_ms, *, allow_equal=False):
        valid = self._finite_number(pts_ms) and pts_ms >= 0
        if valid and self._last_pts_ms is not None:
            valid = pts_ms >= self._last_pts_ms if allow_equal else pts_ms > self._last_pts_ms
        if not valid:
            self._counters['rejected_timestamp'] += 1
            raise ValueError('Source PTS must be finite, nonnegative and forward-moving')
        return int(pts_ms) if isinstance(pts_ms, Integral) else float(pts_ms)

    def _reject_reason(self, detection):
        if not isinstance(detection, dict):
            return 'invalid_detection'
        native_id = detection.get('native_id')
        if not isinstance(native_id, str) or not native_id.strip():
            return 'invalid_native_id'
        vehicle_class = detection.get('vehicle_class')
        if isinstance(vehicle_class, bool) or not isinstance(vehicle_class, Integral):
            return 'invalid_class'
        if vehicle_class not in self.MOTOR_VEHICLE_CLASSES:
            return 'unsupported_class'
        box = detection.get('box')
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or not all(self._finite_number(value) for value in box)):
            return 'invalid_box'
        box = [float(value) for value in box]
        if box[2] <= box[0] or box[3] <= box[1]:
            return 'invalid_box'
        # Guard arithmetic in the reused discontinuity check as well as inputs.
        if not self._finite_number((box[2] - box[0]) * (box[3] - box[1])):
            return 'invalid_box'
        confidence = detection.get('confidence')
        if not self._finite_number(confidence) or not 0 <= confidence <= 1:
            return 'invalid_confidence'
        return None

    def _end(self, native_id: str, pts_ms, reason: str):
        record = self._active.pop(native_id)
        record.update(ended=True, end_pts_ms=pts_ms, end_reason=reason)
        self._scores.pop(native_id)
        self._class_votes.pop(native_id)
        self._counters[f'ended_{reason}'] += 1

    def update(self, detections, pts_ms) -> list[dict]:
        """Consume one strictly increasing source PTS and native detections.

        Invalid detections are counted and omitted. All occurrences of a
        duplicated native ID are omitted, so none can supply extra votes.
        Invalid/nonforward PTS raises ValueError without modifying bindings.
        """
        if self._finished:
            raise ValueError('Vehicle track policy is already finished')
        pts_ms = self._check_timestamp(pts_ms)
        detections = list(detections)
        self._last_pts_ms = pts_ms
        self._counters['source_timestamps'] += 1
        for native_id, record in list(self._active.items()):
            if pts_ms - record['last_pts_ms'] > self.MAX_ABSENCE_MS:
                self._end(native_id, pts_ms, 'gap')

        native_counts = Counter(
            detection['native_id'] for detection in detections
            if isinstance(detection, dict) and isinstance(detection.get('native_id'), str)
        )
        visible = []
        for detection in detections:
            if (isinstance(detection, dict)
                    and isinstance(detection.get('native_id'), str)
                    and native_counts[detection['native_id']] > 1):
                self._counters['rejected_duplicate_native_id'] += 1
                continue
            reason = self._reject_reason(detection)
            if reason:
                self._counters[f'rejected_{reason}'] += 1
                continue

            native_id = detection['native_id']
            box = [float(value) for value in detection['box']]
            vehicle_class = int(detection['vehicle_class'])
            confidence = float(detection['confidence'])
            record = self._active.get(native_id)
            start_reason = 'new_native_id'
            if record is not None and not track_consistent(record['box'], box):
                start_reason = 'discontinuity'
                self._end(native_id, pts_ms, start_reason)
                record = None
            if record is None:
                record = {
                    'passage_id': f'V{len(self._records) + 1:03d}',
                    'native_id': native_id, 'box': box, 'vehicle_class': vehicle_class,
                    'observed_class': vehicle_class, 'class_ambiguous': True, 'class_vote_share': 0.,
                    'confidence': confidence, 'status': 'tentative',
                    'first_pts_ms': pts_ms, 'last_pts_ms': pts_ms,
                    'hits': 0, 'high_confidence_hits': 0, 'high_confidence_hits_total': 0,
                    'reason': 'insufficient_source_timestamps', 'start_reason': start_reason,
                    'qualified_pts_ms': None, 'ended': False,
                    'end_pts_ms': None, 'end_reason': None,
                }
                self._records.append(record)
                self._active[native_id] = record
                self._scores[native_id] = deque(maxlen=10)
                self._class_votes[native_id] = deque(maxlen=10)
                self._counters['passages_started'] += 1

            scores = self._scores[native_id]
            scores.append(confidence >= .4)
            class_votes = self._class_votes[native_id]
            class_votes.append((vehicle_class, confidence))
            weights = {}
            for voted_class, weight in class_votes:
                weights[voted_class] = weights.get(voted_class, 0.) + weight
            total_weight = sum(weights.values())
            largest_weight = max(weights.values())
            leaders = [label for label, weight in weights.items()
                       if math.isclose(weight, largest_weight, rel_tol=1e-12, abs_tol=1e-12)]
            # Exact/numerical ties retain the prior stable label. With no
            # confidence weight there is no supported class label to switch to.
            if total_weight > 0 and len(leaders) == 1:
                record['vehicle_class'] = leaders[0]
            class_vote_share = largest_weight / total_weight if total_weight > 0 else 0.
            record.update(box=box, confidence=confidence, last_pts_ms=pts_ms,
                          observed_class=vehicle_class, class_vote_share=class_vote_share,
                          class_ambiguous=(class_vote_share < .6 and
                                           not math.isclose(class_vote_share, .6, rel_tol=1e-12)),
                          hits=record['hits'] + 1, high_confidence_hits=sum(scores),
                          high_confidence_hits_total=record['high_confidence_hits_total'] + int(confidence >= .4))
            if record['status'] != 'qualified':
                if (record['hits'] >= 3 and record['high_confidence_hits'] >= 2
                        and pts_ms - record['first_pts_ms'] >= self.min_confirmation_span_ms):
                    record.update(status='qualified', reason='qualification_gate_met', qualified_pts_ms=pts_ms)
                    self._counters['passages_qualified'] += 1
                elif record['hits'] < 3:
                    record['reason'] = 'insufficient_source_timestamps'
                elif pts_ms - record['first_pts_ms'] < self.min_confirmation_span_ms:
                    record['reason'] = 'insufficient_confirmation_span'
                else:
                    record['reason'] = 'insufficient_recent_high_confidence'
            self._counters['accepted_observations'] += 1
            visible.append(deepcopy(record))
        return visible

    def finish(self, pts_ms) -> list[dict]:
        """Close active passages at EOF and return all passage summaries.

        EOF may share the final frame's PTS. Repeated finish calls are harmless
        when their timestamp remains valid; updates after EOF are rejected.
        """
        pts_ms = self._check_timestamp(pts_ms, allow_equal=True)
        self._last_pts_ms = pts_ms
        for native_id in list(self._active):
            self._end(native_id, pts_ms, 'eof')
        self._finished = True
        return self.records
