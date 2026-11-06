"""Veto-only secondary OCR reader (e.g. SVTRv2-B). Disabled by default.

The secondary reader can only *block* an automatic confirmation; it never
supplies, replaces or votes characters. A veto needs a position-specific
disagreement that the secondary reader makes with a high peak in at least
`min_frames` distinct crops of the same length. Vetoed readings go to human
review with the disputed positions unknown.

Earlier external evaluation (docs/SVTRV2_COMPARISON_2026-10-31.md) found the
secondary reader wrong on 36 of the primary's 137 correct reads, so agreement
is not required and disagreement alone is not trusted: only confident,
repeated, aligned disagreement counts. Keep it off until `evaluate()` on
development data shows an acceptable blocked-correct cost.
"""
from __future__ import annotations

from collections import Counter

from .ctc_fusion import greedy_reading


class SecondaryVeto:
    def __init__(self, reader, *, threshold=.9, min_frames=2):
        if not .5 <= threshold <= 1 or not 1 <= min_frames <= 8:
            raise ValueError('Invalid veto bounds')
        self.reader, self.threshold, self.min_frames = reader, threshold, min_frames
        self.metrics = Counter()

    def readings(self, crops):
        return [greedy_reading(self.reader.probabilities(crop), self.reader.tokens) for crop in crops]

    def disputed_positions(self, primary, crops):
        """Positions the secondary reader confidently and repeatedly reads differently."""
        result = disputed(primary, self.readings(crops), threshold=self.threshold, min_frames=self.min_frames)
        self.metrics['vetoes' if result else 'passes'] += 1
        return result


def disputed(primary, readings, *, threshold=.9, min_frames=2):
    votes = Counter()
    for reading in readings:
        if len(reading) != len(primary):
            continue  # Different length: no positional alignment, so no veto evidence.
        for index, (character, peak) in enumerate(reading):
            if character != primary[index] and peak >= threshold:
                votes[index] += 1
    return frozenset(index for index, count in votes.items() if count >= min_frames) or None


def evaluate(records, *, threshold=.9, min_frames=2):
    """Measure a veto policy on labelled development records.

    Each record: {'truth': str, 'primary': str or None (the reading the primary
    gate would confirm), 'secondary': [[(char, peak), ...] per crop]}.
    Reports blocked correct readings (false vetoes), blocked wrong readings
    (useful vetoes) and the coverage cost, with denominators.
    """
    confirmed = correct = wrong = blocked_correct = blocked_wrong = 0
    for record in records:
        primary = record.get('primary')
        if not primary:
            continue
        confirmed += 1
        is_correct = primary == record['truth']
        correct += is_correct
        wrong += not is_correct
        vetoed = disputed(primary, [tuple(map(tuple, r)) for r in record.get('secondary', [])],
                          threshold=threshold, min_frames=min_frames) is not None
        blocked_correct += vetoed and is_correct
        blocked_wrong += vetoed and not is_correct
    return {
        'threshold': threshold, 'min_frames': min_frames, 'primary_confirmed': confirmed,
        'primary_correct': correct, 'primary_wrong': wrong,
        'false_vetoes_blocked_correct': blocked_correct, 'useful_vetoes_blocked_wrong': blocked_wrong,
        'coverage_cost': {'count': blocked_correct, 'of': correct, 'rate': round(blocked_correct / correct, 4) if correct else None},
        'wrong_caught': {'count': blocked_wrong, 'of': wrong, 'rate': round(blocked_wrong / wrong, 4) if wrong else None},
        'recommendation': ('insufficient data' if confirmed < 30 else
                           'keep disabled' if blocked_wrong == 0 or blocked_correct > blocked_wrong else
                           'candidate for enabling after sealed-set check'),
    }
