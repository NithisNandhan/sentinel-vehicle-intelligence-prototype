"""Provisional confidence calibration and risk-versus-coverage analysis (development data only).

The existing rule-based gate (`CTCPlateReader.accepted`: score >= 0.85, every
character >= 0.75, runner-up margin, >= 2 frames at >= 0.6) is the SAFETY FLOOR.
A calibrated configuration may only make acceptance stricter, never looser, and
is versioned. Scores are uncalibrated model peaks; nothing here yields a
correctness guarantee. A threshold that shows zero errors on n development
samples is bounded only by the rule of three, which is ~3/n.

Record format: {'score': float, 'min_char': float, 'correct': bool, 'recording': str}
for each reading that the unchanged gate accepted (wrong ones included).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .evaluation_stats import rule_of_three

FLOOR_SCORE = .85
FLOOR_MIN_CHARACTER = .75


@dataclass(frozen=True)
class ThresholdConfig:
    version: str
    score: float = FLOOR_SCORE
    min_character: float = FLOOR_MIN_CHARACTER
    source: str = 'safety-floor'

    def __post_init__(self):
        if self.score < FLOOR_SCORE or self.min_character < FLOOR_MIN_CHARACTER or self.score > 1 or self.min_character > 1:
            raise ValueError('Calibrated thresholds may only be stricter than the safety floor')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(**{key: data[key] for key in ('version', 'score', 'min_character', 'source') if key in data})

    def dump(self, path):
        Path(path).write_text(json.dumps(self.__dict__, indent=2), encoding='utf-8')

    def admits(self, score, min_char):
        return score >= self.score and min_char >= self.min_character


def risk_coverage(records, thresholds):
    """Coverage and error rate of the gate tightened to each score threshold (floor enforced)."""
    total = len(records)
    rows = []
    for threshold in sorted({max(t, FLOOR_SCORE) for t in thresholds}):
        kept = [r for r in records if r['score'] >= threshold]
        wrong = sum(not r['correct'] for r in kept)
        rows.append({'threshold': round(threshold, 4), 'accepted': len(kept), 'wrong': wrong,
                     'coverage': round(len(kept) / total, 4) if total else None,
                     'error_rate': round(wrong / len(kept), 4) if kept else None,
                     'zero_error_upper_95': round(rule_of_three(len(kept)), 4) if kept and not wrong else None})
    return rows


def expected_calibration_error(records, bins=10):
    """Mean |confidence - accuracy| over equal-width score bins, weighted by bin size."""
    if not records:
        return None
    total, error, detail = len(records), 0., []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [r for r in records if low <= r['score'] < high or (index == bins - 1 and r['score'] == 1)]
        if not members:
            continue
        confidence = sum(r['score'] for r in members) / len(members)
        accuracy = sum(r['correct'] for r in members) / len(members)
        error += len(members) / total * abs(confidence - accuracy)
        detail.append({'bin': [round(low, 2), round(high, 2)], 'n': len(members),
                       'mean_score': round(confidence, 4), 'accuracy': round(accuracy, 4)})
    return {'ece': round(error, 4), 'bins': detail, 'n': total,
            'note': 'Small bins are noisy; ECE on a handful of samples is indicative only.'}


def strictest_clean_threshold(records):
    """Lowest score threshold (>= floor) at which no accepted development reading is wrong, or None."""
    candidates = sorted({max(r['score'], FLOOR_SCORE) for r in records} | {FLOOR_SCORE})
    for threshold in candidates:
        kept = [r for r in records if r['score'] >= threshold]
        if kept and all(r['correct'] for r in kept):
            return threshold
    return None


def leave_one_recording_out(records):
    """Fit a clean threshold on all other recordings, apply it to the held-out one."""
    recordings = sorted({r['recording'] for r in records})
    if len(recordings) < 3:
        return {'status': 'not_run', 'reason': 'needs at least 3 labelled recordings', 'recordings': len(recordings)}
    folds = []
    for held in recordings:
        train = [r for r in records if r['recording'] != held]
        test = [r for r in records if r['recording'] == held]
        threshold = strictest_clean_threshold(train)
        kept = [r for r in test if threshold is not None and r['score'] >= threshold]
        folds.append({'held_out': held, 'fitted_threshold': threshold, 'accepted': len(kept),
                      'wrong': sum(not r['correct'] for r in kept), 'of': len(test)})
    return {'status': 'run', 'folds': folds, 'wrong_total': sum(f['wrong'] for f in folds),
            'accepted_total': sum(f['accepted'] for f in folds)}


def report(records, *, version, thresholds=(.85, .9, .95, .98, .99)):
    clean = strictest_clean_threshold(records)
    return {
        'calibration_version': version, 'provisional': True, 'samples': len(records),
        'recordings': len({r['recording'] for r in records}),
        'risk_coverage': risk_coverage(records, thresholds),
        'calibration_error': expected_calibration_error(records),
        'leave_one_recording_out': leave_one_recording_out(records),
        'suggested_stricter_threshold': clean,
        'caution': 'Provisional, development data only. This is not a 99.9% guarantee; the zero-error bound is ~3/n. '
                   'The unchanged rule-based gate remains the safety floor.',
    }
