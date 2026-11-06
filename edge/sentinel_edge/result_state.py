"""Three recognition outcomes for one vehicle passage: confirmed, partial, unreadable.

CONFIRMED  The unchanged multi-frame gate (`CTCPlateReader.accepted`) passed AND the
           reading is a verbatim, strictly valid Indian registration AND no extra
           blocker applies (layout conflict, too few native pixels, secondary-reader
           veto). The existing gate is the safety floor; this module only adds blockers.
PARTIAL    Some character positions are supported by at least two distinct frames and
           no competing hypothesis disagrees at that position; the others are `?`.
           The plate length must be agreed. A partial result can never raise a
           confirmed alert (backend enforces this too). A complete reading blocked by
           one of the extra blockers is also PARTIAL, with `review_required`.
UNREADABLE The vehicle passage exists but the evidence supports no useful reading; the
           reason is kept.

Format rules may reject a reading or a pattern; they never choose or change a
character. Scores are uncalibrated model peaks, not probabilities of correctness.
Full plate text and patterns stay in edge memory; only masked forms are sent centrally.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .layout import CONFLICT, TWO_ROW
from .plate_format import UNKNOWN, pattern_compatible, strict_status
from .privacy import mask_plate

CONFIRMED, PARTIAL, UNREADABLE = 'confirmed', 'partial', 'unreadable'
STATES = (CONFIRMED, PARTIAL, UNREADABLE)

# Reasons a passage is unreadable or a complete reading was held back for review.
REASONS = frozenset({
    'no_plate_found', 'plate_too_small', 'non_plate_rejected', 'insufficient_observations',
    'no_complete_candidate', 'length_uncertain', 'insufficient_known_characters',
    'format_impossible', 'uncertain_characters',
    # Blockers on an otherwise accepted reading (sent to review as PARTIAL):
    'format_lookalike', 'format_ambiguous', 'format_invalid', 'format_adjusted',
    'layout_conflict', 'insufficient_resolution', 'secondary_reader_veto', 'length_disagreement',
})


@dataclass(frozen=True)
class ResultPolicy:
    """Versioned result-state policy. Values are conservative starting points, not calibrated."""
    version: str = 'result-policy-1'
    min_support_frames: int = 2       # Same as accepted(): >= 2 frames with a peak >= .6.
    support_peak: float = .6
    min_mean_peak: float = .75        # Same as accepted()'s minimum character score.
    competitor_margin: float = 2.0    # Log-likelihood window for hypotheses that still compete.
    min_known: int = 5                # Fewer known characters is not a useful partial reading.
    max_unknown: int = 3
    require_strict_format: bool = True
    layout_gate: bool = True
    min_confirm_height_one_row: int = 16
    min_confirm_height_two_row: int = 28

    def as_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass(frozen=True, repr=False)
class PositionEvidence:
    index: int
    predicted: str
    known: bool
    support_frames: int
    mean_peak: float
    alternatives: tuple = ()

    def public(self, *, reveal=False):
        """Reviewer view. Without `reveal` the characters are withheld."""
        return {'index': self.index, 'known': self.known, 'support_frames': self.support_frames,
                'mean_peak': round(self.mean_peak, 3),
                **({'predicted': self.predicted, 'alternatives': list(self.alternatives)} if reveal else {})}


@dataclass(frozen=True, repr=False)
class RecognitionResult:
    state: str
    reason: str | None
    observations: int
    plate: str | None = field(default=None, repr=False)      # CONFIRMED only. Edge memory only.
    pattern: str | None = field(default=None, repr=False)    # PARTIAL only, with '?'. Edge-local.
    length: int | None = None
    unknown_positions: tuple = ()
    positions: tuple = ()
    review_required: bool = False
    format_status: str | None = None
    layout: str = 'unknown'
    score: float | None = None
    length_votes: dict = field(default_factory=dict)
    # False when a minority of frames read a different length: shown for review,
    # but never used for partial watchlist matching.
    length_certain: bool = True
    policy_version: str = ResultPolicy.version

    def __post_init__(self):
        if self.state not in STATES:
            raise ValueError('Unknown result state')
        if self.state == CONFIRMED and (not self.plate or self.reason or self.unknown_positions or self.review_required):
            raise ValueError('A confirmed result is a complete, unblocked reading')
        if self.state == PARTIAL and (not self.pattern or self.plate or self.length != len(self.pattern)):
            raise ValueError('A partial result carries a pattern of agreed length and no full plate')
        if self.state == UNREADABLE and (self.plate or self.pattern or not self.reason):
            raise ValueError('An unreadable result carries a reason and no text')
        if self.reason is not None and self.reason not in REASONS:
            raise ValueError('Unknown result reason')

    @property
    def known_characters(self):
        if self.state == CONFIRMED:
            return len(self.plate)
        if self.state == PARTIAL:
            return sum(c != UNKNOWN for c in self.pattern)
        return 0

    def masked(self):
        """What may leave the edge: first two and last two positions, '?' kept."""
        if self.state == CONFIRMED:
            return mask_plate(self.plate)
        if self.state == PARTIAL:
            text = self.pattern
            if len(text) < 5:
                return '*' * len(text)
            return text[:2] + '*' * (len(text) - 4) + text[-2:]
        return None


def _unreadable(reason, observations, *, layout='unknown', policy=ResultPolicy(), length_votes=None, score=None):
    return RecognitionResult(UNREADABLE, reason, observations, layout=layout, policy_version=policy.version,
                             length_votes=length_votes or {}, score=score)


def analyse_positions(reader, best, competitors, policy, *, disputed=()):
    """Per-position evidence for `best` against every retained frame and competing hypothesis."""
    peaks = reader.character_evidence(best)
    weights = reader.frame_weights()
    readings = reader.frame_readings()
    total = sum(weights) or 1.
    result = []
    for index, character in enumerate(best):
        aligned = [values[index] if len(values) == len(best) else 0. for values in peaks]
        support = sum(value >= policy.support_peak for value in aligned)
        mean = sum(weight * value for weight, value in zip(weights, aligned)) / total
        competing = {c.plate[index] for c in competitors if len(c.plate) == len(best)} - {character}
        disagreeing = {reading[index][0] for reading in readings
                       if len(reading) == len(best) and reading[index][0] != character
                       and reading[index][1] >= policy.support_peak}
        known = (support >= policy.min_support_frames and mean >= policy.min_mean_peak
                 and not competing and not disagreeing and index not in disputed)
        result.append(PositionEvidence(index, character, known, support, mean,
                                       tuple(sorted(competing | disagreeing))))
    return tuple(result)


def frame_lengths(reader, policy):
    """Plate length as read directly by each frame (greedy CTC), ignoring weak frames.

    Returns (agreed_length or None, unanimous, votes). CTC fusion can rank a
    candidate that *drops* or *adds* a character above the length every frame
    actually read; this independent check catches inserted/deleted characters.
    """
    votes = Counter()
    for reading in reader.frame_readings():
        if reading and sum(peak for _, peak in reading) / len(reading) >= policy.support_peak:
            votes[len(reading)] += 1
    if not votes:
        return None, False, {}
    (length, count), *rest = votes.most_common()
    others = sum(votes.values()) - count
    agreed = length if count >= policy.min_support_frames and count > others else None
    return agreed, agreed is not None and not others, {str(k): v for k, v in sorted(votes.items())}


def classify(reader, *, threshold=.85, policy=ResultPolicy(), layout='unknown', max_plate_height=None,
             veto_positions=None, empty_reason='no_plate_found'):
    """Final outcome for one passage's fused evidence. Never invents a character."""
    observations = reader.observation_count if reader is not None else 0
    if not observations:
        return _unreadable(empty_reason, 0, layout=layout, policy=policy)
    candidates = reader.candidates()
    accepted = reader.accepted(threshold) if observations >= 3 else None
    agreed, unanimous, votes = frame_lengths(reader, policy)
    blockers = []
    if accepted is not None:
        status = strict_status(accepted.plate)
        if policy.require_strict_format and status != 'valid':
            blockers.append('format_' + status)
        if accepted.format_adjusted:
            blockers.append('format_adjusted')
        if not unanimous or agreed != len(accepted.plate):
            blockers.append('length_disagreement')
        if policy.layout_gate and layout == CONFLICT:
            blockers.append('layout_conflict')
        minimum = policy.min_confirm_height_two_row if layout == TWO_ROW else policy.min_confirm_height_one_row
        if max_plate_height is not None and max_plate_height < minimum:
            blockers.append('insufficient_resolution')
        if veto_positions:
            blockers.append('secondary_reader_veto')
        if not blockers:
            return RecognitionResult(CONFIRMED, None, observations, plate=accepted.plate, length=len(accepted.plate),
                                     format_status=status, layout=layout, score=round(accepted.score, 4),
                                     length_votes=votes, policy_version=policy.version)
    if observations < policy.min_support_frames:
        return _unreadable('insufficient_observations', observations, layout=layout, policy=policy)
    if agreed is None:
        return _unreadable('length_uncertain', observations, layout=layout, policy=policy, length_votes=votes)
    # Only hypotheses of the length the frames actually read can describe the plate.
    same_length = [c for c in candidates if len(c.plate) == agreed]
    if not same_length:
        return _unreadable('no_complete_candidate', observations, layout=layout, policy=policy, length_votes=votes)
    best = accepted if accepted is not None and len(accepted.plate) == agreed else same_length[0]
    competitors = [c for c in same_length if c.log_score >= best.log_score - policy.competitor_margin]
    positions = analyse_positions(reader, best.plate, competitors, policy, disputed=frozenset(veto_positions or ()))
    unknown = tuple(p.index for p in positions if not p.known)
    known = len(best.plate) - len(unknown)
    if known < policy.min_known or len(unknown) > policy.max_unknown:
        return _unreadable('insufficient_known_characters', observations, layout=layout, policy=policy,
                           length_votes=votes, score=round(best.score, 4))
    pattern = ''.join(UNKNOWN if not p.known else p.predicted for p in positions)
    if not pattern_compatible(pattern):
        return _unreadable('format_impossible', observations, layout=layout, policy=policy, length_votes=votes)
    if not unanimous and 'length_disagreement' not in blockers:
        blockers.append('length_disagreement')
    reason = (blockers[0] if blockers else
              'insufficient_observations' if observations < 3 and not unknown else 'uncertain_characters')
    return RecognitionResult(PARTIAL, reason, observations, pattern=pattern, length=len(pattern),
                             unknown_positions=unknown, positions=positions,
                             review_required=bool(blockers) or not unknown, length_certain=unanimous,
                             format_status=strict_status(best.plate), layout=layout,
                             score=round(best.score, 4), length_votes=votes, policy_version=policy.version)
