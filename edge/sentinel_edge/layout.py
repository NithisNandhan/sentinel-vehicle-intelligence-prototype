"""One-row / two-row plate layout with an explicit 'uncertain' outcome.

Reuses the native-coordinate row split from `hybrid.layout_hypotheses`. Nothing
here resizes or squashes the crop: row crops are original pixels. The decision
is a geometry heuristic, not a learned classifier, so a passage whose frames
disagree (some one-row, some two-row) is a `conflict` and cannot be confirmed
automatically (result_state.py); it goes to review instead.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

ONE_ROW, TWO_ROW, UNCERTAIN = 'one_row', 'two_row', 'uncertain'
# Passage-level outcomes: frames disagree, or no frame was decisive.
CONFLICT, UNDETERMINED = 'conflict', 'undetermined'
# Indian single-row plates are about 4.5:1, two-row about 1.7:1 (width/height).
CLEAR_ONE_ROW_ASPECT = 3.0
CLEAR_TWO_ROW_ASPECT = 2.3


@dataclass(frozen=True)
class LayoutDecision:
    layout: str
    aspect: float
    split: float | None = None   # Fraction of crop height where rows separate.

    def rows(self, crop):
        """Native-pixel rows, top first. One element for a one-row plate."""
        if self.layout != TWO_ROW or self.split is None:
            return [crop]
        boundary = round(crop.shape[0] * self.split)
        if not 0 < boundary < crop.shape[0]:
            return [crop]
        return [crop[:boundary], crop[boundary:]]


def classify(crop) -> LayoutDecision:
    height, width = crop.shape[:2]
    if not height or not width:
        return LayoutDecision(UNCERTAIN, 0.)
    aspect = width / height
    split = None
    if aspect <= 3.5:
        from .hybrid import layout_hypotheses
        split = next((s for layout, _, s in layout_hypotheses(crop) if layout == 'two-row'), None)
    if aspect >= CLEAR_ONE_ROW_ASPECT and split is None:
        return LayoutDecision(ONE_ROW, round(aspect, 3))
    if aspect <= CLEAR_TWO_ROW_ASPECT and split is not None:
        return LayoutDecision(TWO_ROW, round(aspect, 3), split)
    # Tall without a clean ink gap, or wide with an apparent gap: do not guess.
    return LayoutDecision(UNCERTAIN, round(aspect, 3), split)


def passage_layout(decisions) -> str:
    """Combine per-frame layouts.

    `conflict`: some frames were clearly one-row and others clearly two-row; the
    reading cannot be confirmed automatically. `undetermined`: no frame was
    decisive (common for small crops); this alone does not block confirmation.
    """
    counts = Counter(decisions)
    certain = {key for key in (ONE_ROW, TWO_ROW) if counts[key]}
    if len(certain) == 2:
        return CONFLICT
    if certain:
        return certain.pop()
    return UNDETERMINED if counts else 'unknown'


def join_row_probabilities(top, bottom):
    """Top row then bottom row, separated by one certain CTC blank (see two_line.py)."""
    from .two_line import join_rows
    return join_rows(top, bottom)
