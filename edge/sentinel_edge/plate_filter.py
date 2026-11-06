"""Opt-in screen for plate proposals that are not plates (off by default).

The plate detector sometimes boxes blank bus roof panels, bars or grilles. The OCR
then still emits text, and occasionally a well-formed plate string. Such a proposal is
dropped before it becomes fusion evidence when its uncalibrated detector score is very
low, its shape cannot be a plate, or the OCR is not confident about a single character.

The limits were chosen on 50 AI-drafted labels from one clip (docs/WORK_LOG.md,
Phase 3); they are a starting point, not calibrated values. Fusion and acceptance
thresholds are unchanged.
"""

MIN_DETECTOR_CONFIDENCE = .35   # Lowest labelled plate scored .44; three non-plates scored .26-.40.
CONFIDENT_PEAK = .6             # Same per-character peak that acceptance requires in two frames.
MIN_CONFIDENT_CHARACTERS = 1    # 12 of 14 labelled non-plates had none.
ASPECT_RANGE = (.8, 8.)         # Width/height: two-line plates ~1.5, single-line ~3-5.


def confident_characters(rows, peak=CONFIDENT_PEAK):
    """Greedy CTC characters (blank = index 0) whose peak probability reaches `peak`."""
    count, previous = 0, None
    for row in rows:
        index = max(range(len(row)), key=row.__getitem__)
        if index and index != previous and row[index] >= peak:
            count += 1
        previous = index
    return count


class NonPlateFilter:
    def __init__(self, *, min_detector_confidence=MIN_DETECTOR_CONFIDENCE,
                 min_confident_characters=MIN_CONFIDENT_CHARACTERS, aspect_range=ASPECT_RANGE):
        self.min_detector_confidence = min_detector_confidence
        self.min_confident_characters = min_confident_characters
        self.aspect_range = aspect_range

    def reason(self, crop, rows, detector_confidence=None):
        """Why this proposal is not plate evidence, or None to keep it.

        An unknown detector score (adapters that report none) is never held against it.
        """
        height, width = crop.shape[:2]
        if not height or not self.aspect_range[0] <= width / height <= self.aspect_range[1]:
            return 'Proposal shape cannot be a plate'
        if detector_confidence is not None and detector_confidence < self.min_detector_confidence:
            return 'Plate detector score too low for plate evidence'
        if confident_characters(rows) < self.min_confident_characters:
            return 'OCR found no confident character; likely not a plate'
        return None
