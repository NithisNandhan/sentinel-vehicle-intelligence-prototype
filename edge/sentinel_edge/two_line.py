"""Opt-in row-by-row reading of two-line plates (off by default).

The OCR network reads a single text line: every crop is squeezed into one 48 px row,
so a two-line plate's rows are mixed. A crop is treated as two-line only when its box
is tall for its width AND an ink gap separates two inked rows. Each row is then read
on its own and the rows are joined top then bottom. Nothing here changes fusion or
acceptance: the joined rows are ordinary CTC evidence for the same reader.
"""
from .hybrid import layout_hypotheses

# Single-line Indian plates are about 4.5:1 (height/width ~0.22); two-line about 1.7:1.
TWO_LINE_MIN_HEIGHT_RATIO = .45


def two_line_split(crop):
    """Fraction of the crop height where the rows separate, or None for one line."""
    height, width = crop.shape[:2]
    if not width or height / width < TWO_LINE_MIN_HEIGHT_RATIO:
        return None
    return next((split for layout, _, split in layout_hypotheses(crop) if layout == 'two-row'), None)


def join_rows(top, bottom):
    """Top then bottom, with one certain blank step so a repeated character at the
    join (`...1` + `1...`) is not collapsed into one by CTC decoding."""
    blank = [1.] + [0.] * (len(top[0]) - 1)
    return [*top, blank, *bottom]


def row_aware_probabilities(ocr, crop):
    """CTC rows for the crop: rows read separately when it is a two-line plate."""
    split = two_line_split(crop)
    if split is None:
        return ocr.probabilities(crop)
    boundary = round(crop.shape[0] * split)
    return join_rows(ocr.probabilities(crop[:boundary]), ocr.probabilities(crop[boundary:]))
