"""Opt-in row-by-row reading of two-line plates."""
import unittest

import numpy as np

from sentinel_edge.ctc_fusion import greedy_text
from sentinel_edge.plate_options import PlateOptions
from sentinel_edge.two_line import join_rows, row_aware_probabilities, two_line_split

TOKENS = ('', '1', 'A')


def plate(height, width, bands):
    """White plate with dark text bands at the given row ranges."""
    image = np.full((height, width, 3), 255, np.uint8)
    for top, bottom in bands:
        image[top:bottom, 6:width - 6] = 0
    return image


class RecordingOCR:
    tokens = TOKENS

    def __init__(self):
        self.shapes = []

    def probabilities(self, image):
        self.shapes.append(image.shape[:2])
        return [[0., 1., 0.], [1., 0., 0.]]  # Reads "1".


class TwoLineTests(unittest.TestCase):
    def test_a_wide_single_line_plate_is_never_split(self):
        image = plate(24, 110, [(6, 18)])
        self.assertIsNone(two_line_split(image))
        ocr = RecordingOCR()
        self.assertEqual(row_aware_probabilities(ocr, image), [[0., 1., 0.], [1., 0., 0.]])
        self.assertEqual(ocr.shapes, [(24, 110)])

    def test_a_tall_box_without_an_ink_gap_is_not_split(self):
        self.assertIsNone(two_line_split(plate(40, 70, [(4, 36)])))

    def test_a_two_line_plate_is_read_row_by_row_top_first(self):
        image = plate(40, 70, [(4, 16), (24, 36)])
        split = two_line_split(image)
        self.assertIsNotNone(split)
        self.assertTrue(16 <= round(40 * split) <= 24)
        ocr = RecordingOCR()
        rows = row_aware_probabilities(ocr, image)
        self.assertEqual(len(ocr.shapes), 2)
        self.assertEqual(sum(shape[0] for shape in ocr.shapes), 40)
        self.assertEqual(greedy_text(rows, TOKENS), '11')

    def test_a_repeated_character_at_the_join_is_kept(self):
        one = [[0., 1., 0.]]
        self.assertEqual(greedy_text(one + one, TOKENS), '1')  # Plain concatenation would collapse it.
        self.assertEqual(greedy_text(join_rows(one, one), TOKENS), '11')

    def test_the_setting_is_off_by_default(self):
        self.assertFalse(PlateOptions().two_line)
        self.assertFalse(PlateOptions.from_env({}).two_line)
        self.assertFalse(PlateOptions.from_env({'SENTINEL_TWO_LINE_PLATES': 'yes'}).two_line)
        self.assertTrue(PlateOptions.from_env({'SENTINEL_TWO_LINE_PLATES': 'true'}).two_line)


if __name__ == '__main__':
    unittest.main()
