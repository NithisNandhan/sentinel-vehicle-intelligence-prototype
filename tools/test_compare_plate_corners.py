"""Small scoring/geometry regressions; no models, network or private labels."""
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_plate_corners import evaluate_reference, native_crop, overlap


def target(box=None, time=1000):
    return dict(number=1, position_ms=time, box=box or [110, 220, 150, 240], actual='AB12CD3456')


def proposal(box=None, text='AB12CD3456', confidence=.9, **extra):
    return dict(box=box or [10, 20, 50, 40], native_reading=text,
                confidence=confidence, **extra)


def row(predictions=(), box=None, time=1000, identifier='auto-region'):
    return dict(id=identifier, position_ms=time,
                region_box=box or [100, 200, 200, 300], predictions=list(predictions))


class PlateCornerComparisonTests(unittest.TestCase):
    def test_overlap_reports_iou_and_target_coverage_separately(self):
        self.assertEqual(overlap([0, 0, 20, 20], [5, 5, 15, 15]), (.25, 1.))
        self.assertEqual(overlap([0, 0, 1, 1], [2, 2, 3, 3]), (0., 0.))
        self.assertEqual(overlap([0, 0, 10, 10], [0, 0, 10, 10]), (1., 1.))

    def test_local_box_maps_with_native_vehicle_offset(self):
        result = evaluate_reference(target(), [row([proposal()])])
        self.assertTrue(result['roi_available'])
        self.assertTrue(result['found'])
        self.assertEqual(result['matched_candidates'], 1)
        self.assertAlmostEqual(result['iou'], 1.)
        self.assertAlmostEqual(result['reference_box_coverage'], 1.)
        self.assertTrue(result['native_exact'])
        self.assertEqual(result['region_id'], 'auto-region')

    def test_selection_uses_geometry_not_correct_text_or_higher_confidence(self):
        correct_lower_overlap = proposal([10, 20, 40, 40], confidence=.99)
        wrong_better_overlap = proposal(text='AB12CD3458', confidence=.4)
        result = evaluate_reference(target(), [row([correct_lower_overlap, wrong_better_overlap])])
        self.assertTrue(result['found'])
        self.assertEqual(result['matched_candidates'], 2)
        self.assertIs(result['crop'], wrong_better_overlap)
        self.assertFalse(result['native_exact'])
        self.assertFalse(result['either_exact'])

    def test_equal_geometry_uses_confidence_not_ocr_for_tie_break(self):
        correct = proposal(confidence=.5)
        wrong = proposal(text='AB12CD3458', confidence=.8)
        result = evaluate_reference(target(), [row([correct, wrong])])
        self.assertIs(result['crop'], wrong)
        self.assertFalse(result['native_exact'])

    def test_other_timestamps_never_supply_reference_match(self):
        result = evaluate_reference(target(), [row([proposal()], time=1001)])
        self.assertFalse(result['roi_available'])
        self.assertFalse(result['found'])
        self.assertEqual(result['matched_candidates'], 0)
        self.assertFalse(result['either_exact'])

    def test_missing_roi_differs_from_detector_miss_and_partial_roi(self):
        missing = evaluate_reference(target(), [])
        missed = evaluate_reference(target(), [row()])
        partial = evaluate_reference(target(), [row(box=[100, 200, 130, 300])])
        self.assertFalse(missing['roi_available'])
        self.assertTrue(missed['roi_available'])
        self.assertFalse(partial['roi_available'])
        for result in (missing, missed, partial):
            self.assertFalse(result['found'])
            self.assertFalse(result['native_exact'])
            self.assertFalse(result['rectified_exact'])

    def test_rectified_exact_and_either_view_oracle_remain_distinct(self):
        p = proposal(text='AB12CD3458', rectified_reading='AB12CD3456')
        result = evaluate_reference(target(), [row([p])])
        self.assertFalse(result['native_exact'])
        self.assertTrue(result['rectified_exact'])
        self.assertTrue(result['either_exact'])
        # Absence of a rectified output must never masquerade as a successful read.
        result = evaluate_reference(target(), [row([proposal()])])
        self.assertTrue(result['native_exact'])
        self.assertFalse(result['rectified_exact'])

    def test_native_crop_clips_to_image_floor_ceil_and_copies_pixels(self):
        image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        original = image.copy()
        crop, origin = native_crop(image, [-1.2, 1.4, 4.1, 9.2])
        self.assertEqual(origin, (0, 1))
        np.testing.assert_array_equal(crop, image[1:6, 0:5])
        crop[:] = 0
        np.testing.assert_array_equal(image, original)
        with self.assertRaises(ValueError):
            native_crop(image, [9, 0, 10, 1])


if __name__ == '__main__':
    unittest.main()
