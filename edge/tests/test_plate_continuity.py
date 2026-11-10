"""Geometry-only continuity tests; no OCR models, images or labels required."""
import unittest

from sentinel_edge.plate_continuity import PlateContinuityPolicy, continuity_reason


class PlateContinuityTests(unittest.TestCase):
    def test_first_observation_and_normal_motion_including_one_missing_sample(self):
        self.assertIsNone(continuity_reason(None, None, (10, 20, 110, 60), 0))
        for gap in (500, 1000):
            self.assertIsNone(continuity_reason((10, 20, 110, 60), 0, (20, 22, 122, 64), gap))
        self.assertEqual(continuity_reason((10, 20, 110, 60), 0, (10, 20, 110, 60), 1001), 'gap')

    def test_verified_2500ms_mixed_vehicle_pair_splits(self):
        previous, current = (1946, 1690, 2150, 1775), (1336, 1567, 1469, 1632)
        self.assertEqual(continuity_reason(previous, 28000, current, 30500), 'gap')
        self.assertEqual(continuity_reason(previous, 28000, current, 28500), 'relative_jump')

    def test_scale_change_is_bounded_in_both_dimensions_and_directions(self):
        normal = (0, 0, 100, 40)
        for changed in ((0, 0, 251, 40), (0, 0, 100, 101)):
            self.assertEqual(continuity_reason(normal, 0, changed, 500), 'scale_change')
            self.assertEqual(continuity_reason(changed, 0, normal, 500), 'scale_change')
        self.assertIsNone(continuity_reason(normal, 0, (0, 0, 250, 100), 500))

    def test_jump_boundary_is_inclusive(self):
        # A 30 by 40 box has diagonal 50: permitted displacement is exactly 75.
        self.assertIsNone(continuity_reason((0, 0, 30, 40), 0, (75, 0, 105, 40), 500))
        self.assertEqual(continuity_reason((0, 0, 30, 40), 0, (76, 0, 106, 40), 500), 'relative_jump')

    def test_invalid_or_nonforward_time_is_rejected(self):
        box = (0, 0, 100, 40)
        for stamp in (True, -1, 1.5, float('nan'), None):
            self.assertEqual(continuity_reason(box, 0, box, stamp), 'invalid_timestamp')
        for stamp in (0, 1):
            self.assertEqual(continuity_reason(box, 1, box, stamp), 'non_forward_timestamp')
        self.assertEqual(continuity_reason(box, None, box, 10), 'invalid_timestamp')

    def test_malformed_geometry_is_rejected_without_coercing_strings_or_bools(self):
        box = (0, 0, 100, 40)
        for bad in (None, (), (0, 0, 0, 1), (1, 0, 0, 1), (False, 0, 100, 40),
                    ('0', 0, 100, 40), (0, 0, float('inf'), 40), (0, 0, float('nan'), 40),
                    (0, 0, 10**1000, 40), (-1e308, 0, 1e308, 40)):
            self.assertEqual(continuity_reason(None, None, bad, 0), 'invalid_box')
            self.assertEqual(continuity_reason(bad, 0, box, 500), 'invalid_box')

    def test_policy_validation_and_custom_bounds(self):
        for values in ({'max_gap_ms': True}, {'max_gap_ms': -1}, {'max_scale_change': .9},
                       {'max_scale_change': float('nan')}, {'max_relative_jump': -1},
                       {'max_relative_jump': True}):
            with self.assertRaises(ValueError):
                PlateContinuityPolicy(**values)
        policy = PlateContinuityPolicy(max_gap_ms=500)
        self.assertEqual(continuity_reason((0, 0, 100, 40), 0, (0, 0, 100, 40), 501, policy), 'gap')


if __name__ == '__main__':
    unittest.main()
