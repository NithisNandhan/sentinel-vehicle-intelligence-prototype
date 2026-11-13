"""Prevent neighbouring plates becoming evidence for the wrong vehicle."""
from types import SimpleNamespace
import unittest

import numpy as np

from sentinel_edge.experimental_detectors import VehicleTrack
from sentinel_edge.plate_ownership import ownership_reason
from sentinel_edge.plate_identity import reading_conflict_reason
from sentinel_edge.tracked_ocr import TrackedOCR
from ctc_fixtures import TOKENS, rows_for


class PlateOwnershipTests(unittest.TestCase):
    def test_partial_prefix_cannot_hide_conflicting_high_score_plate_suffix(self):
        left = tuple((c,.98) for c in 'MH47BM5561')
        right = tuple((c,.98) for c in 'DLH5533')
        self.assertEqual(reading_conflict_reason(left,right),'numeric_suffix')
        self.assertEqual(reading_conflict_reason(right,left),'numeric_suffix')
        # A missing prefix or one questionable digit is not enough by itself.
        self.assertIsNone(reading_conflict_reason(left,tuple((c,.98) for c in 'BM5561')))
        self.assertIsNone(reading_conflict_reason(left,tuple((c,.98) for c in 'BM5562')))
        self.assertIsNone(reading_conflict_reason(left,tuple((c,.8) for c in 'DLH5533')))
        self.assertIsNone(reading_conflict_reason(left,tuple((c,.98) for c in 'TOYOTA')))

    def test_separate_vehicle_and_partial_overlap_do_not_block_owner(self):
        plate = (30, 60, 130, 90)
        self.assertIsNone(ownership_reason(plate, 'a', [('a', (0, 0, 180, 100)), ('b', (135, 0, 300, 100))]))
        self.assertIsNone(ownership_reason(plate, 'a', [('a', (0, 0, 180, 100)), ('b', (100, 0, 300, 100))]))

    def test_nested_or_duplicate_vehicle_regions_are_ambiguous(self):
        for other in [(20, 10, 150, 100), (0, 0, 180, 100)]:
            self.assertEqual(ownership_reason((30, 60, 130, 90), 'a',
                [('a', (0, 0, 180, 100)), ('b', other)]), 'overlapping_vehicles')

    def test_owner_is_required_and_malformed_geometry_is_not_admitted(self):
        self.assertEqual(ownership_reason((30, 60, 130, 90), 'a', [('b', (0, 0, 180, 100))]), 'outside_owner')
        self.assertEqual(ownership_reason((30, 60, float('nan'), 90), 'a', []), 'invalid_plate_box')
        self.assertIsNone(ownership_reason((30, 60, 130, 90), 'a', [('a', (0, 0, 180, 100)), ('b', None)]))

    def test_rejection_happens_before_ocr_and_does_not_refresh_history(self):
        calls = []
        model = SimpleNamespace(tokens=TOKENS, probabilities=lambda crop: calls.append(1) or rows_for(list('MH49BE5557')))
        pipeline = TrackedOCR(model, SimpleNamespace(box=lambda image: (30, 60, 130, 90)))
        image = np.zeros((100, 300, 3), np.uint8)
        owner = VehicleTrack('a', (0, 0, 180, 100), 2)
        self.assertEqual(len(pipeline.process(image, [owner], 0)), 1)
        history = pipeline.histories['a']
        original_passage = history.passage_id
        displays = pipeline.process(image, [owner, VehicleTrack('b', (20, 10, 150, 100), 2)], 500)
        self.assertEqual(displays, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(history.last_plate_pts_ms, 0)
        self.assertEqual(history.reader.observation_count, 1)
        self.assertEqual(history.passage_id, original_passage)
        self.assertEqual(pipeline.metrics['plate_ownership_rejections'], 2)

    def test_prepared_collector_proposals_use_the_same_guard(self):
        model = SimpleNamespace(tokens=TOKENS, probabilities=lambda crop: self.fail('Ambiguous crop reached OCR'))
        pipeline = TrackedOCR(model, None)
        image = np.zeros((100, 300, 3), np.uint8)
        cache = {'a': dict(box=(30, 60, 130, 90), crop=image[60:90, 30:130], ms=0, calls=1, fallback=False)}
        tracks = [VehicleTrack('a', (0, 0, 180, 100), 2), VehicleTrack('b', (20, 10, 150, 100), 2)]
        self.assertEqual(pipeline.process(image, tracks, 0, plate_cache=cache), [])
        self.assertEqual(pipeline.metrics['plate_ownership_rejections'], 1)


if __name__ == '__main__':
    unittest.main()
