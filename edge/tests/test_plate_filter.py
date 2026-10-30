"""Opt-in rejection of plate proposals that are not plates (blank panels, bars, grilles)."""
from types import SimpleNamespace
import unittest

import numpy as np

from sentinel_edge.plate_filter import NonPlateFilter, confident_characters
from sentinel_edge.plate_options import PlateOptions
from sentinel_edge.tracked_ocr import TrackedOCR

TOKENS = ('', *'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
TEXT = 'AP01AB1234'  # Well-formed Indian layout: what an invented reading can look like.


def rows_for(text, peak):
    """One CTC row per character; the rest of the mass on the blank."""
    rows = []
    for char in text:
        row = [0.] * len(TOKENS)
        row[TOKENS.index(char)], row[0] = peak, 1. - peak
        rows.append(row)
    return rows


def engine(peak, options=None, score=None):
    detector = SimpleNamespace(box=lambda image: (5, 5, 150, 35), last_confidence=score)
    return TrackedOCR(SimpleNamespace(tokens=TOKENS, probabilities=lambda image: rows_for(TEXT, peak)),
                      detector, options=options)


def run(pipeline, frames=4):
    track = SimpleNamespace(track_id='1', box=(0, 0, 180, 60), vehicle_class=2)
    for i in range(frames):
        image = np.full((60, 180, 3), 100, np.uint8)
        image[10, 10] = i  # Distinct crops, as from distinct frames.
        pipeline.process(image, [track], i * 500)
    return pipeline.histories['1']


class FilterTests(unittest.TestCase):
    def test_confident_characters_follow_ctc_collapse(self):
        self.assertEqual(confident_characters(rows_for(TEXT, .9)), 10)
        self.assertEqual(confident_characters(rows_for(TEXT, .55)), 0)
        self.assertEqual(confident_characters(rows_for('AA', .9)), 1)  # Repeat without blank collapses.

    def test_reasons(self):
        screen, crop = NonPlateFilter(), np.zeros((20, 80, 3), np.uint8)
        self.assertIsNone(screen.reason(crop, rows_for(TEXT, .9), .8))
        self.assertIsNone(screen.reason(crop, rows_for(TEXT, .9), None))  # Unknown score is not held against it.
        self.assertIn('score', screen.reason(crop, rows_for(TEXT, .9), .2))
        self.assertIn('confident', screen.reason(crop, rows_for(TEXT, .5), .8))
        self.assertIn('shape', screen.reason(np.zeros((10, 107, 3), np.uint8), rows_for(TEXT, .9), .8))
        self.assertIn('shape', screen.reason(np.zeros((40, 20, 3), np.uint8), rows_for(TEXT, .9), .8))

    def test_off_by_default(self):
        self.assertFalse(PlateOptions().reject_non_plates)
        self.assertFalse(PlateOptions.from_env({}).reject_non_plates)
        self.assertIsNone(engine(.9).non_plates)


class FusionSafetyTests(unittest.TestCase):
    def test_fusion_never_accepts_a_low_confidence_invented_reading_even_without_the_filter(self):
        history = run(engine(.55))
        self.assertEqual(history.reader.observation_count, 4)  # Off: weak evidence is kept, as before.
        self.assertIsNone(history.reader.accepted(.85))

    def test_with_the_filter_a_non_plate_never_becomes_evidence(self):
        pipeline = engine(.55, PlateOptions(reject_non_plates=True), score=.8)
        history = run(pipeline)
        self.assertEqual(history.reader.observation_count, 0)
        self.assertIsNone(history.reader.accepted(.85))
        self.assertEqual(pipeline.metrics['non_plate_rejections'], 4)
        self.assertEqual(pipeline.metrics['distinct_ocr_observations'], 0)
        self.assertIn('confident', history.last_plate_failure)
        self.assertEqual(history.plate_detections, 0)

    def test_a_low_detector_score_is_rejected_before_it_can_reset_the_reader(self):
        pipeline = engine(.9, PlateOptions(reject_non_plates=True), score=.8)
        history = run(pipeline, frames=3)
        self.assertEqual(history.reader.observation_count, 3)
        pipeline.detector.last_confidence = .1
        image = np.full((60, 180, 3), 100, np.uint8)
        pipeline.process(image, [SimpleNamespace(track_id='1', box=(0, 0, 180, 60), vehicle_class=2)], 1500)
        self.assertEqual(pipeline.histories['1'].reader.observation_count, 3)

    def test_a_clear_plate_is_still_accepted_with_the_filter(self):
        for options in (PlateOptions(), PlateOptions(reject_non_plates=True)):
            history = run(engine(.95, options, score=.8))
            self.assertEqual(history.reader.observation_count, 4)
            self.assertEqual(history.reader.accepted(.85).plate, TEXT)


if __name__ == '__main__':
    unittest.main()
