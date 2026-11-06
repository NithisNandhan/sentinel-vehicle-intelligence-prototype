"""The plate detector's own score is kept for diagnostics; box choice is unchanged."""
import unittest
from types import SimpleNamespace

import numpy as np

from sentinel_edge.detector_comparison import decode_yolo11
from sentinel_edge.plate_search import TargetedPlateSearch


class ScoredDetector:
    """Returns one box per call and remembers its score, like PlateDetector.box."""
    def __init__(self, results):
        self.results, self.last_confidence = list(results), None

    def box(self, image):
        box, self.last_confidence = self.results.pop(0)
        return box


class PlateConfidenceTests(unittest.TestCase):
    def test_decoder_scores_match_the_unscored_boxes(self):
        output = np.zeros((1, 5, 4), np.float32)
        output[0, :, 0] = (100, 100, 60, 20, .9)
        output[0, :, 1] = (300, 300, 60, 20, .4)
        output[0, :, 2] = (301, 300, 60, 20, .3)  # Suppressed by NMS.
        plain = decode_yolo11(output, 1., 0, 0, (512, 512, 3))
        scored = decode_yolo11(output, 1., 0, 0, (512, 512, 3), with_confidence=True)
        self.assertEqual([box for box, _ in scored], plain)
        self.assertEqual([round(score, 2) for _, score in scored], [.9, .4])

    def test_primary_box_reports_its_score(self):
        search = TargetedPlateSearch(ScoredDetector([((10, 10, 110, 40), .7)]))
        self.assertEqual(search.box(np.zeros((200, 400, 3), np.uint8)), (10, 10, 110, 40))
        self.assertEqual(search.last_confidence, .7)

    def test_fallback_reports_the_score_of_the_box_it_returns(self):
        detector = ScoredDetector([(None, None), ((10, 10, 60, 30), .3), ((10, 10, 110, 40), .6),
                                   (None, None), (None, None)])
        search = TargetedPlateSearch(detector)
        self.assertEqual(search.box(np.zeros((200, 400, 3), np.uint8)), (150, 10, 250, 40))  # Largest area.
        self.assertTrue(search.last_fallback)
        self.assertEqual(search.last_confidence, .6)

    def test_no_box_or_unscored_adapter_reports_none(self):
        search = TargetedPlateSearch(ScoredDetector([((0, 0, 2, 2), .9)]))
        self.assertIsNone(search.box(np.zeros((40, 100, 3), np.uint8)))
        self.assertIsNone(search.last_confidence)
        search = TargetedPlateSearch(SimpleNamespace(box=lambda image: (10, 10, 110, 40)))
        search.box(np.zeros((200, 400, 3), np.uint8))
        self.assertIsNone(search.last_confidence)


if __name__ == '__main__':
    unittest.main()
