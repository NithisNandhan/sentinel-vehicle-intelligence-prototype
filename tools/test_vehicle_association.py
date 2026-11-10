"""Fail-closed replay boundary tests; no model loading or GPU inference."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_vehicle_association import (CountedEncoder, MODEL_SHA,
                                           validate_alignment, validate_capture)


class ReplayBoundaries(unittest.TestCase):
    def setUp(self):
        self.document = dict(model_sha256=MODEL_SHA, vehicle_inference_frames=2,
                             decoded_frames=2, width=30, height=20, frames=[
                                 dict(position_ms=0, discontinuity=False, detections=[[1, 2, 10, 15, .8, 2]]),
                                 dict(position_ms=40, discontinuity=False, detections=[])])

    def test_valid_capture_and_exact_native_alignment(self):
        validate_capture(self.document)
        frame = SimpleNamespace(pts_ms=140, discontinuity=False, image=np.zeros((20, 30, 3)))
        validate_alignment(frame, self.document['frames'][1], SimpleNamespace(start_pts_ms=100), self.document)

    def test_capture_rejects_bad_model_clock_and_boxes(self):
        for mutate in (
                lambda d: d.update(model_sha256='unexpected'),
                lambda d: d.update(vehicle_inference_frames=1),
                lambda d: d['frames'][1].update(position_ms=0),
                lambda d: d['frames'][0]['detections'][0].__setitem__(2, 1),
                lambda d: d['frames'][0]['detections'][0].__setitem__(4, float('nan')),
                lambda d: d['frames'][0]['detections'][0].__setitem__(5, 0)):
            document = copy.deepcopy(self.document)
            mutate(document)
            with self.assertRaises(ValueError):
                validate_capture(document)

    def test_native_clock_shape_and_discontinuity_must_match(self):
        source = SimpleNamespace(start_pts_ms=100)
        for pts, shape, discontinuity in ((139, (20, 30, 3), False),
                                          (140, (21, 30, 3), False),
                                          (140, (20, 30, 3), True)):
            frame = SimpleNamespace(pts_ms=pts, discontinuity=discontinuity, image=np.zeros(shape))
            with self.assertRaises(ValueError):
                validate_alignment(frame, self.document['frames'][1], source, self.document)

    def test_appearance_counts_real_embeddings(self):
        encoder = CountedEncoder(lambda image, boxes: [np.array([1., 2., 3.])] * len(boxes))
        encoder(None, [1, 2])
        self.assertEqual((encoder.calls, encoder.features, encoder.dimension), (1, 2, 3))

    def test_appearance_rejects_omitted_zero_nan_and_changed_features(self):
        for features in ([], [None], [np.zeros(3)], [np.array([float('nan')])]):
            with self.assertRaises(ValueError):
                CountedEncoder(lambda image, boxes: features)(None, [1])
        outputs = iter(([np.ones(3)], [np.ones(4)]))
        encoder = CountedEncoder(lambda image, boxes: next(outputs))
        encoder(None, [1])
        with self.assertRaises(ValueError):
            encoder(None, [1])


if __name__ == '__main__':
    unittest.main()
