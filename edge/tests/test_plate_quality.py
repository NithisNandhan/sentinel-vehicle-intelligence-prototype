import unittest

from sentinel_edge.plate_quality import crop_quality, deskew_view, select_frames
from sentinel_edge.reconstruction import CropSample


class PlateQualityTests(unittest.TestCase):
    def test_quality_is_bounded_and_does_not_mutate_native_pixels(self):
        import numpy as np
        import cv2
        image = np.full((32, 128, 3), 210, np.uint8)
        cv2.putText(image, 'TEST', (4, 25), cv2.FONT_HERSHEY_SIMPLEX, .8, (10, 10, 10), 2)
        original = image.copy()
        quality = crop_quality(image)
        self.assertTrue(.1 <= quality <= 1.)
        self.assertTrue(np.array_equal(original, image))
        self.assertGreater(quality, crop_quality(cv2.GaussianBlur(image, (15, 15), 0)))
        self.assertGreater(quality, crop_quality(cv2.resize(image, (24, 6))))
        self.assertEqual(crop_quality(np.zeros_like(image)), .1)

    def test_best_frames_preserve_time_order_and_distinct_sources(self):
        samples = [CropSample(None, i * 400, str(i), .1 + i / 20.) for i in range(10)]
        selected = select_frames(samples, limit=3)
        self.assertEqual([sample.source_id for sample in selected], ['7', '8', '9'])
        samples += [CropSample(None, 4000, '9', 1.), CropSample(None, 4010, 'other', .9)]
        selected = select_frames(samples, limit=3)
        self.assertEqual(len({sample.source_id for sample in selected}), 3)
        self.assertTrue(all(b.pts_ms - a.pts_ms >= 150 for a, b in zip(selected, selected[1:])))

    def test_invalid_selection_bounds_and_quality_fail(self):
        for samples in ([CropSample(None, 0, 'x', float('nan'))], [CropSample(None, -1, 'x', 1.)],
                        [CropSample(None, 0, 'x', 0.)], [CropSample(None, 0, 'x', 1.)] * 41):
            with self.assertRaises(ValueError):
                select_frames(samples)

    def test_deskew_no_edges_abstains_and_rotation_preserves_original(self):
        import cv2
        import numpy as np
        self.assertEqual(deskew_view(np.zeros((32, 128, 3), np.uint8)), (None, None))
        image = np.zeros((60, 180, 3), np.uint8)
        for y in (10, 30):
            cv2.line(image, (10, y), (160, y + 22), (255, 255, 255), 2)
        original = image.copy()
        corrected, angle = deskew_view(image)
        self.assertIsNotNone(corrected)
        self.assertAlmostEqual(angle, 8.3, delta=2.)
        self.assertGreaterEqual(corrected.shape[0], image.shape[0])
        self.assertGreaterEqual(corrected.shape[1], image.shape[1])
        self.assertTrue(np.array_equal(image, original))
