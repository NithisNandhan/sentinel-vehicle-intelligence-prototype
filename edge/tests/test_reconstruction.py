from __future__ import annotations

import unittest
import numpy as np

from sentinel_edge.reconstruction import CropBank, native_crop, reconstruct


class CropTests(unittest.TestCase):
    def test_crop_keeps_native_pixels_and_owns_memory(self):
        image = np.arange(90, dtype=np.uint8).reshape(5, 6, 3)
        crop = native_crop(image, (-2, 1, 4, 9))
        expected = image[1:5, 0:4].copy()
        image[:] = 0
        np.testing.assert_array_equal(crop, expected)

    def test_empty_and_nonfinite_bounds_rejected(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        for box in ((5, 5, 2, 2), (0, 0, float('nan'), 5)):
            with self.assertRaises(ValueError):
                native_crop(image, box)

    def test_bank_has_bounded_memory_and_no_duplicate_evidence(self):
        bank = CropBank(max_bytes=72, max_frames=3, max_tracks=2)
        image = np.ones((2, 3, 3), dtype=np.uint8)
        self.assertTrue(bank.add('a', image, 1000))
        self.assertFalse(bank.add('a', image, 1300))
        for index in range(5):
            bank.add('a', image * (index + 2), 1600 + index * 300)
        self.assertEqual(len(bank.get('a')), 3)
        self.assertLessEqual(bank.bytes_used, 72)
        bank.add('b', image, 4000)
        bank.add('c', image, 4300)
        self.assertLessEqual(len(bank.tracks), 2)
        bank.clear()
        self.assertEqual(bank.bytes_used, 0)

    def test_large_crop_and_repeated_timestamp_rejected(self):
        bank = CropBank(max_bytes=20)
        self.assertFalse(bank.add('a', np.zeros((8, 8, 3), np.uint8), 0))
        self.assertTrue(bank.add('a', np.zeros((2, 2, 3), np.uint8), 1000))
        self.assertFalse(bank.add('a', np.ones((2, 2, 3), np.uint8), 1000))


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        import cv2
        self.cv2 = cv2
        self.base = np.random.default_rng(3).integers(30, 220, (48, 160, 3), dtype=np.uint8)
        self.base = cv2.GaussianBlur(self.base, (3, 3), 0)

    def samples(self):
        bank = CropBank()
        for index in range(3):
            image = self.base.copy()
            image[0, index] = index
            bank.add('a', image, index * 300)
        return bank.get('a')

    def test_identity_reconstruction_retains_three_sources(self):
        result = reconstruct(self.samples())
        self.assertIsNotNone(result)
        self.assertEqual(len(result.source_ids), 3)
        self.assertEqual(result.image.shape, self.base.shape)
        self.assertGreaterEqual(float(result.valid_mask.mean()), .7)
        self.assertLess(float(np.abs(result.image.astype(float) - self.base).mean()), 1)

    def test_too_few_observations_cannot_reconstruct(self):
        self.assertIsNone(reconstruct(self.samples()[:2]))

    def test_bad_alignment_cannot_add_false_support(self):
        def reject(reference, neighbour):
            return None
        self.assertIsNone(reconstruct(self.samples(), aligner=reject))

    def test_known_translation(self):
        bank = CropBank()
        for index in range(3):
            transform = np.float32([[1, 0, index], [0, 1, 0]])
            image = self.cv2.warpAffine(self.base, transform, (160, 48))
            bank.add('a', image, index * 300)
        result = reconstruct(bank.get('a'))
        self.assertIsNotNone(result)
        self.assertLess(float(np.abs(result.image[5:-5, 5:-5].astype(float) -
                                     self.base[5:-5, 5:-5]).mean()), 5)

    def test_unrelated_images_are_rejected(self):
        bank = CropBank()
        for index in range(3):
            bank.add('a', np.random.default_rng(index).integers(0, 255, (48, 160, 3), np.uint8), index * 300)
        self.assertIsNone(reconstruct(bank.get('a')))
