"""Image/CTC ablation boundaries, without checkpoints or network calls."""
import unittest

import numpy as np

from sentinel_edge.awiros_experiment import input_array
from sentinel_edge.ocr_crop_experiment import contrast_view, decoded, observe_policies, padded_view, row_boundary


class CropExperimentTests(unittest.TestCase):
    def test_native_tensor_keeps_bgr_order_and_publisher_padding(self):
        image = np.zeros((24, 50, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        tensor = input_array(image)
        self.assertEqual(tensor.shape, (1, 3, 48, 320))
        self.assertTrue((tensor[0, 0, :, :100] == 1).all())
        self.assertTrue((tensor[0, 2, :, :100] == -1).all())
        self.assertTrue((tensor[:, :, :, 100:] == -1).all())

    def test_padding_and_contrast_never_mutate_native_pixels(self):
        image = np.random.default_rng(42).integers(0, 256, (24, 100, 3), dtype=np.uint8)
        original = image.copy()
        padded, contrast = padded_view(image), contrast_view(image)
        self.assertTrue(np.array_equal(image, original))
        self.assertTrue(np.array_equal(padded[1:-1, 2:-2], original))
        self.assertEqual(contrast.dtype, image.dtype)
        self.assertEqual(contrast.shape, image.shape)

    def test_empty_or_wrong_type_input_is_rejected(self):
        for image in (np.zeros((0, 4, 3), np.uint8), np.zeros((12, 12), np.uint8),
                      np.zeros((12, 12, 3), np.float32)):
            with self.assertRaises(ValueError):
                padded_view(image)

    def test_row_split_needs_central_gap_and_two_ink_bands(self):
        image = np.full((60, 120, 3), 255, np.uint8)
        image[8:23, 15:105] = 0
        image[38:53, 15:105] = 0
        self.assertTrue(24 <= row_boundary(image) <= 37)
        self.assertIsNone(row_boundary(np.full_like(image, 255)))
        self.assertIsNone(row_boundary(np.full((24, 140, 3), 255, np.uint8)))

    def test_raw_ctc_repeat_semantics_survive_case_conversion(self):
        tokens = ("", "A", "a", "1")
        self.assertEqual(decoded([[0, 1, 0, 0], [0, 0, 1, 0]], tokens), "AA")
        self.assertEqual(decoded([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], tokens, beam=True), "AA")

    def test_policy_outputs_are_views_not_temporal_observations(self):
        class Reader:
            tokens = ("", "A", "1")
            calls = 0
            def synchronize(self):
                pass
            def probabilities(self, _):
                self.calls += 1
                return [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        reader = Reader()
        image = np.full((24, 140, 3), 255, np.uint8)
        results = observe_policies(reader, image, ("native", "native_beam", "padding", "two_row"))
        self.assertEqual(reader.calls, 2)  # Native shared, one padded view; no row split.
        self.assertTrue(all(row["reading"] == "A1" for row in results.values()))
        self.assertTrue(all("frames" not in row for row in results.values()))
        with self.assertRaises(ValueError):
            observe_policies(reader, image, ("unknown",))


if __name__ == "__main__":
    unittest.main()
