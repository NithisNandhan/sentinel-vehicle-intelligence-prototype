import unittest

import numpy as np

from sentinel_edge.ocr_style import mixed_font_plate


class StyleTests(unittest.TestCase):
    def test_seeded_mixed_size_training_render_is_deterministic_and_bounded(self):
        first = mixed_font_plate("MH12AB1234", np.random.default_rng(42))
        second = mixed_font_plate("MH12AB1234", np.random.default_rng(42))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first.shape[0], 64)
        self.assertEqual(first.shape[2], 3)
        self.assertGreater(first.shape[1], 80)
        self.assertLess(first.shape[1], 800)
        self.assertGreater(int(first.max()) - int(first.min()), 100)

    def test_invalid_training_labels_cannot_silently_render(self):
        for text in ["", "123", "MH12 AB1234", "mh12ab1234", "MH12?B1234"]:
            with self.assertRaises(ValueError):
                mixed_font_plate(text, np.random.default_rng(42))


if __name__ == "__main__":
    unittest.main()
