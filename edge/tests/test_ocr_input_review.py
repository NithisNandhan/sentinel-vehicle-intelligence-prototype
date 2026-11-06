"""Protect the diagnostic score/character mapping and exact input display."""
import unittest

from sentinel_edge.ocr_input_review import compare_label, inspect_crop, posterior_details


class OCRInputReviewTests(unittest.TestCase):
    tokens = ("", "A", "a", "B", "1", " ")

    def test_ctc_repeats_and_case_are_not_collapsed_twice(self):
        rows = [[0., .9, .1, 0., 0., 0.], [0., .1, .9, 0., 0., 0.],
                [1., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 0.]]
        result = posterior_details(rows, self.tokens)
        self.assertEqual(result["reading"], "AA1")
        self.assertEqual([c["timestep"] for c in result["characters"]], [0, 1, 3])
        self.assertEqual([c["score"] for c in result["characters"]], [1., 1., 1.])

    def test_peak_scores_and_alternatives_preserve_blank_mass(self):
        result = posterior_details([[.1, .5, .1, .2, .1, 0.],
                                    [.1, .6, .05, .15, .1, 0.]], self.tokens)
        char = result["characters"][0]
        self.assertEqual(result["reading"], "A")
        self.assertEqual(char["timestep"], 1)
        self.assertAlmostEqual(char["score"], .65)
        self.assertAlmostEqual(result["plate_score"], .65)
        self.assertEqual(char["alternatives"][0], {"character": "B", "score": .15})
        self.assertEqual(result["timesteps"][1]["blank"], .1)

    def test_missing_characters_never_get_an_invented_score(self):
        result = posterior_details([[0., .9, .1, 0., 0., 0.],
                                    [1., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., .1, .9, 0.]], self.tokens)
        compared = compare_label(result, "AB1")
        missing = compared["aligned"][1]
        self.assertIsNone(missing["observed_index"])
        self.assertIsNone(missing["expected_support"])
        self.assertEqual(compared["character_errors"], 1)
        self.assertFalse(compared["exact"])

    def test_empty_reading_and_insertions_remain_visible(self):
        empty = posterior_details([[1., 0., 0., 0., 0., 0.]], self.tokens)
        self.assertEqual(empty["reading"], "")
        self.assertIsNone(empty["plate_score"])
        result = posterior_details([[0., 1., 0., 0., 0., 0.],
                                    [1., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 0., 0.]], self.tokens)
        self.assertEqual(compare_label(result, "A")["extra_indices"], [1])

    def test_display_reverses_actual_tensor_and_keeps_padding(self):
        import base64
        import cv2
        import numpy as np
        image = np.full((24, 80, 3), (17, 101, 249), dtype=np.uint8)

        class Reader:
            tokens = OCRInputReviewTests.tokens

            def probabilities(self, crop):
                np.testing.assert_array_equal(crop, image)
                return [[0., 1., 0., 0., 0., 0.]]

        view = inspect_crop(Reader(), image)
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(view["model_input"].split(",")[1]),
                                            dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (48, 320, 3))
        np.testing.assert_array_equal(decoded[:, :160], np.full((48, 160, 3), (17, 101, 249), dtype=np.uint8))
        self.assertFalse(decoded[:, 160:].any())
        self.assertEqual(view["content_width"], 160)
        self.assertEqual(view["padding_width"], 160)
        self.assertEqual(view["tensor_shape"], [1, 3, 48, 320])


if __name__ == "__main__":
    unittest.main()
