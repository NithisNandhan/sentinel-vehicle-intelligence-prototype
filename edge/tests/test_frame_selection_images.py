import unittest

from sentinel_edge.frame_selection import assess_image


class FrameSelectionImageTests(unittest.TestCase):
    def plate(self):
        import cv2
        import numpy as np
        image = np.full((50, 200, 3), 225, np.uint8)
        cv2.putText(image, 'MH01AB1234', (7, 33), cv2.FONT_HERSHEY_SIMPLEX, .6, (20,20,20), 2)
        return image

    def test_native_pixels_are_unchanged_and_blur_lowers_image_score(self):
        import cv2
        import numpy as np
        image = self.plate()
        before = image.copy()
        result = assess_image(image)
        blurred = assess_image(cv2.GaussianBlur(image, (19,19), 0))
        self.assertTrue(np.array_equal(image, before))
        self.assertGreater(result['score'], blurred['score'])
        self.assertFalse(result['completeness_verified'])

    def test_tiny_native_crop_is_not_rescued_by_display_enlargement(self):
        import cv2
        small = cv2.resize(self.plate(), (30,8))
        self.assertIn('too_few_native_pixels', assess_image(small)['rejection_reasons'])


if __name__ == '__main__':
    unittest.main()
