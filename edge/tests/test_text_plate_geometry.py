"""Synthetic image geometry, independent of models, labels and OCR strings."""
import json
import unittest

import cv2
import numpy as np

from sentinel_edge.text_plate_geometry import text_plate_candidates


def scene():
    return np.full((240, 480, 3), 210, np.uint8)


def quad(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def text(image, value, point, size=.9):
    cv2.putText(image, value, point, cv2.FONT_HERSHEY_SIMPLEX, size, (30, 30, 30), 2)


class TextPlateGeometryTests(unittest.TestCase):
    def test_native_image_and_json_contract(self):
        image = scene()
        text(image, 'AB1234', (120, 155))
        original = image.copy()
        results = text_plate_candidates(image, [quad(118, 132, 231, 158)])
        self.assertTrue(results)
        self.assertTrue(np.array_equal(image, original))
        result = json.loads(json.dumps(results))[0]
        self.assertEqual(result['geometry']['boundary'], 'estimated_text_enclosure')
        self.assertFalse(result['geometry']['physical_plate_corners_verified'])
        self.assertEqual(result['text_members'], [0])
        self.assertEqual(len(result['quad']), 4)
        self.assertGreaterEqual(result['character_components'], 3)

    def test_weak_seed_does_not_require_ocr_or_three_characters(self):
        result = text_plate_candidates(scene(), [quad(120, 130, 148, 152)])[0]
        self.assertEqual(result['character_components'], 0)
        self.assertTrue(any('Weak stroke' in reason for reason in result['reasons']))

    def test_preserves_tilt_and_perspective_instead_of_axis_box(self):
        image = scene()
        seed = [[100, 110], [250, 140], [245, 167], [96, 142]]
        result = text_plate_candidates(image, [seed])[0]
        q = result['quad']
        self.assertGreater(q[1][1]-q[0][1], 20)
        self.assertEqual(result['seed_quad'], [[float(x), float(y)] for x, y in seed])
        self.assertGreater(q[2][0], q[3][0])
        self.assertGreater(q[3][1], q[0][1])

    def test_neighbouring_fragments_generate_same_row_enclosure(self):
        image = scene()
        polygons = [quad(100, 130, 155, 155), quad(164, 130, 230, 155)]
        candidates = text_plate_candidates(image, polygons)
        grouped = [c for c in candidates if c['text_members'] == [0, 1]]
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]['geometry']['relation'], 'same_row')
        self.assertLess(grouped[0]['box'][0], 100)
        self.assertGreater(grouped[0]['box'][2], 230)

    def test_two_rows_are_grouped(self):
        polygons = [quad(120, 100, 180, 121), quad(100, 128, 210, 150)]
        groups = [c for c in text_plate_candidates(scene(), polygons) if len(c['text_members']) == 2]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['geometry']['relation'], 'two_row')

    def test_three_adjacent_text_fragments_can_form_one_enclosure(self):
        polygons = [quad(70,130,110,155),quad(119,130,159,155),quad(168,130,215,155)]
        results = text_plate_candidates(scene(),polygons)
        self.assertTrue(any(c['text_members'] == [0,1,2] for c in results))

    def test_three_rows_do_not_form_one_registration_group(self):
        polygons = [quad(100,70,210,90),quad(100,100,210,120),quad(100,130,210,150)]
        results = text_plate_candidates(scene(),polygons)
        self.assertFalse(any(len(c['text_members']) == 3 for c in results))

    def test_far_or_differently_oriented_text_is_not_joined(self):
        polygons = [quad(30, 100, 90, 125), quad(250, 100, 320, 125),
                    [[130, 80], [160, 110], [145, 125], [115, 95]]]
        results = text_plate_candidates(scene(), polygons)
        self.assertTrue(results)
        self.assertTrue(all(len(c['text_members']) == 1 for c in results))

    def test_two_positive_plates_veto_text_union_even_when_small(self):
        polygons = [quad(100, 130, 155, 155), quad(164, 130, 230, 155)]
        positives = [[100, 130, 155, 155], [164, 130, 230, 155]]
        results = text_plate_candidates(scene(), polygons, positives)
        self.assertTrue(results)
        self.assertTrue(all(len(c['text_members']) == 1 for c in results))
        broad = text_plate_candidates(scene(), [quad(90, 120, 240, 165)], positives)
        self.assertEqual(broad, [])

    def test_border_clipping_keeps_valid_native_coordinates(self):
        image = scene()
        result = text_plate_candidates(image, [quad(0, 2, 90, 24)])[0]
        self.assertEqual(result['box'][0], 0)
        self.assertEqual(result['box'][1], 0)
        for x, y in result['quad']:
            self.assertTrue(0 <= x < image.shape[1] and 0 <= y < image.shape[0])

    def test_invalid_inputs_are_rejected(self):
        invalid = [[], [[0, 0]]*4, quad(-1, 2, 40, 30), quad(10, 10, 500, 40),
                   [[10, 10], [90, 40], [90, 10], [10, 40]],
                   [[10, 10], [float('nan'), 10], [60, 40], [10, 40]]]
        self.assertEqual(text_plate_candidates(scene(), invalid), [])
        with self.assertRaises(ValueError):
            text_plate_candidates(np.zeros((0, 10, 3), np.uint8), [])
        with self.assertRaises(ValueError):
            text_plate_candidates(scene(), [], max_candidates=0)

    def test_duplicate_outlines_are_not_counted_as_two_text_members(self):
        seed = quad(100, 130, 220, 155)
        results = text_plate_candidates(scene(), [seed, seed])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['text_members'], [0])

    def test_grows_from_fragment_to_connected_native_characters(self):
        image = scene()
        text(image, 'AB12345', (100, 155))
        # A detector outline catches only the leading two characters.
        result = text_plate_candidates(image, [quad(98, 131, 138, 158)])[0]
        self.assertGreater(result['box'][2], 195)
        self.assertGreater(result['geometry']['added_components'], 0)


if __name__ == '__main__':
    unittest.main()
