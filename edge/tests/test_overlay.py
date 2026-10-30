"""The call-outs are display only: they must never fail on awkward geometry."""
from types import SimpleNamespace
import unittest

import numpy as np

from sentinel_edge.overlay import (PLATE_LABEL_SIZE, VEHICLE_TAG_SIZE, draw_plate_callout, draw_plate_label,
                                   draw_vehicle, scale_for)


class OverlayTests(unittest.TestCase):
    def test_scale_grows_with_resolution(self):
        self.assertEqual(scale_for(np.zeros((480, 640, 3), np.uint8)), 1.)
        self.assertEqual(scale_for(np.zeros((2160, 3840, 3), np.uint8)), 3.)

    def test_callout_draws_inside_any_frame_and_changes_only_the_copy(self):
        crop = np.full((30, 120, 3), 200, np.uint8)
        for shape, vehicle, plate in (((1080, 1920, 3), (400, 300, 900, 800), (560, 650, 680, 680)),
                                      ((2160, 3840, 3), (3300, 20, 3839, 700), (3500, 500, 3690, 600)),   # top-right edge
                                      ((120, 160, 3), (0, 0, 160, 120), (5, 5, 150, 35)),                  # tiny test frame
                                      ((1080, 1920, 3), (0, 900, 300, 1079), (20, 1000, 140, 1030))):     # bottom-left edge
            original = np.full(shape, 60, np.uint8)
            view = original.copy()
            draw_vehicle(view, SimpleNamespace(track_id='7', box=vehicle), scale_for(view))
            for status in ('accepted by model — NOT verified', 'uncertain', 'collecting distinct frames', 'something new'):
                draw_plate_callout(view, vehicle, plate, crop, 'TS07FS8429', status, 3, scale_for(view))
            draw_plate_callout(view, vehicle, plate, crop, None, 'collecting distinct frames', 1, scale_for(view))
            self.assertEqual(view.shape, original.shape)
            self.assertTrue((view != original).any())
            self.assertTrue((original == 60).all())

    def test_plate_label_marks_only_the_plate_area_on_any_frame(self):
        for shape, plate in (((1080, 1920, 3), (560, 650, 680, 680)),
                             ((2160, 3840, 3), (3700, 2, 3839, 60)),   # top-right edge
                             ((120, 160, 3), (5, 5, 150, 35))):         # tiny test frame
            original = np.full(shape, 60, np.uint8)
            for reading in ('TS07FS8429', None):
                view = original.copy()
                draw_plate_label(view, plate, reading, 'uncertain', scale_for(view))
                self.assertEqual(view.shape, original.shape)
                self.assertTrue((view != original).any())
                self.assertTrue((original == 60).all())
        # Nothing is drawn far from the plate, so vehicles elsewhere stay visible.
        view = np.full((1080, 1920, 3), 60, np.uint8)
        draw_plate_label(view, (900, 700, 1020, 730), 'TS07FS8429', 'uncertain', scale_for(view))
        rows, cols = np.nonzero((view != 60).any(axis=2))
        self.assertGreaterEqual(rows.min(), 600)
        self.assertLessEqual(rows.max(), 740)
        self.assertLessEqual(cols.max() - cols.min(), 400)

    def test_vehicle_tag_is_smaller_than_the_plate_reading(self):
        self.assertLess(VEHICLE_TAG_SIZE, PLATE_LABEL_SIZE)
        view = np.full((1080, 1920, 3), 60, np.uint8)
        draw_vehicle(view, SimpleNamespace(track_id='7', box=(400, 300, 900, 800), vehicle_class=2, confidence=.25),
                     scale_for(view))
        rows = np.nonzero((view[300:400, 405:895] != 60).any(axis=(1, 2)))[0]
        tag_height = rows.max() - rows.min() + 1
        view = np.full((1080, 1920, 3), 60, np.uint8)
        draw_plate_label(view, (560, 650, 680, 680), 'TS07FS8429', 'uncertain', scale_for(view))
        rows = np.nonzero((view[:648] != 60).any(axis=(1, 2)))[0]
        self.assertLess(tag_height, rows.max() - rows.min() + 1)

    def test_an_oversized_crop_is_bounded_to_a_third_of_the_frame(self):
        view = np.zeros((300, 400, 3), np.uint8)
        draw_plate_callout(view, (10, 10, 390, 290), (50, 50, 350, 250), np.full((200, 300, 3), 255, np.uint8),
                           'AB12CD3456', 'uncertain', 2, 1.)
        self.assertEqual(view.shape, (300, 400, 3))
