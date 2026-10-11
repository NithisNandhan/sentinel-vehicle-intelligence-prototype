"""The call-outs are display only: they must never fail on awkward geometry."""
from types import SimpleNamespace
import unittest

import numpy as np

from sentinel_edge.overlay import draw_plate_callout, draw_vehicle, scale_for


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

    def test_an_oversized_crop_is_bounded_to_a_third_of_the_frame(self):
        view = np.zeros((300, 400, 3), np.uint8)
        draw_plate_callout(view, (10, 10, 390, 290), (50, 50, 350, 250), np.full((200, 300, 3), 255, np.uint8),
                           'AB12CD3456', 'uncertain', 2, 1.)
        self.assertEqual(view.shape, (300, 400, 3))
