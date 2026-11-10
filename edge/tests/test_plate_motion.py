import unittest
import cv2
import numpy as np
from sentinel_edge.plate_motion import PlateMotion


class MotionTests(unittest.TestCase):
    @staticmethod
    def image(offset=0):
        image=np.zeros((200,400,3),dtype=np.uint8)
        patch=np.random.default_rng(42).integers(20,235,(60,120,3),dtype=np.uint8)
        image[80:140,70+offset:190+offset]=patch
        return image

    def test_native_translation_support_and_no_extra_observation(self):
        tracker=PlateMotion(width=400)
        tracker.advance(self.image(),0);tracker.observe('PT1',[70,80,190,140])
        tracker.advance(self.image(3),33)
        self.assertEqual(tracker.support([73,80,193,140]),{})
        tracker.advance(self.image(6),67)
        self.assertGreater(tracker.support([76,80,196,140])['PT1'],.9)
        self.assertEqual(tracker.states['PT1']['observed'],0)
        self.assertEqual(tracker.support([280,0,390,60]),{})

    def test_gap_and_lost_pixels_do_not_supply_motion(self):
        tracker=PlateMotion(width=400)
        tracker.advance(self.image(),0);tracker.observe('PT1',[70,80,190,140])
        tracker.advance(np.zeros((200,400,3),dtype=np.uint8),33)
        self.assertEqual(tracker.support([70,80,190,140]),{})
        tracker.advance(self.image(),1100)
        self.assertEqual(tracker.states,{})

    def test_bounds_and_forward_time(self):
        tracker=PlateMotion(width=200,max_active=1)
        tracker.advance(self.image(),0)
        tracker.observe('a',[-40,-20,500,400]);tracker.observe('b',[70,80,190,140])
        self.assertLessEqual(len(tracker.states),1)
        with self.assertRaises(ValueError):tracker.advance(self.image(),0)

    def test_failed_reseed_invalidates_old_prediction(self):
        tracker=PlateMotion(width=400)
        tracker.advance(self.image(),0);tracker.observe('PT1',[70,80,190,140])
        tracker.advance(self.image(3),33);tracker.advance(self.image(6),67)
        self.assertIn('PT1',tracker.support([76,80,196,140]))
        tracker.observe('PT1',[280,0,390,60])
        self.assertNotIn('PT1',tracker.states)
        self.assertEqual(tracker.support([76,80,196,140]),{})

    def test_repeated_rotation_does_not_accumulate_axis_aligned_growth(self):
        tracker=PlateMotion(width=400);image=self.image()
        tracker.advance(image,0);tracker.observe('PT1',[70,80,190,140])
        for i in range(1,8):
            matrix=cv2.getRotationMatrix2D((130,110),i*.3,1.)
            tracker.advance(cv2.warpAffine(image,matrix,(400,200)),i*33)
        box=tracker.states['PT1']['box']
        self.assertLess(box[2]-box[0],126)
        self.assertLess(box[3]-box[1],67)
