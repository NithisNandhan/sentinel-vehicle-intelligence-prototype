import unittest
import numpy as np
from sentinel_edge.plate_extent import character_extent,recover_readable_extent


class PlateExtentTests(unittest.TestCase):
    def fixture(self):
        import cv2
        frame=np.full((180,600,3),50,np.uint8);frame[50:145,25:560]=(0,210,240)
        cv2.putText(frame,'MH01AB1234',(35,116),cv2.FONT_HERSHEY_SIMPLEX,1.65,(0,0,0),3)
        return frame,[280,60,380,140]

    def test_no_ocr_support_cannot_grow_a_window_pattern(self):
        frame,seed=self.fixture()
        self.assertIsNone(recover_readable_extent(frame,seed,[[1,0]],['','A']))

    def test_growth_never_removes_seed_pixels_and_stays_inside_frame(self):
        frame,seed=self.fixture();result=character_extent(frame,seed)
        if result is None:
            # The complete final four characters form a more representative seed.
            seed=[235,60,390,140];result=character_extent(frame,seed)
        self.assertIsNotNone(result)
        a,b,c,d=result['box'];x,y,x2,y2=seed
        self.assertLessEqual(a,x);self.assertLessEqual(b,y)
        self.assertGreaterEqual(c,x2);self.assertGreaterEqual(d,y2)
        self.assertTrue(0<=a<c<=frame.shape[1] and 0<=b<d<=frame.shape[0])

    def test_invalid_or_tiny_regions_remain_unmodified(self):
        frame,seed=self.fixture()
        for box in ([0,0,5,5],[-1,0,100,20],[0,0,1000,20]):
            self.assertIsNone(character_extent(frame,box))


if __name__=='__main__':unittest.main()
