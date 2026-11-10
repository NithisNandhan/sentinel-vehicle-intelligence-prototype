import unittest
import numpy as np
from sentinel_edge.vehicle_reid import preprocess


class VehicleAppearanceTests(unittest.TestCase):
    def test_original_rgb_raw_range_and_shape(self):
        image=np.full((8,8,3),[10,20,30],dtype=np.uint8)
        tensor=preprocess(image,[[4,4,8,8]])
        self.assertEqual(tensor.shape,(1,3,208,208))
        self.assertEqual(tensor.dtype,np.float32)
        np.testing.assert_array_equal(tensor[0,:,0,0],[30.,20.,10.])

    def test_outside_or_invalid_boxes_fail(self):
        image=np.zeros((8,8,3),dtype=np.uint8)
        for box in [[20,20,2,2],[4,4,0,2],[float('nan'),4,2,2]]:
            with self.assertRaises(ValueError):preprocess(image,[box])


if __name__=='__main__':unittest.main()
