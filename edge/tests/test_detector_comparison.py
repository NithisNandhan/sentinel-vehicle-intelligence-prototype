import unittest

import numpy as np

from sentinel_edge.detector_comparison import decode_yolo11, iou, letterbox, score_target, summarize, transformed


class DetectorComparisonTests(unittest.TestCase):
    def test_no_detection_and_background_not_called_false_positive(self):
        self.assertEqual(score_target([], [0,0,10,10])['hit_iou50'], 0)
        score = score_target([[0,0,10,10],[20,20,30,30]], [0,0,10,10])
        self.assertEqual(score['hit_iou50'], 1)
        self.assertEqual(score['unmatched_boxes_not_verified_fp'], 1)
        self.assertNotIn('precision', score)
        self.assertEqual(iou([0,0,1,1],[2,2,3,3]), 0)

    def test_resolution_changes_annotation_by_actual_rounded_dimensions(self):
        image = np.zeros((101,201,3),dtype=np.uint8)
        result,box = transformed(image,'resolution_half',[10,20,100,80])
        self.assertEqual(result.shape[:2], (50,100))
        self.assertAlmostEqual(box[2],100*100/201)
        self.assertEqual(image.shape[:2], (101,201))

    def test_letterbox_rgb_and_unpadding_single_class_predictions(self):
        image = np.zeros((100,200,3),dtype=np.uint8); image[:,:,2] = 255
        tensor,scale,left,top = letterbox(image,640,640)
        self.assertEqual(tensor.shape,(1,3,640,640))
        self.assertEqual(float(tensor[0,0,320,320]),1)
        output = np.asarray([[[320],[320],[320],[160],[.9]]],dtype=np.float32)
        boxes = decode_yolo11(output,scale,left,top,image.shape)
        np.testing.assert_allclose(boxes, [[50,25,150,75]])
        with self.assertRaises(ValueError): decode_yolo11(np.zeros((1,6,8400)),1,0,0,image.shape)

    def test_summary_keeps_sample_size_and_no_accuracy_claim(self):
        row = {'elapsed_ms':10,**score_target([[0,0,10,10]],[0,0,10,10])}
        result = summarize([row,row])
        self.assertEqual(result['target_hits_iou50'], 2)
        self.assertEqual(result['cases'], 2)
        self.assertNotIn('accuracy', result)


if __name__ == '__main__': unittest.main()
