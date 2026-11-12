import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sentinel_edge.experimental_detectors import PlateDetector, PROFILES
from sentinel_edge.plate_search import TargetedPlateSearch


class PlateQualitySearchTests(unittest.TestCase):
    def test_default_adapter_retains_single_call_and_outputs(self):
        detector=PlateDetector.__new__(PlateDetector)
        detector.native_scale=False
        values=[([10,10,80,30],.8)]
        detector._scored_boxes_single=Mock(return_value=values)
        self.assertIs(detector.scored_boxes(np.zeros((100,200,3),np.uint8)),values)
        self.assertEqual(detector.last_scale_calls,1)
        detector._scored_boxes_single.assert_called_once()

    def test_quality_profile_uses_same_parent_and_protects_original_crop(self):
        self.assertEqual(PROFILES['general'],('yolo26s','parent'))
        self.assertEqual(PROFILES['general_quality'],('yolo26s','parent_scale'))
        detector=PlateDetector.__new__(PlateDetector);detector.native_scale=True
        detector._scored_boxes_single=Mock(side_effect=[
            [([20,20,80,40],.6)], [([20,20,80,40],.99)], [], [], [], []])
        result=detector.scored_boxes(np.zeros((200,400,3),np.uint8))
        self.assertEqual(result,[([20.,20.,80.,40.],.6)])
        self.assertEqual(detector.last_scale_calls,6)

    def test_existing_fallback_does_not_recursively_tile_quality_miss(self):
        detector=SimpleNamespace(box=Mock(return_value=None),last_scale_calls=6)
        search=TargetedPlateSearch(detector)
        self.assertIsNone(search.box(np.zeros((400,600,3),np.uint8)))
        detector.box.assert_called_once()
        self.assertEqual(search.last_calls,6)

    def test_quality_success_reports_actual_model_calls_and_score(self):
        detector=SimpleNamespace(box=Mock(return_value=(20,20,80,40)),
                                 last_scale_calls=6,last_confidence=.6)
        search=TargetedPlateSearch(detector)
        self.assertEqual(search.box(np.zeros((200,400,3),np.uint8)),(20,20,80,40))
        self.assertEqual(search.last_calls,6)
        self.assertEqual(search.last_confidence,.6)


if __name__=='__main__':unittest.main()
