from types import SimpleNamespace
import unittest
import numpy as np

from sentinel_edge.crop_completion import extends_same_plate,completeness,CompletePlateSearch,intersects_another_plate


class CropCompletionTests(unittest.TestCase):
    def test_enclosing_text_can_extend_a_digit_fragment_but_not_another_plate(self):
        seed=[150,80,230,120]
        self.assertTrue(extends_same_plate(seed,[40,78,232,122]))
        self.assertFalse(extends_same_plate(seed,[240,80,360,120]))
        self.assertFalse(extends_same_plate(seed,[0,0,500,200]))
        self.assertFalse(extends_same_plate(seed,[30,160,232,202]))
        self.assertFalse(extends_same_plate(seed,[160,85,220,115]))

    def test_partial_crop_has_lower_selection_priority_than_complete_plate(self):
        import cv2
        image=np.full((80,400,3),(0,210,240),np.uint8)
        cv2.putText(image,'MH01AB1234',(12,57),cv2.FONT_HERSHEY_SIMPLEX,1.55,(0,0,0),3)
        self.assertGreater(completeness(image),completeness(image[:,270:]))

    def test_independent_plate_veto_does_not_treat_an_enclosing_same_plate_as_another(self):
        seed=(100,100,200,140);expanded=(100,100,330,140)
        self.assertTrue(intersects_another_plate(seed,expanded,[((230,100,330,140),'plate',.9)]))
        self.assertFalse(intersects_another_plate(seed,expanded,[(expanded,'plate',.9)]))
        self.assertFalse(intersects_another_plate(seed,expanded,[((230,150,330,190),'plate',.9)]))

    def test_related_outline_expands_native_box_without_an_extra_observation(self):
        import cv2
        image=np.full((160,450,3),60,np.uint8)
        image[60:130,30:420]=(0,210,240)
        cv2.putText(image,'MH01AB1234',(35,113),cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,0,0),3)
        primary=SimpleNamespace(box=lambda _: (270,60,420,130),last_text_quad=None,last_confidence=.9,
            last_calls=1,last_fallback=False,last_raw_proposals=[((30,60,420,130),'text',None,((30,60),(420,60),(420,130),(30,130)))],
            text_detector=None,diagnostics=lambda:{})
        wrapped=CompletePlateSearch(primary)
        output=wrapped.box(image)
        self.assertLess(output[0],100);self.assertGreaterEqual(output[2],420)
        self.assertLessEqual(output[1],60);self.assertGreaterEqual(output[3],130)
        self.assertTrue(wrapped.last_completion['recovered'])
        self.assertEqual(wrapped.last_calls,1)
        self.assertTrue(np.array_equal(image[60:130,30:420][0,0],np.array([0,210,240],np.uint8)))

    def test_retries_are_bounded_and_unrelated_text_never_replaces_seed(self):
        import cv2
        image=np.full((180,480,3),180,np.uint8)
        cv2.putText(image,'1234',(270,110),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,0),2)
        calls=[]
        primary=SimpleNamespace(box=lambda _: (265,70,365,120),last_text_quad=None,last_confidence=.9,
            last_calls=1,last_fallback=False,last_raw_proposals=[],text_detector=SimpleNamespace(polygons=lambda x:calls.append(1) or []),
            begin_frame=lambda:None,diagnostics=lambda:{})
        wrapped=CompletePlateSearch(primary,max_retries=1);wrapped.begin_frame()
        for _ in range(4):wrapped.box(image)
        self.assertLessEqual(len(calls),1)


if __name__=='__main__':unittest.main()
