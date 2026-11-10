import unittest
import numpy as np
from sentinel_edge.plate_evidence import character_support,character_components,decide_plate_evidence,annotate_plate_evidence

TOKENS=('', 'A', '1', '-')


def feature(t,*,chars=0,strokes=1,y=.3,source=None):
    return dict(position_ms=t,source_id=source or str(t),confident_characters=chars,
                character_components=strokes,observed_parent_y=y)


class PlateEvidenceTests(unittest.TestCase):
    def test_repeated_upper_body_blank_is_suspect(self):
        result=decide_plate_evidence([feature(0),feature(250),feature(500,strokes=5)])
        self.assertEqual(result['status'],'suspected_non_plate')
        self.assertTrue(result['excluded_from_fusion'])

    def test_one_confident_character_in_any_original_protects_a_difficult_plate(self):
        result=decide_plate_evidence([feature(0),feature(250),feature(500,chars=1)])
        self.assertEqual(result['status'],'text_evidence_present')
        self.assertFalse(result['excluded_from_fusion'])

    def test_reading_failure_alone_never_rejects_a_plate(self):
        for features in ([feature(0,y=None),feature(250,y=None)],
                         [feature(0,y=.8),feature(250,y=.8)],
                         [feature(0,strokes=5),feature(250,strokes=5)],
                         [feature(0),feature(250,y=None)], [feature(0)]):
            with self.subTest(features=features):
                self.assertEqual(decide_plate_evidence(features)['status'],'uncertain')

    def test_duplicates_and_adjacent_frames_are_not_independent_support(self):
        for frames in ([feature(0,source='same'),feature(250,source='same')],
                       [feature(0),feature(50)]):
            self.assertFalse(decide_plate_evidence(frames)['excluded_from_fusion'])

    def test_ctc_only_counts_confident_collapsed_alphanumerics(self):
        rows=[[.1,.9,0,0],[.1,.9,0,0],[1,0,0,0],[.1,.9,0,0],[.1,0,0,.9]]
        self.assertEqual(character_support(rows,TOKENS),2)
        self.assertEqual(character_support(np.array(rows),TOKENS),2)
        with self.assertRaises(ValueError):character_support([[float('nan'),0,0,1]],TOKENS)

    def test_components_preserve_both_text_polarities(self):
        import cv2
        plate=np.full((50,200,3),255,np.uint8)
        cv2.putText(plate,'AB123',(8,38),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        self.assertGreaterEqual(character_components(plate),3)
        self.assertGreaterEqual(character_components(255-plate),3)
        self.assertEqual(character_components(np.zeros((50,200,3),np.uint8)),0)

    def test_predicted_or_unlocked_parent_cannot_supply_rejection_context(self):
        for observed,owner in ((False,'VT1'),(True,None)):
            group=dict(id='PT1',parent_vehicle_id='VT1',methods=dict(multi=dict(top='A1',accepted=None)))
            samples=[dict(position_ms=t,source_id=str(t),vehicle_id=owner,box=[20,10,70,30],rows=[[1,0,0,0]]) for t in (0,250)]
            timeline=dict(frames=[dict(t=t,tracks=[dict(id='VT1',box=[0,0,100,100],observed=observed)]) for t in (0,250)])
            annotate_plate_evidence([group],[dict(id='PT1',observations=samples)],TOKENS,timeline,
                                    lambda _:np.zeros((20,50,3),np.uint8))
            self.assertFalse(group['plate_validation']['excluded_from_fusion'])

    def test_quarantine_keeps_raw_hypothesis_and_ownership_but_blocks_acceptance(self):
        group=dict(id='PT1',parent_vehicle_id='VT1',methods=dict(multi=dict(top='A1',accepted='A1',accepted_for_vehicle='A1')))
        samples=[dict(position_ms=t,source_id=str(t),vehicle_id='VT1',box=[20,10,70,30],rows=[[1,0,0,0]]) for t in (0,250)]
        timeline=dict(frames=[dict(t=t,tracks=[dict(id='VT1',box=[0,0,100,100],observed=True)]) for t in (0,250)])
        annotate_plate_evidence([group],[dict(id='PT1',observations=samples)],TOKENS,timeline,
                                lambda _:np.zeros((20,50,3),np.uint8))
        self.assertTrue(group['plate_validation']['excluded_from_fusion'])
        self.assertEqual(group['parent_vehicle_id'],'VT1')
        self.assertEqual(group['methods']['multi']['top'],'A1')
        self.assertIsNone(group['methods']['multi']['accepted'])
        self.assertIsNone(group['methods']['multi']['accepted_for_vehicle'])


if __name__=='__main__':unittest.main()
