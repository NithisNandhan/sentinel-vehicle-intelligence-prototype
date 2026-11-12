import unittest
import numpy as np
from sentinel_edge.plate_region_gate import verify_proposals


class GateTests(unittest.TestCase):
    def test_only_likely_plate_can_supply_ocr_evidence_and_all_decisions_survive(self):
        statuses=iter(['likely_non_plate','uncertain','likely_plate'])
        class Verifier:
            def verify(self,image,context=None):
                return dict(status=next(statuses),classifier_score=.5,reasons=['fixture'])
        proposals=[dict(box=[10,10,90,40],quad=None,confidence=.8) for _ in range(3)]
        source=np.arange(100*200*3,dtype=np.uint8).reshape(100,200,3)
        before=source.copy();saved=[]
        allowed,audit,counts=verify_proposals(source,proposals,Verifier(),
            observe=lambda i,p,n,w:saved.append((i,p['ocr_eligible'],n.copy(),w.copy())))
        self.assertEqual(len(allowed),1);self.assertEqual(len(audit),3)
        self.assertEqual([row[1] for row in saved],[False,False,True])
        self.assertEqual(counts,{'likely_non_plate':1,'uncertain':1,'likely_plate':1})
        self.assertTrue(np.array_equal(source,before))

    def test_out_of_frame_or_nonfinite_coordinates_are_not_silently_cropped(self):
        for box in ([0,0,300,20],[float('nan'),0,40,20]):
            with self.subTest(box=box),self.assertRaises(ValueError):
                verify_proposals(np.zeros((30,100,3),np.uint8),[dict(box=box)],None)

    def test_verified_geometry_keeps_priority_without_manufacturing_detector_score(self):
        class Verifier:
            def verify(self,image,context=None):
                return dict(status='likely_plate',classifier_score=.9,reasons=[])
        proposals=[dict(box=[0,0,80,30],quad=None,confidence=.99),
                    dict(box=[0,0,90,30],quad=None,confidence=0.,geometry={'boundary':'estimated_text_enclosure'})]
        allowed,_,_=verify_proposals(np.zeros((50,100,3),np.uint8),proposals,Verifier())
        self.assertEqual(allowed[0]['confidence'],0.)
        self.assertEqual(len(allowed),2,'Actual spatial deduplication remains a caller responsibility')

    def test_optional_provisional_policy_reads_uncertain_but_not_rejected_regions(self):
        statuses=iter(['uncertain','likely_non_plate'])
        class Verifier:
            def verify(self,image,context=None):
                return dict(status=next(statuses),classifier_score=.5,reasons=[])
        source=np.zeros((30,100,3),np.uint8)
        proposals=[dict(box=[0,0,80,25],quad=None,confidence=0.) for _ in range(2)]
        allowed,audit,_=verify_proposals(source,proposals,Verifier(),allow_uncertain=True)
        self.assertEqual(len(allowed),1)
        self.assertEqual(allowed[0]['plate_verification']['status'],'uncertain')
        self.assertTrue(allowed[0]['provisional_region'])
        self.assertFalse(audit[1]['ocr_eligible'])


if __name__=='__main__':unittest.main()
