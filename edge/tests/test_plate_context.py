import unittest
from sentinel_edge.plate_context import expanded_box,fragment_pair,spans_fragments,context_proposals


class ContextTests(unittest.TestCase):
    def test_padding_preserves_source_bounds(self):
        self.assertEqual(expanded_box([5,5,90,90],(100,100,3)),[0,0,100,100])
        self.assertEqual(expanded_box([1000,1000,2000,2000],(4000,4000,3)),[800,800,2200,2200])

    def test_known_split_requires_new_evidence(self):
        a=[2297,1935,2510,2047];b=[2493,1890,2921,2160]
        self.assertTrue(fragment_pair(a,b))
        self.assertTrue(spans_fragments([2290,1900,2915,2150],a,b))
        self.assertFalse(spans_fragments(a,a,b))
        self.assertFalse(fragment_pair(a,[3000,1935,3210,2047]))

    def test_two_separate_plates_not_joined(self):
        self.assertFalse(fragment_pair([0,0,100,30],[120,0,220,30]))
        self.assertFalse(fragment_pair([0,0,100,30],[20,0,80,30]))

    def test_context_is_native_slice_and_one_observation(self):
        import numpy as np
        frame=np.arange(300*500*3,dtype=np.uint8).reshape(300,500,3)
        class Search:
            last_text_quad=None;last_confidence=.9
            def box(self,region):
                self.region=region.copy();return [50,50,150,80]
        search=Search()
        proposals,metrics=context_proposals(frame,[dict(box=[100,100,200,200])],search)
        self.assertTrue(np.array_equal(search.region,frame[52:248,52:248]))
        self.assertEqual(proposals[0]['box'],[102,102,202,132])
        self.assertEqual(metrics['padded_search'],1)

    def test_fragments_replaced_only_after_whole_plate_redetection(self):
        import numpy as np
        class Search:
            last_text_quad=None;last_confidence=.9
            def __init__(self,last):self.boxes=iter([[58,58,158,108],[63,58,168,108],last])
            def box(self,region):return next(self.boxes)
        frame=np.zeros((400,500,3),dtype=np.uint8)
        vehicles=[dict(box=[90,90,210,200]),dict(box=[180,90,310,200])]
        recovered,metrics=context_proposals(frame,vehicles,Search([35,19,245,79]))
        self.assertEqual(len(recovered),1)
        self.assertEqual(recovered[0]['box'],[95,95,305,155])
        self.assertEqual(len(recovered[0]['recovered_from']),2)
        self.assertEqual(metrics['fragments_recovered'],1)
        retained,metrics=context_proposals(frame,vehicles,Search([40,24,140,74]))
        self.assertEqual(len(retained),2)
        self.assertEqual(metrics['fragment_recovery_rejected'],1)

    def test_oriented_alternative_has_crop_local_quad_and_preserved_boundary_metadata(self):
        import numpy as np
        class Search:
            last_text_quad=None;last_confidence=None
            def box(self,region):
                self.last_candidates=[dict(box=[50,50,150,90],quad=[[50,55],[145,50],[150,85],[55,90]],
                    confidence=None,completion=None,proposal_sources=['text_corners'],
                    geometry={'boundary':'estimated_text_enclosure'})]
                return self.last_candidates[0]['box']
        proposals,_=context_proposals(np.zeros((300,500,3),np.uint8),[dict(box=[100,100,200,200])],Search())
        self.assertEqual(proposals[0]['quad'],[(0,5),(95,0),(100,35),(5,40)])
        self.assertEqual(proposals[0]['box'],[102,102,202,142])
        self.assertEqual(proposals[0]['geometry']['boundary'],'estimated_text_enclosure')


if __name__=='__main__':unittest.main()
