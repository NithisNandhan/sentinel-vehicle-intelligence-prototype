import unittest
from types import SimpleNamespace

import numpy as np

from sentinel_edge.automatic_crops import NativePlateCropper, Proposal, clip, native_crop, rectify_boundary, windows
from sentinel_edge.text_boundaries import warp_polygon


class NativeCropTests(unittest.TestCase):
    def test_tiles_cover_all_source_pixels_with_bounded_overlap(self):
        covered = np.zeros((216,384),bool)
        for x,y,x2,y2 in windows((216,384,3),256):
            covered[y:y2,x:x2] = True
        self.assertTrue(covered.all())
        self.assertEqual(len(windows((2160,3840,3),768)),28)

    def test_source_crop_is_unresized_and_original_is_unchanged(self):
        image = np.arange(120*300*3,dtype=np.uint8).reshape(120,300,3)
        original = image.copy()
        crop,box = native_crop(image,(10,20,110,50))
        a,b,c,d = box
        np.testing.assert_array_equal(crop,image[b:d,a:c])
        crop[:] = 0
        np.testing.assert_array_equal(image,original)
        self.assertIsNone(clip((0,0,float('nan'),50),image.shape))

    def test_detection_does_not_depend_on_vehicle_or_reference(self):
        seen = []
        def boxes(image):
            seen.append(image.shape)
            return [[30,30,100,60]] if len(seen)>1 else []
        cropper = NativePlateCropper(SimpleNamespace(boxes=boxes),sizes=(256,))
        proposals = cropper.proposals(np.zeros((400,600,3),np.uint8))
        self.assertGreater(len(proposals),1)
        self.assertTrue(all(p.source=='tile-256' for p in proposals))
        self.assertEqual(cropper.calls,1+len(windows((400,600,3),256)))

    def test_refinement_cannot_jump_to_nearby_plate(self):
        detector = SimpleNamespace(boxes=lambda image:[[0,0,100,30]])
        cropper = NativePlateCropper(detector)
        proposal = Proposal((200,100,300,140),'test')
        self.assertEqual(cropper.refine(np.zeros((400,600,3),np.uint8),proposal),proposal)

    def test_work_budget_and_proposal_cap_are_enforced(self):
        cropper = NativePlateCropper(SimpleNamespace(boxes=lambda image:[[30,30,100,60]]),
                                    sizes=(256,),max_calls=2,max_proposals=1)
        self.assertEqual(len(cropper.proposals(np.zeros((600,600,3),np.uint8))),1)
        self.assertEqual(cropper.calls,2)
        self.assertTrue(cropper.budget_exhausted)

    def test_blank_crop_abstains_from_corner_estimation(self):
        view,quad=rectify_boundary(np.zeros((100,200,3),np.uint8),(30,20,150,70))
        self.assertIsNone(view)
        self.assertIsNone(quad)

    def test_polygon_warp_is_separate_and_rejects_tiny_regions(self):
        image=np.full((100,200,3),128,np.uint8)
        before=image.copy()
        result=warp_polygon(image,[(20,20),(160,25),(155,65),(15,60)])
        self.assertEqual(result.ndim,3)
        np.testing.assert_array_equal(image,before)
        self.assertIsNone(warp_polygon(image,[(0,0),(5,0),(5,4),(0,4)]))


if __name__=='__main__':
    unittest.main()
