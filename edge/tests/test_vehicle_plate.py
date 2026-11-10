import unittest
from sentinel_edge.vehicle_plate import VehiclePlateRouter,deduplicate_views,one_view_per_timestamp,SceneBoundary,vehicle_signature


def proposal(box=(100,100,200,150),text='KA01AB1234'):
    return dict(box=list(box),confidence=.9,signature=[1.,0.],reading=tuple((c,.96) for c in text))


def parent(name='VT1',velocity=(0.,0.),observed=True,box=(0,0,400,400)):
    return dict(id=name,box=list(box),qualified=observed,observed=observed,
                status='visible' if observed else 'predicted',velocity_px_per_ms=velocity,observation_span_ms=1000)


class VehiclePlateTests(unittest.TestCase):
    def test_visual_identity_survives_wrong_ocr_with_established_same_parent(self):
        router=VehiclePlateRouter(visual_identity=True)
        first=router.route([proposal()],[parent()],0)[0]
        router.route([proposal()],[parent()],250)
        changed=router.route([proposal(text='TN99XY5678')],[parent()],500)[0]
        self.assertEqual(changed['group'],first['group'])
        self.assertTrue(changed['recognition_conflict'])
        self.assertEqual(changed['owner'],'VT1')

    def test_visual_identity_still_forks_for_different_parent(self):
        router=VehiclePlateRouter(visual_identity=True)
        first=router.route([proposal()],[parent('A')],0)[0]
        router.route([proposal()],[parent('A')],250)
        changed=router.route([proposal(text='TN99XY5678')],[parent('B')],500)[0]
        self.assertNotEqual(changed['group'],first['group'])

    def test_same_ocr_is_not_a_visual_identity_link(self):
        router=VehiclePlateRouter(visual_identity=True)
        first=router.route([proposal()],[],0)[0]
        other=proposal();other['signature']=[0.,1.]
        second=router.route([other],[],250)[0]
        self.assertNotEqual(first['group'],second['group'])

    def test_same_parent_and_relative_location_support_changed_crop_appearance(self):
        router=VehiclePlateRouter(visual_identity=True)
        first=router.route([proposal()],[parent()],0)[0]
        changed=proposal(text='KA01');changed['signature']=[.4,.916515]
        second=router.route([changed],[parent()],250)[0]
        self.assertEqual(first['group'],second['group']);self.assertEqual(second['owner'],'VT1')

    def test_parent_anchor_does_not_join_distant_plate_surface(self):
        router=VehiclePlateRouter(visual_identity=True)
        first=router.route([proposal()],[parent()],0)[0]
        changed=proposal((210,100,310,150),text='KA01');changed['signature']=[.4,.916515]
        second=router.route([changed],[parent()],250)[0]
        self.assertNotEqual(first['group'],second['group'])

    def test_strong_checked_flow_can_keep_unowned_visual_identity(self):
        router=VehiclePlateRouter(visual_identity=True)
        first=router.route([proposal()],[],0)[0]
        changed=proposal(text='TN99XY5678');changed['motion_support']={first['group']:.95}
        second=router.route([changed],[],250)[0]
        self.assertEqual(first['group'],second['group']);self.assertTrue(second['recognition_conflict'])
        self.assertIsNone(second['owner'])

    def test_nested_partial_crop_removed_but_separate_side_view_survives(self):
        full=proposal();fragment=proposal((110,100,135,150),text='KA')
        side=proposal((300,100,380,150))
        selected,metrics=deduplicate_views([fragment,full,side])
        self.assertEqual(len(selected),2);self.assertIn(full,selected);self.assertIn(side,selected)
        self.assertEqual(metrics['nested_or_overlapping_plate_views'],1)

    def test_strong_conflicting_text_is_not_silently_suppressed(self):
        selected,_=deduplicate_views([proposal(),proposal(text='TN99XY5678')])
        self.assertEqual(len(selected),2)

    def test_motion_resolves_overlapping_parent_boxes(self):
        router=VehiclePlateRouter()
        previous=proposal((200,100,300,150));current=proposal((150,100,250,150))
        vehicles=[parent('car',(-.2,0)),parent('other',(-2.,0))]
        first=router.route([previous],vehicles,0)[0]
        second=router.route([current],vehicles,250)[0]
        third=router.route([proposal((100,100,200,150))],vehicles,500)[0]
        self.assertIsNone(first['owner']);self.assertEqual(second['owner_candidate'],'car')
        self.assertEqual(third['owner'],'car');self.assertEqual(third['group'],first['group'])

    def test_predicted_parent_never_confirms_owner(self):
        router=VehiclePlateRouter();vehicles=[parent(observed=False)]
        router.route([proposal()],vehicles,0)
        second=router.route([proposal()],vehicles,250)[0]
        self.assertFalse(second['owner_locked']);self.assertIsNone(second['owner'])

    def test_ordinary_parent_change_forks_before_any_old_ocr_is_reused(self):
        router=VehiclePlateRouter()
        first=router.route([proposal()],[parent('A')],0)[0]
        second=router.route([proposal()],[parent('A')],250)[0]
        self.assertEqual(second['owner'],'A')
        other=router.route([proposal()],[parent('B')],500)[0]
        self.assertNotEqual(other['group'],first['group'])
        self.assertIsNone(other['owner'])
        self.assertEqual(other['owner_candidate'],'B')

    def test_reviewed_uncertain_region_veto_cannot_be_overridden_by_motion(self):
        router=VehiclePlateRouter();veto=parent('reviewed',observed=False)
        veto['ownership_veto']=True
        vehicles=[parent('car',(-.2,0)),parent('other',(-2.,0)),veto]
        router.route([proposal((200,100,300,150))],vehicles,0)
        second=router.route([proposal((150,100,250,150))],vehicles,250)[0]
        self.assertIsNone(second['owner_candidate']);self.assertFalse(second['owner_locked'])
        self.assertEqual(second['ownership_status'],'reviewed_overlap_veto')

    def test_parent_preserves_plate_history_during_occlusion_without_fake_votes(self):
        router=VehiclePlateRouter();vehicles=[parent()]
        first=router.route([proposal()],vehicles,0)[0]
        router.route([proposal()],vehicles,250)
        router.retained_parent_ids={'VT1'}
        router.route([],[],1500)
        self.assertEqual(len(router.groups[first['group']].observations),2)
        recovered=router.route([proposal()],vehicles,2000)[0]
        self.assertEqual(recovered['group'],first['group'])
        router.route([],[],8000)
        self.assertEqual(router.groups,{})

    def test_two_views_one_timestamp_are_one_vote_and_conflicts_exposed(self):
        tokens=['','A','B','C','X','Y','Z'];make=lambda chars:[[(.99 if i==tokens.index(c) else .01/6) for i in range(7)] for c in chars]
        observations=[dict(position_ms=0,source_id='a',quality=.5,rows=make('ABC')),
                      dict(position_ms=0,source_id='b',quality=.9,rows=make('ABC')),
                      dict(position_ms=250,source_id='c',quality=.7,rows=make('XYZ'))]
        views,metadata=one_view_per_timestamp(observations,tokens)
        self.assertEqual(len(views),2);self.assertEqual(views[0]['source_id'],'b')
        self.assertEqual(metadata['same_frame_views_not_counted'],1)
        self.assertTrue(metadata['conflicting_plate_views'])

    def test_tracking_descriptor_and_scene_boundary(self):
        import numpy as np
        black=np.zeros((100,200,3),dtype=np.uint8);white=np.full_like(black,255)
        self.assertTrue(vehicle_signature(black,[0,0,100,100]))
        scene=SceneBoundary();self.assertFalse(scene.changed(black));self.assertFalse(scene.changed(black))
        self.assertTrue(scene.changed(white))
