import unittest

from sentinel_edge.vehicle_lifecycle import VehicleLifecycle


def obs(native='native1', box=(100, 100, 140, 140), signature=(1., 0.),
        vehicle_class=2, qualified=True, motion_support=None):
    return dict(id=native, box=box, signature=signature, vehicle_class=vehicle_class,
                qualified=qualified, motion_support=motion_support or {})


class VehicleLifecycleTests(unittest.TestCase):
    def lifecycle(self, **kwargs):
        return VehicleLifecycle((500, 800), **kwargs)

    def test_same_native_gap_prediction_and_stationary_recovery(self):
        life = self.lifecycle()
        first = life.update([obs()], 0)[0]
        predicted = life.update([], 500)[0]
        self.assertEqual(predicted['id'], first['id'])
        self.assertEqual(predicted['status'], 'predicted')
        self.assertFalse(predicted['observed'])
        self.assertIsNone(predicted['observed_box'])
        self.assertEqual(life.records[first['id']]['observation_count'], 1)
        self.assertEqual(life.update([], 1200)[0]['status'], 'lost_unresolved')
        recovered = life.update([obs(qualified=False)], 2000)[0]
        self.assertEqual(recovered['id'], first['id'])
        self.assertTrue(recovered['qualified'])
        self.assertTrue(recovered['observed'])
        self.assertEqual(life.records[first['id']]['observation_count'], 2)

    def test_new_native_recovers_with_strong_appearance(self):
        life = self.lifecycle()
        life.update([obs()], 0)
        life.update([], 500)
        recovered = life.update([obs('native2')], 1000)[0]
        self.assertEqual(recovered['id'], 'VT0001')
        self.assertEqual(recovered['native_ids'], ['native1', 'native2'])
        self.assertEqual(life.native_aliases['native2'], 'VT0001')

    def test_checked_motion_requires_appearance_too(self):
        life = self.lifecycle()
        life.update([obs()], 0)
        recovered = life.update([obs('native2', signature=(.6, .8),
                                    motion_support={'VT0001': .3})], 500)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]['id'], 'VT0001')
        life = self.lifecycle()
        life.update([obs()], 0)
        result = life.update([obs('native2', signature=(0., 1.),
                                 motion_support={'VT0001': 1.})], 500)
        self.assertEqual(len(result), 2)

    def test_different_new_native_class_and_gross_jump_cannot_merge(self):
        for changed in (obs('new',vehicle_class=7), obs(box=(600, 350, 640, 390))):
            with self.subTest(changed=changed):
                life = self.lifecycle()
                life.update([obs()], 0)
                result = life.update([changed], 100)
                self.assertEqual(len(result), 2)
                self.assertEqual(life.native_aliases[changed['id']], 'VT0002')
                self.assertFalse(next(t for t in result if t['id'] == 'VT0001')['observed'])

    def test_native_label_flicker_never_changes_geometrically_continuous_identity(self):
        life=self.lifecycle()
        identities=[]
        for index,cls in enumerate([2,2,7,2,7,2,7,2,2,7]):
            tracks=life.update([obs(vehicle_class=cls,box=(100+index,100,140+index,140))],index*100)
            identities.append(next(t['id'] for t in tracks if t['observed']))
        self.assertEqual(set(identities),{'VT0001'})
        self.assertEqual(life.records['VT0001']['observation_count'],10)
        self.assertEqual(life.records['VT0001']['vehicle_class'],2)

    def test_two_equal_candidates_abstain(self):
        life = self.lifecycle()
        life.update([obs('a'), obs('b', box=(120, 100, 160, 140))], 0)
        result = life.update([obs('new', box=(110, 100, 150, 140))], 100)
        self.assertEqual(len(result), 3)
        self.assertEqual(next(t for t in result if t['observed'])['id'], 'VT0003')

    def test_one_detection_per_parent_per_pts(self):
        life = self.lifecycle()
        life.update([obs()], 0)
        result = life.update([obs('a'), obs('b')], 100)
        self.assertEqual(life.records['VT0001']['observation_count'], 1)
        self.assertEqual(sum(t['observed'] for t in result), 2)

    def test_native_continuation_beats_new_alias_candidate(self):
        life = self.lifecycle()
        life.update([obs()], 0)
        life.update([obs(), obs('other')], 100)
        self.assertEqual(life.native_aliases['native1'], 'VT0001')
        self.assertEqual(life.native_aliases['other'], 'VT0002')
        self.assertEqual(life.records['VT0001']['observation_count'], 2)

    def test_boundary_exit_requires_border_and_outward_velocity(self):
        life = self.lifecycle()
        life.update([obs(box=(740, 100, 780, 140))], 0)
        life.update([obs(box=(770, 100, 810, 140))], 100)
        self.assertEqual(life.update([], 300), [])
        self.assertEqual(life.records['VT0001']['close_reason'], 'boundary_exit')
        life = self.lifecycle()
        life.update([obs(box=(100, 100, 140, 140))], 0)
        life.update([obs(box=(130, 100, 170, 140))], 100)
        self.assertEqual(life.update([], 1100)[0]['status'], 'predicted')
        self.assertEqual(life.update([], 5200), [])
        self.assertEqual(life.records['VT0001']['close_reason'], 'unresolved_timeout')
        self.assertEqual(life.counters['boundary_exit'], 0)

    def test_stationary_border_is_not_an_exit(self):
        life = self.lifecycle()
        for pts in (0, 100):
            life.update([obs(box=(780, 100, 820, 140))], pts)
        self.assertEqual(life.update([], 500)[0]['status'], 'predicted')

    def test_velocity_is_smoothed_and_only_real_observations_extend_span(self):
        life = self.lifecycle()
        life.update([obs()], 0)
        life.update([obs(box=(110, 100, 150, 140))], 100)
        current = life.update([obs()], 200)[0]
        self.assertGreater(current['velocity_px_per_ms'][0], 0)
        self.assertLess(current['velocity_px_per_ms'][0], .1)
        predicted = life.update([], 300)[0]
        self.assertEqual(predicted['observation_span_ms'], 200)
        self.assertEqual(predicted['observation_count'], 3)
        self.assertAlmostEqual(predicted['box'][0], 100+100*current['velocity_px_per_ms'][0])

    def test_scene_reset_closes_before_creating_new_parent(self):
        life = self.lifecycle()
        life.update([obs()], 0)
        result = life.update([obs()], 100, scene_reset=True)
        self.assertEqual(result[0]['id'], 'VT0002')
        self.assertEqual(life.records['VT0001']['close_reason'], 'scene_reset')

    def test_exclusions_qualification_and_capacity(self):
        life = self.lifecycle(max_active=1, excluded=('excluded',))
        self.assertEqual(life.update([obs('excluded'), obs('early', qualified=False)], 0), [])
        result = life.update([obs('a'), obs('b')], 100)
        self.assertEqual(len(result), 1)
        self.assertEqual(life.counters['capacity_rejected'], 1)
        self.assertEqual(life.counters['excluded_observations'], 1)

    def test_validation_is_atomic_and_pts_strict(self):
        life = self.lifecycle()
        life.update([obs()], 100)
        before = life.records
        for bad in (-1, 99, 100, float('nan'), float('inf'), True):
            with self.subTest(pts=bad), self.assertRaises(ValueError):
                life.update([], bad, scene_reset=True)
            self.assertEqual(life.records, before)
        for bad in (obs(box=(0, 0, float('nan'), 4)), obs(signature=(float('inf'), 0)),
                    obs(box=(0, 0, 0, 4)), obs(signature=(0, 0))):
            with self.subTest(observation=bad), self.assertRaises(ValueError):
                life.update([obs('valid'), bad], 200, scene_reset=True)
            self.assertEqual(life.records, before)
        self.assertEqual(life.update([], 200)[0]['status'], 'predicted')

    def test_state_and_closed_summaries_are_bounded(self):
        life = self.lifecycle(max_active=1)
        for pts in range(40):
            life.update([obs(str(pts))], pts, scene_reset=pts > 0)
        self.assertLessEqual(len(life.records), 9)
        self.assertEqual(life.counters['created'], 40)
        self.assertNotIn('history', next(iter(life.records.values())))
        for pts in range(40, 100):
            life.update([obs(str(pts))], pts)
        self.assertLessEqual(len(life.native_aliases), 32)
        self.assertEqual(len(life._active['VT0040']['history']), 2)
        self.assertEqual(len(life._active['VT0040']['signatures']), 2)


if __name__ == '__main__':
    unittest.main()
