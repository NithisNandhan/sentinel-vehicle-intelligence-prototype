import unittest

from sentinel_edge.plate_identity import PlateIdentityTracker


def plate(box=(20, 50, 60, 70), text='KA01AB1234', signature=(1., 0.), peak=.96, **metadata):
    return dict(box=box, confidence=.9, signature=signature,
                reading=tuple((char, peak) for char in text), **metadata)


def vehicle(name='A', box=(0, 0, 100, 100), qualified=True, **extra):
    return dict(id=name, box=box, qualified=qualified, **extra)


class PlateIdentityTests(unittest.TestCase):
    def test_motion_support_survives_appearance_change_but_not_text_conflict(self):
        tracker=PlateIdentityTracker()
        first=tracker.route([plate()],[],0)[0]
        second=tracker.route([plate(text='KA01AB123',signature=(0.,1.),motion_support={first['group']:.95})],[],100)[0]
        self.assertEqual(first['group'],second['group'])
        conflict=tracker.route([plate(text='TN99XY1234',signature=(1.,0.),motion_support={first['group']:.95})],[],200)[0]
        self.assertNotEqual(first['group'],conflict['group'])

    def test_vehicle_id_switch_preserves_plate_but_reconfirms_owner(self):
        tracker = PlateIdentityTracker()
        rows = [tracker.route([plate((20 + offset, 50, 60 + offset, 70))],
                              [vehicle(owner, (-20, 0, 200, 100))], pts)[0]
                for offset, owner, pts in [(0, 'A', 0), (20, 'A', 100),
                                          (40, 'B', 200), (60, 'B', 300)]]
        self.assertEqual(len({row['group'] for row in rows}), 1)
        self.assertEqual([row['owner'] for row in rows], [None, 'A', None, 'B'])
        self.assertEqual([row['owner_candidate'] for row in rows], ['A', 'A', 'B', 'B'])
        self.assertEqual(tracker.counters['owner_conflict'], 1)

    def test_strong_contradictory_characters_forbid_identical_appearance(self):
        tracker = PlateIdentityTracker()
        first = tracker.route([plate()], [], 0)[0]
        second = tracker.route([plate(text='TN99XY1234')], [], 100)[0]
        self.assertNotEqual(first['group'], second['group'])
        self.assertEqual(tracker.counters['association_text_conflict'], 1)

    def test_anchor_retains_strong_conflict_through_partial_reading(self):
        tracker = PlateIdentityTracker()
        first = tracker.route([plate()], [], 0)[0]
        partial = tracker.route([plate(text='1234', peak=.5)], [], 100)[0]
        other = tracker.route([plate(text='TN99XY1234')], [], 200)[0]
        self.assertEqual(first['group'], partial['group'])
        self.assertNotEqual(first['group'], other['group'])

    def test_ambiguous_owner_keeps_reading_and_does_not_publish_old_lock(self):
        tracker = PlateIdentityTracker()
        tracker.route([plate()], [vehicle()], 0)
        locked = tracker.route([plate()], [vehicle()], 100)[0]
        # Rival covers 25% of the plate, insufficient to contain it.
        ambiguous = tracker.route([plate()], [vehicle(), vehicle('B', (50, 0, 100, 100), False)], 200)[0]
        self.assertEqual(locked['group'], ambiguous['group'])
        self.assertEqual(ambiguous['ownership_status'], 'ambiguous_owner')
        self.assertIsNone(ambiguous['owner'])
        self.assertIsNone(ambiguous['owner_candidate'])
        self.assertFalse(ambiguous['owner_locked'])
        self.assertEqual(ambiguous['reading'], plate()['reading'])

    def test_excluded_flagged_and_tentative_never_own(self):
        for excluded, candidate, status in [(['A'], vehicle(), 'excluded_owner'),
                                             ([], vehicle(qualified=False), 'unqualified_owner'),
                                             ([], vehicle(flagged=True), 'unqualified_owner')]:
            with self.subTest(status=status, candidate=candidate):
                tracker = PlateIdentityTracker(excluded)
                tracker.route([plate()], [candidate], 0)
                row = tracker.route([plate()], [candidate], 100)[0]
                self.assertIsNone(row['owner'])
                self.assertEqual(row['ownership_status'], status)

    def test_same_frame_duplicates_count_once_and_metadata_is_preserved(self):
        tracker = PlateIdentityTracker()
        payload = dict(rows=[{'text': 'not-a-vote'}], source_id='roi-A', quad=[1, 2, 3, 4])
        rows = tracker.route([plate(**payload), plate((21, 50, 61, 70))], [vehicle()], 0)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]['owner_locked'])
        self.assertEqual(tracker.counters['duplicate_proposals'], 1)
        for key, value in payload.items():
            self.assertEqual(rows[0][key], value)
        self.assertTrue(tracker.route([plate()], [vehicle()], 100)[0]['owner_locked'])

    def test_equally_plausible_histories_abstain(self):
        tracker = PlateIdentityTracker()
        first = tracker.route([plate((0, 0, 40, 20)), plate((80, 0, 120, 20))], [], 0)
        middle = tracker.route([plate((40, 0, 80, 20))], [], 100)[0]
        self.assertNotIn(middle['group'], {row['group'] for row in first})
        self.assertEqual(tracker.counters['ambiguous_association'], 1)

    def test_two_competing_observations_do_not_enter_same_group(self):
        tracker = PlateIdentityTracker()
        original = tracker.route([plate((40, 0, 80, 20))], [], 0)[0]
        rows = tracker.route([plate((0, 0, 40, 20)), plate((80, 0, 120, 20))], [], 100)
        self.assertEqual(len({row['group'] for row in rows}), 2)
        self.assertNotIn(original['group'], {row['group'] for row in rows})

    def test_simultaneous_independent_plates_have_separate_groups(self):
        tracker = PlateIdentityTracker()
        first = tracker.route([plate(), plate((140, 50, 180, 70), 'TN99XY5678', (0., 1.))], [], 0)
        second = tracker.route([plate((145, 50, 185, 70), 'TN99XY5678', (0., 1.)),
                                plate((25, 50, 65, 70))], [], 100)
        self.assertEqual([row['group'] for row in first], [row['group'] for row in reversed(second)])
        self.assertEqual(len({row['group'] for row in second}), 2)

    def test_complete_strong_text_fallback_but_no_fragment_concatenation(self):
        tracker = PlateIdentityTracker()
        first = tracker.route([plate()], [], 0)[0]
        same = tracker.route([plate(signature=(0., 1.))], [], 100)[0]
        fragment = tracker.route([plate(text='KA01', signature=(1., 0.))], [], 200)[0]
        self.assertEqual(first['group'], same['group'])
        self.assertNotEqual(first['group'], fragment['group'])
        self.assertEqual(fragment['reading'], plate(text='KA01')['reading'])

    def test_stationary_and_velocity_prediction(self):
        tracker = PlateIdentityTracker()
        first = tracker.route([plate()], [], 0)[0]
        self.assertEqual(first['group'], tracker.route([plate()], [], 100)[0]['group'])
        tracker = PlateIdentityTracker()
        first = tracker.route([plate((0, 0, 40, 20))], [], 0)[0]
        tracker.route([plate((200, 0, 240, 20))], [], 100)
        # 400 pixels from last observation exceeds the small-crop floor gate;
        # the two-observation velocity prediction permits the continuation.
        predicted = tracker.route([plate((600, 0, 640, 20))], [], 300)[0]
        self.assertEqual(first['group'], predicted['group'])

    def test_geometry_limits_gap_expiry_and_capacity(self):
        tracker = PlateIdentityTracker()
        first = tracker.route([plate()], [], 0)[0]
        far = tracker.route([plate((1000, 0, 1040, 20))], [], 100)[0]
        self.assertNotEqual(first['group'], far['group'])
        tracker.route([], [], 1101)
        self.assertEqual(tracker.groups, {})
        self.assertEqual(tracker.counters['expired_groups'], 2)
        rows = tracker.route([plate((300 * index, 0, 300 * index + 40, 20)) for index in range(513)], [], 1200)
        self.assertEqual(len(rows), 513)
        self.assertEqual(len(tracker.groups), 512)
        self.assertEqual(tracker.counters['capacity_evictions'], 1)

    def test_invalid_pts_and_nonfinite_signature(self):
        tracker = PlateIdentityTracker()
        for pts in [None, -1, True, 1.5]:
            with self.subTest(pts=pts), self.assertRaises(ValueError):
                tracker.route([], [], pts)
        tracker.route([], [], 100)
        for pts in [100, 99]:
            with self.subTest(pts=pts), self.assertRaises(ValueError):
                tracker.route([], [], pts)
        for signature in [(float('nan'), 1), (float('inf'), 1), (0., 0.), (), ('x', 1)]:
            with self.subTest(signature=signature):
                self.assertEqual(tracker.route([plate(signature=signature)], [], tracker.last_pts + 1), [])
        self.assertEqual(tracker.counters['invalid_signature'], 5)

    def test_duplicate_vehicle_ids_are_rejected_before_state_advances(self):
        tracker = PlateIdentityTracker()
        with self.assertRaises(ValueError):
            tracker.route([plate()], [vehicle(), vehicle()], 0)
        self.assertEqual(tracker.route([plate()], [], 0)[0]['group'], 'PT0001')
