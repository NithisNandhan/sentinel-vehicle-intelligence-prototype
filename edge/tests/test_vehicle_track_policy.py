import json
import unittest

from sentinel_edge.vehicle_track_policy import VehicleTrackPolicy


def detection(native_id='1', confidence=.6, vehicle_class=2, box=None):
    return {'native_id': native_id, 'confidence': confidence,
            'vehicle_class': vehicle_class, 'box': [0, 0, 100, 50] if box is None else box}


class VehicleTrackPolicyTests(unittest.TestCase):
    def test_three_source_timestamps_two_high_scores_and_span_qualify_stationary_box(self):
        policy = VehicleTrackPolicy()
        self.assertEqual(policy.update([detection(confidence=.4)], 0)[0]['status'], 'tentative')
        self.assertEqual(policy.update([detection(confidence=.1)], 100)[0]['status'], 'tentative')
        record = policy.update([detection(confidence=.4)], 150)[0]
        self.assertEqual(record['status'], 'qualified')
        self.assertEqual(record['hits'], 3)
        self.assertEqual(record['high_confidence_hits'], 2)
        self.assertEqual(record['qualified_pts_ms'], 150)
        self.assertEqual(record['box'], [0, 0, 100, 50])
        self.assertEqual(policy.counters['passages_qualified'], 1)

    def test_three_fast_frames_remain_tentative_until_confirmation_span(self):
        policy = VehicleTrackPolicy()
        for pts in (0, 33, 66):
            record = policy.update([detection()], pts)[0]
            self.assertEqual(record['status'], 'tentative')
        self.assertEqual(record['reason'], 'insufficient_confirmation_span')
        self.assertEqual(policy.update([detection()], 150)[0]['status'], 'qualified')

    def test_confirmation_span_is_configurable_and_validated(self):
        policy = VehicleTrackPolicy(min_confirmation_span_ms=0)
        for pts in (0, 1, 2):
            record = policy.update([detection()], pts)[0]
        self.assertEqual(record['status'], 'qualified')
        for bad in (-1, float('nan'), float('inf'), True, '150'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                VehicleTrackPolicy(min_confirmation_span_ms=bad)

    def test_low_confidence_never_qualifies_but_continues_qualified_track(self):
        policy = VehicleTrackPolicy()
        for pts in range(0, 2000, 100):
            record = policy.update([detection(confidence=.39)], pts)[0]
        self.assertEqual(record['status'], 'tentative')
        self.assertEqual(record['reason'], 'insufficient_recent_high_confidence')
        policy.update([detection()], 2000)
        record = policy.update([detection()], 2100)[0]
        self.assertEqual(record['status'], 'qualified')
        for pts in range(2200, 3400, 100):
            record = policy.update([detection(confidence=.01)], pts)[0]
        self.assertEqual(record['status'], 'qualified')
        self.assertEqual(record['high_confidence_hits'], 0)
        self.assertEqual(record['high_confidence_hits_total'], 2)

    def test_high_confidence_gate_uses_last_ten_observations(self):
        policy = VehicleTrackPolicy()
        policy.update([detection()], 0)
        for pts in range(100, 1100, 100):
            policy.update([detection(confidence=.1)], pts)
        record = policy.update([detection()], 1100)[0]
        self.assertEqual(record['high_confidence_hits_total'], 2)
        self.assertEqual(record['high_confidence_hits'], 1)
        self.assertEqual(record['status'], 'tentative')
        self.assertEqual(policy.update([detection()], 1200)[0]['status'], 'qualified')

    def test_gap_exact_boundary_retains_binding_then_expires_after_1000_ms(self):
        policy = VehicleTrackPolicy()
        original = policy.update([detection()], 0)[0]['passage_id']
        self.assertEqual(policy.update([detection()], 1000)[0]['passage_id'], original)
        self.assertEqual(policy.update([], 2000), [])
        self.assertFalse(policy.records[0]['ended'])
        replacement = policy.update([detection()], 2001)[0]
        self.assertEqual(replacement['passage_id'], 'V002')
        self.assertEqual(replacement['hits'], 1)
        self.assertEqual(policy.records[0]['end_reason'], 'gap')
        self.assertEqual(policy.counters['ended_gap'], 1)

    def test_class_change_keeps_passage_but_discontinuity_ends_it(self):
        policy = VehicleTrackPolicy()
        for pts in (0, 100, 150):
            original = policy.update([detection()], pts)[0]
        self.assertEqual(original['status'], 'qualified')
        second = policy.update([detection(vehicle_class=7)], 200)[0]
        self.assertEqual(second['passage_id'], 'V001')
        self.assertEqual(second['status'], 'qualified')
        self.assertEqual(second['observed_class'], 7)
        self.assertEqual(second['vehicle_class'], 2)
        for pts in (300, 350):
            second = policy.update([detection(vehicle_class=7)], pts)[0]
        self.assertEqual(second['status'], 'qualified')
        third = policy.update([detection(vehicle_class=7, box=[500, 0, 600, 50])], 400)[0]
        self.assertEqual(third['passage_id'], 'V002')
        self.assertEqual(third['status'], 'tentative')
        self.assertEqual([r['end_reason'] for r in policy.records], ['discontinuity', None])

    def test_car_truck_flicker_keeps_passage_and_qualifies(self):
        policy = VehicleTrackPolicy()
        for index, label in enumerate((2, 7, 2, 7, 2, 7, 2, 7, 2, 7)):
            current = policy.update([detection(vehicle_class=label)], index * 100)[0]
            self.assertEqual(current['passage_id'], 'V001')
            self.assertEqual(current['vehicle_class'], 2)
            self.assertEqual(current['observed_class'], label)
        self.assertEqual(current['status'], 'qualified')
        self.assertTrue(current['class_ambiguous'])
        self.assertAlmostEqual(current['class_vote_share'], .5)
        self.assertEqual(len(policy.records), 1)

    def test_confidence_weighted_class_dominance_updates_label_without_new_passage(self):
        policy = VehicleTrackPolicy()
        policy.update([detection(vehicle_class=2, confidence=.2)], 0)
        current = policy.update([detection(vehicle_class=7, confidence=.8)], 100)[0]
        self.assertEqual(current['vehicle_class'], 7)
        self.assertFalse(current['class_ambiguous'])
        self.assertAlmostEqual(current['class_vote_share'], .8)
        self.assertEqual(current['passage_id'], 'V001')
        self.assertEqual(current['hits'], 2)

    def test_weighted_class_tie_retains_previous_stable_label(self):
        policy = VehicleTrackPolicy()
        policy.update([detection(vehicle_class=7, confidence=.8)], 0)
        current = policy.update([detection(vehicle_class=2, confidence=.8)], 100)[0]
        self.assertEqual(current['vehicle_class'], 7)
        self.assertEqual(current['observed_class'], 2)
        self.assertTrue(current['class_ambiguous'])
        self.assertEqual(current['class_vote_share'], .5)

    def test_class_votes_use_only_last_ten_observations(self):
        policy = VehicleTrackPolicy()
        policy.update([detection(vehicle_class=2, confidence=1.)], 0)
        for pts in range(100, 1100, 100):
            current = policy.update([detection(vehicle_class=7, confidence=.05)], pts)[0]
            if pts < 1000:
                self.assertEqual(current['vehicle_class'], 2)
        self.assertEqual(current['vehicle_class'], 7)
        self.assertAlmostEqual(current['class_vote_share'], 1.)
        self.assertFalse(current['class_ambiguous'])
        self.assertEqual(current['passage_id'], 'V001')

    def test_class_ambiguity_at_sixty_percent_and_zero_confidence(self):
        policy = VehicleTrackPolicy()
        zero = policy.update([detection(confidence=0.)], 0)[0]
        self.assertTrue(zero['class_ambiguous'])
        self.assertEqual(zero['class_vote_share'], 0.)
        for pts, label in zip((100, 200, 300, 400, 500), (7, 7, 2, 2, 2)):
            current = policy.update([detection(vehicle_class=label)], pts)[0]
        self.assertEqual(current['vehicle_class'], 2)
        self.assertAlmostEqual(current['class_vote_share'], .6)
        self.assertFalse(current['class_ambiguous'])

    def test_simultaneous_tracks_remain_independent_and_missing_boxes_are_not_shown(self):
        policy = VehicleTrackPolicy()
        observations = [detection('a'), detection('b', confidence=.1)]
        for pts in (0, 100, 150):
            current = policy.update(observations, pts)
        self.assertEqual([r['passage_id'] for r in current], ['V001', 'V002'])
        self.assertEqual([r['status'] for r in current], ['qualified', 'tentative'])
        visible = policy.update([detection('b', confidence=.1)], 200)
        self.assertEqual([r['native_id'] for r in visible], ['b'])
        self.assertEqual(policy.update([], 250), [])
        self.assertEqual(policy.records[0]['last_pts_ms'], 150)

    def test_duplicate_ids_are_all_rejected_and_never_supply_votes(self):
        policy = VehicleTrackPolicy()
        self.assertEqual(policy.update([detection(), detection()], 0), [])
        self.assertEqual(policy.records, [])
        self.assertEqual(policy.counters['rejected_duplicate_native_id'], 2)
        policy.update([detection()], 100)
        invalid_duplicate = detection(confidence=float('nan'))
        self.assertEqual(policy.update([detection(), invalid_duplicate], 200), [])
        record = policy.update([detection()], 300)[0]
        self.assertEqual(record['hits'], 2)
        self.assertEqual(record['status'], 'tentative')

    def test_motor_vehicle_classes_only(self):
        policy = VehicleTrackPolicy()
        current = policy.update([detection(str(cls), vehicle_class=cls) for cls in range(9)], 0)
        self.assertEqual([r['vehicle_class'] for r in current], [2, 3, 5, 7])
        self.assertEqual(policy.counters['rejected_unsupported_class'], 5)

    def test_numeric_and_shape_validation(self):
        cases = [
            ('invalid_confidence', {'confidence': value})
            for value in (float('nan'), float('inf'), -.1, 1.1, True, '.5', None)
        ] + [
            ('invalid_box', {'box': value}) for value in (
                [], [0, 0, 1], [0, 0, 0, 1], [0, 0, 1, -1],
                [0, 0, float('inf'), 1], [0, 0, float('nan'), 1],
                [False, 0, 1, 1], [0, 0, '1', 1], [0, 0, 1e308, 1e308],
                [10**20, 0, 10**20 + 1, 1], None)
        ] + [
            ('invalid_class', {'vehicle_class': value}) for value in (True, 2., '2', None)
        ] + [('invalid_native_id', {'native_id': value}) for value in ('', ' ', 1, None)]
        for reason, overrides in cases:
            with self.subTest(overrides=overrides):
                policy = VehicleTrackPolicy()
                invalid = detection()
                invalid.update(overrides)
                self.assertEqual(policy.update([invalid], 0), [])
                self.assertEqual(policy.counters[f'rejected_{reason}'], 1)
                self.assertEqual(policy.records, [])
        policy = VehicleTrackPolicy()
        self.assertEqual(policy.update([None, {}, []], 0), [])
        self.assertEqual(policy.counters['rejected_invalid_detection'], 2)

    def test_nonforward_and_invalid_pts_are_rejected_without_votes(self):
        policy = VehicleTrackPolicy()
        policy.update([detection()], 100)
        for pts in (100, 99, -1, float('nan'), float('inf'), True, '200', None):
            with self.subTest(pts=pts), self.assertRaises(ValueError):
                policy.update([detection()], pts)
        self.assertEqual(policy.records[0]['hits'], 1)
        self.assertEqual(policy.counters['rejected_timestamp'], 8)
        self.assertEqual(policy.update([detection()], 200)[0]['hits'], 2)

    def test_eof_and_detached_serializable_records_with_no_verified_designation(self):
        policy = VehicleTrackPolicy()
        for pts in (0, 100, 150):
            visible = policy.update([detection()], pts)
        visible[0]['box'][0] = 999
        self.assertEqual(policy.records[0]['box'][0], 0)
        records = policy.finish(150)
        self.assertTrue(records[0]['ended'])
        self.assertEqual(records[0]['end_reason'], 'eof')
        self.assertEqual(records[0]['last_pts_ms'], 150)
        self.assertEqual(records[0]['end_pts_ms'], 150)
        self.assertNotIn('verified', json.dumps(records, allow_nan=False))
        records[0]['status'] = 'changed externally'
        self.assertEqual(policy.records[0]['status'], 'qualified')
        self.assertEqual(policy.finish(150), policy.records)
        self.assertEqual(policy.counters['ended_eof'], 1)
        with self.assertRaises(ValueError):
            policy.update([detection()], 200)
        with self.assertRaises(ValueError):
            policy.finish(149)


if __name__ == '__main__':
    unittest.main()
