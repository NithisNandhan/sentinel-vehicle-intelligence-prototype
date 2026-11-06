"""Passages, result states and plate-level tracking inside TrackedOCR.

All OCR output here is SYNTHETIC (ctc_fixtures). These tests prove the plumbing
and the safety rules, not recognition accuracy.
"""
from types import SimpleNamespace
import unittest

import numpy as np

from ctc_fixtures import TOKENS, rows_for
from sentinel_edge.experimental_detectors import VehicleTrack
from sentinel_edge.plate_options import PlateOptions
from sentinel_edge.result_state import CONFIRMED, PARTIAL, UNREADABLE
from sentinel_edge.tracked_ocr import TrackedOCR

PLATE = 'MH49BE5557'
CANDIDATE = PlateOptions.from_env({'SENTINEL_PROFILE': 'candidate'})


class Scene:
    """A synthetic camera: each frame's crop differs visibly unless `still` is set."""
    def __init__(self, specs, *, plate_box=(5, 5, 150, 35), options=CANDIDATE):
        self.specs = list(specs)
        self.calls = 0
        self.outcomes, self.accepted = [], []
        def probabilities(crop):
            spec = self.specs[min(self.calls, len(self.specs) - 1)]
            self.calls += 1
            return rows_for(spec)
        self.engine = TrackedOCR(SimpleNamespace(tokens=TOKENS, probabilities=probabilities),
                                 SimpleNamespace(box=lambda image: plate_box), options=options,
                                 on_finalized=self.outcomes.append, keep_evidence=True,
                                 on_accepted=lambda track, candidate, pts: self.accepted.append((track, candidate.plate)))

    @staticmethod
    def image(seed, still=False):
        rng = np.random.default_rng(0 if still else seed)
        image = rng.integers(0, 255, (60, 180, 3), dtype=np.uint8)
        if still:
            image[10, 10] = seed % 255  # Different bytes inside the plate crop, same view.
        return image


def track(track_id='1', box=(0, 0, 180, 60), vehicle_class=2):
    return VehicleTrack(track_id, box, vehicle_class)


class PassageTests(unittest.TestCase):
    def test_confirmed_passage_is_reported_once_when_it_ends(self):
        scene = Scene([list(PLATE)] * 3)
        for i in range(3):
            scene.engine.process(scene.image(i), [track()], i * 500)
        self.assertEqual([plate for _, plate in scene.accepted], [PLATE])
        self.assertEqual(scene.outcomes, [])
        scene.engine.process(scene.image(9), [], 5000)  # Gap expires the passage.
        self.assertEqual(len(scene.outcomes), 1)
        outcome = scene.outcomes[0]
        self.assertEqual((outcome.result.state, outcome.cause, outcome.confirmed_plate), (CONFIRMED, 'expired', PLATE))
        self.assertTrue(outcome.evidence and outcome.frame_jpeg)
        self.assertLessEqual(len(outcome.evidence), 3)
        self.assertIsNotNone(outcome.indicators)

    def test_deletion_path_accepted_by_old_gate_is_not_delivered(self):
        scene = Scene([list('MH49BE555O')] * 3)
        for i in range(3):
            scene.engine.process(scene.image(i), [track()], i * 500)
        self.assertEqual(scene.accepted, [], 'Result states must block the length-disagreeing reading')
        scene.engine.flush()
        self.assertNotEqual(scene.outcomes[0].result.state, CONFIRMED)
        self.assertGreater(sum(v for k, v in scene.engine.metrics.items() if k.startswith('confirmation_blocked_')), 0)
        baseline = Scene([list('MH49BE555O')] * 3, options=PlateOptions())
        for i in range(3):
            baseline.engine.process(baseline.image(i), [track()], i * 500)
        self.assertEqual([plate for _, plate in baseline.accepted], ['MH49BE555'],
                         'Baseline profile keeps the existing behaviour (documented risk)')

    def test_partial_and_unreadable_passages(self):
        weak = list(PLATE)
        weak[5] = ('E', .45, 'F')
        scene = Scene([weak, weak, list('MH49BF5557')])
        for i in range(3):
            scene.engine.process(scene.image(i), [track()], i * 500)
        scene.engine.flush()
        self.assertEqual(scene.outcomes[0].result.state, PARTIAL)
        self.assertEqual(scene.outcomes[0].result.pattern, 'MH49B?5557')
        none = Scene([list(PLATE)])
        none.engine.detector = SimpleNamespace(box=lambda image: None)
        for i in range(3):
            none.engine.process(none.image(i), [track()], i * 500)
        none.engine.flush()
        self.assertEqual((none.outcomes[0].result.state, none.outcomes[0].result.reason), (UNREADABLE, 'no_plate_found'))

    def test_small_plate_reason_and_transient_tracks(self):
        scene = Scene([list(PLATE)], plate_box=(5, 5, 20, 10))
        for i in range(2):
            scene.engine.process(scene.image(i), [track()], i * 500)
        scene.engine.process(scene.image(3), [track('2', (0, 0, 120, 50))], 1500)
        scene.engine.flush()
        reasons = [o.result.reason for o in scene.outcomes]
        self.assertEqual(reasons, ['plate_too_small'])
        self.assertEqual(scene.engine.metrics['transient_tracks_not_reported'], 1)

    def test_tracker_id_switch_keeps_one_passage(self):
        scene = Scene([list(PLATE)] * 4)
        scene.engine.process(scene.image(0), [track('1', (0, 0, 180, 60))], 0)
        scene.engine.process(scene.image(1), [track('1', (4, 0, 184, 60))], 500)
        # The tracker loses the vehicle for one sample, then gives it a new ID.
        scene.engine.process(scene.image(2), [], 1000)
        scene.engine.process(scene.image(3), [track('7', (8, 0, 188, 60))], 1500)
        scene.engine.flush()
        self.assertEqual(scene.engine.metrics['track_reassociations'], 1)
        self.assertEqual(len(scene.outcomes), 1)
        self.assertEqual(scene.outcomes[0].track_ids, ('1', '7'))
        self.assertEqual(scene.outcomes[0].result.state, CONFIRMED)

    def test_ambiguous_reassociation_never_mixes_vehicles(self):
        scene = Scene([list(PLATE)] * 6)
        scene.engine.process(scene.image(0), [track('1', (0, 0, 180, 60)), track('2', (10, 0, 190, 60))], 0)
        scene.engine.process(scene.image(1), [track('3', (5, 0, 185, 60))], 500)
        self.assertEqual(scene.engine.metrics['track_reassociations'], 0)
        other_class = Scene([list(PLATE)] * 6)
        other_class.engine.process(other_class.image(0), [track('1', vehicle_class=2)], 0)
        other_class.engine.process(other_class.image(1), [track('9', vehicle_class=3)], 500)
        self.assertEqual(other_class.engine.metrics['track_reassociations'], 0)

    def test_near_identical_frames_are_not_independent_confirmations(self):
        scene = Scene([list(PLATE)] * 5)
        for i in range(5):
            scene.engine.process(scene.image(i, still=True), [track()], i * 500)
        self.assertEqual(scene.engine.histories['1'].reader.observation_count, 1)
        self.assertEqual(scene.accepted, [])
        self.assertGreaterEqual(scene.engine.metrics['near_duplicate_crops'], 4)

    def test_adaptive_sampling_signal(self):
        scene = Scene([list(PLATE)] * 3)
        scene.engine.process(scene.image(0), [track()], 0)
        self.assertTrue(scene.engine.needs_more_evidence())
        for i in (1, 2):
            scene.engine.process(scene.image(i), [track()], i * 500)
        self.assertFalse(scene.engine.needs_more_evidence())

    def test_pts_reversal_finalizes_open_passages(self):
        scene = Scene([list(PLATE)] * 3)
        scene.engine.process(scene.image(0), [track()], 1000)
        scene.engine.process(scene.image(1), [track()], 1500)
        scene.engine.process(scene.image(2), [track()], 200)
        self.assertEqual([o.cause for o in scene.outcomes], ['reset'])

    def test_baseline_profile_reports_nothing(self):
        scene = Scene([list(PLATE)] * 3, options=PlateOptions())
        for i in range(3):
            scene.engine.process(scene.image(i), [track()], i * 500)
        scene.engine.flush()
        self.assertEqual(scene.outcomes, [])


class ProfileTests(unittest.TestCase):
    def test_profiles_and_overrides(self):
        self.assertEqual(PlateOptions.from_env({}).manifest()['profile'], 'baseline')
        self.assertFalse(any(v for k, v in PlateOptions.from_env({}).manifest().items() if k != 'profile'))
        candidate = PlateOptions.from_env({'SENTINEL_PROFILE': 'candidate'})
        self.assertTrue(candidate.result_states and candidate.plate_level_tracking)
        self.assertFalse(candidate.multi_source_plates or candidate.secondary_veto or candidate.partial_watchlist)
        self.assertTrue(PlateOptions.from_env({'SENTINEL_PROFILE': 'candidate-gpu'}).multi_source_plates)
        rolled_back = PlateOptions.from_env({'SENTINEL_PROFILE': 'candidate', 'SENTINEL_PLATE_LEVEL_TRACKING': 'false'})
        self.assertFalse(rolled_back.plate_level_tracking)
        with self.assertRaises(ValueError):
            PlateOptions.from_env({'SENTINEL_PROFILE': 'experimental'})


if __name__ == '__main__':
    unittest.main()
