"""Opt-in motion linking for vehicles ByteTrack has not confirmed (often fast motorcycles)."""
from types import SimpleNamespace
import unittest

import numpy as np

from sentinel_edge.experimental_detectors import VehicleTracker
from sentinel_edge.vision import associate_vehicle_tracks

CAR, MOTORCYCLE, TRUCK = 2, 3, 7


def bike(x, y):
    """A 60 x 130 motorcycle box; between 500 ms samples it moves ~90 px, with no overlap kept."""
    return (x, y, x + 60, y + 130)


class Sequence:
    """Feed frames through the association, keeping its state like VehicleTracker does."""
    def __init__(self, motion=True):
        self.previous, self.next_id, self.frame = {}, 0, 0
        self.aliases = {} if motion else None

    def step(self, detections):
        self.frame += 1
        tracks, self.previous, self.next_id = associate_vehicle_tracks(
            detections, self.previous, self.frame, self.next_id, aliases=self.aliases)
        return {box: track_id for track_id, box, _ in tracks}


class MotionLinkingTests(unittest.TestCase):
    def test_off_by_default_keeps_the_previous_behaviour(self):
        run = Sequence(motion=False)
        first = run.step([(None, bike(100, 100), MOTORCYCLE)])[bike(100, 100)]
        second = run.step([(None, bike(100, 190), MOTORCYCLE)])[bike(100, 190)]
        self.assertNotEqual(first, second)  # No overlap, so a new ID, exactly as before.

    def test_a_fast_bike_keeps_one_id_between_samples(self):
        run = Sequence()
        ids = {run.step([(None, bike(100, 100 + 90 * step), MOTORCYCLE)])[bike(100, 100 + 90 * step)]
               for step in range(4)}
        self.assertEqual(len(ids), 1)

    def test_two_nearby_bikes_never_merge_or_swap(self):
        # At x=140 the boxes touch, and the wrong pairing is also within the motion rule:
        # only best-pairs-first assignment keeps them apart, whatever the detection order.
        for right_x in (175, 140):
            for reverse in (False, True):
                run = Sequence()
                trails = {'left': [], 'right': []}
                for step in range(4):
                    y = 100 + 90 * step
                    rows = [(None, bike(100, y), MOTORCYCLE), (None, bike(right_x, y), MOTORCYCLE)]
                    shown = run.step(rows[::-1] if reverse else rows)
                    self.assertEqual(len(set(shown.values())), 2)
                    trails['left'].append(shown[bike(100, y)])
                    trails['right'].append(shown[bike(right_x, y)])
                self.assertEqual(len(set(trails['left'])), 1, (right_x, reverse))
                self.assertEqual(len(set(trails['right'])), 1, (right_x, reverse))
                self.assertNotEqual(trails['left'][0], trails['right'][0])

    def test_side_by_side_cars_get_distinct_ids_and_no_duplicates(self):
        run = Sequence()
        left, right = (100, 300, 300, 420), (320, 300, 520, 420)
        for step in range(3):
            moved_left = (left[0], left[1] + 20 * step, left[2], left[3] + 20 * step)
            moved_right = (right[0], right[1] + 20 * step, right[2], right[3] + 20 * step)
            shown = run.step([('1', moved_left, CAR), (None, moved_right, CAR)])
            self.assertEqual(len(shown), 2)
            self.assertEqual(len(set(shown.values())), 2)
            self.assertEqual(shown[moved_left], '1')

    def test_confirmation_keeps_the_id_first_shown(self):
        run = Sequence()
        first = run.step([(None, bike(100, 100), MOTORCYCLE)])[bike(100, 100)]
        self.assertTrue(first.startswith('det-'))
        self.assertEqual(run.step([('7', bike(100, 190), MOTORCYCLE)])[bike(100, 190)], first)
        self.assertEqual(run.step([('7', bike(100, 280), MOTORCYCLE)])[bike(100, 280)], first)

    def test_a_new_bytetrack_id_never_takes_over_another_bytetrack_id(self):
        run = Sequence()
        run.step([('3', bike(100, 100), MOTORCYCLE)])
        self.assertEqual(run.step([('9', bike(100, 190), MOTORCYCLE)])[bike(100, 190)], '9')

    def test_a_returning_bytetrack_id_never_duplicates_a_reused_id(self):
        run = Sequence()
        shown = run.step([(None, bike(100, 100), MOTORCYCLE)])[bike(100, 100)]
        run.step([('4', bike(100, 190), MOTORCYCLE)])           # '4' confirms and keeps the first ID.
        run.step([('8', bike(100, 280), MOTORCYCLE)])           # '4' lost; '8' continues the bike, '4' alias dropped.
        both = run.step([('4', bike(100, 370), MOTORCYCLE), ('8', bike(300, 370), MOTORCYCLE)])  # '4' returns.
        self.assertEqual(len(set(both.values())), 2)
        self.assertIn(shown, both.values())

    def test_an_implausible_jump_is_not_linked(self):
        run = Sequence()
        first = run.step([(None, bike(100, 100), MOTORCYCLE)])[bike(100, 100)]
        self.assertNotEqual(run.step([(None, bike(100, 400), MOTORCYCLE)])[bike(100, 400)], first)

    def test_classes_never_link(self):
        run = Sequence()
        first = run.step([(None, bike(100, 100), MOTORCYCLE)])[bike(100, 100)]
        self.assertNotEqual(run.step([(None, bike(100, 150), CAR)])[bike(100, 150)], first)


class _Tensor:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def cpu(self):
        return self

    def tolist(self):
        return list(self.value)

    def __getitem__(self, index):
        return self


def _box(track_id, box, cls, score):
    return SimpleNamespace(cls=_Tensor(cls), xyxy=_Tensor(box), conf=_Tensor(score),
                           id=_Tensor(track_id) if track_id is not None else None)


def _tracker(keep, confirmed, held_back):
    tracker = object.__new__(VehicleTracker)
    tracker.name, tracker.device = 'yolo26s', 'cpu'
    tracker.frame_index, tracker.next_detection_id, tracker.previous_tracks = 0, 0, {}
    tracker.aliases = {} if keep else None
    strack = [SimpleNamespace(is_activated=False, cls=float(cls), xyxy=np.array(box, float), score=score)
              for box, cls, score in held_back]
    tracker.model = SimpleNamespace(
        track=lambda *args, **kwargs: [SimpleNamespace(boxes=[_box(*row) for row in confirmed])],
        predictor=SimpleNamespace(trackers=[SimpleNamespace(tracked_stracks=strack)]))
    return tracker


class VehicleTrackerTests(unittest.TestCase):
    def test_held_back_detections_are_ignored_by_default(self):
        tracks = _tracker(False, [(1, (100, 300, 300, 420), CAR, .9)],
                          [((500, 100, 560, 230), MOTORCYCLE, .4)]).track(np.zeros((8, 8, 3), np.uint8))
        self.assertEqual([t.track_id for t in tracks], ['1'])

    def test_a_held_back_bike_is_shown_with_its_detector_score(self):
        tracks = _tracker(True, [(1, (100, 300, 300, 420), CAR, .9)],
                          [((500, 100, 560, 230), MOTORCYCLE, .4)]).track(np.zeros((8, 8, 3), np.uint8))
        self.assertEqual(len(tracks), 2)
        bike_track = next(t for t in tracks if t.vehicle_class == MOTORCYCLE)
        self.assertAlmostEqual(bike_track.confidence, .4)
        self.assertTrue(bike_track.track_id.startswith('det-'))

    def test_a_car_never_gets_a_second_id_from_an_overlapping_held_back_box(self):
        tracks = _tracker(True, [(1, (100, 300, 300, 420), CAR, .9)],
                          [((105, 305, 300, 425), TRUCK, .3), ((102, 298, 298, 418), CAR, .3)]
                          ).track(np.zeros((8, 8, 3), np.uint8))
        self.assertEqual([t.track_id for t in tracks], ['1'])

    def test_a_minimum_height_keeps_unconfirmed_vehicles_only_on_tall_frames(self):
        confirmed, held_back = [(1, (100, 300, 300, 420), CAR, .9)], [((500, 100, 560, 230), MOTORCYCLE, .4)]
        for height, expected in ((720, 1), (2159, 1), (2160, 2)):
            tracker = _tracker(True, confirmed, held_back)
            tracker.unconfirmed_min_height = 2160
            self.assertEqual(len(tracker.track(np.zeros((height, 8, 3), np.uint8))), expected)

    def test_the_minimum_height_setting_defaults_to_every_height_and_fails_loudly(self):
        from unittest.mock import patch
        from sentinel_edge.experimental_detectors import unconfirmed_min_height
        for value, expected in (('', 0), ('2160', 2160)):
            with patch.dict('os.environ', {'SENTINEL_KEEP_UNCONFIRMED_MIN_HEIGHT': value}):
                self.assertEqual(unconfirmed_min_height(), expected)
        for value in ('4k', '-1', '99999'):
            with patch.dict('os.environ', {'SENTINEL_KEEP_UNCONFIRMED_MIN_HEIGHT': value}):
                with self.assertRaises(ValueError):
                    unconfirmed_min_height()


if __name__ == '__main__':
    unittest.main()
