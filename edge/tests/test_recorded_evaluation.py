"""Scoring rules on hand-built cases, plus one run with a MOCKED recogniser.

These tests check that the arithmetic and denominators are right. They are not
a measurement of the real models.
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from sentinel_edge.recorded_evaluation import (
    LabelError, Passage, TrackOutcome, analyse_recording, evaluate, iou, load_labels, merge, normalise, score)
from sentinel_edge.recorded_source import RecordingLibrary

from recorded_fixtures import make_video, register
from test_recorded_session import TEXT, fakes

PLATE, OTHER = 'GJ01AB1234', 'MH02EK0837'


def passage(pid='p1', start=1000, end=4000, plate=PLATE, readable=True, boxes=(), visible=None, vehicle='v1'):
    return Passage(pid, vehicle, start, end, plate if readable else None, readable,
                   readable if visible is None else visible, tuple(boxes))


def track(first=1500, last=3500, *, accepted=(), candidates=(), observations=3, vehicle=(), plate=()):
    return TrackOutcome(first, last, list(vehicle), list(plate), observations, list(candidates), list(accepted))


class ScoringTests(unittest.TestCase):
    def test_a_correct_accepted_reading(self):
        result = score([passage()], [track(accepted=[PLATE], candidates=[PLATE, OTHER])])
        self.assertEqual(result['exact_read_coverage'], {'count': 1, 'of': 1, 'rate': 1.})
        self.assertEqual(result['accepted_precision'], {'count': 1, 'of': 1, 'rate': 1.})
        self.assertEqual(result['candidate_recall']['top_1']['count'], 1)
        self.assertEqual(result['per_passage'][0]['outcome'], 'correct')
        self.assertIsNone(result['per_passage'][0]['failing_stage'])

    def test_precision_is_undefined_not_zero_or_perfect_when_nothing_is_accepted(self):
        result = score([passage()], [track(candidates=[OTHER, PLATE])])
        self.assertIsNone(result['accepted_precision']['rate'])
        self.assertIn('undefined', result['accepted_precision']['note'])
        self.assertEqual(result['abstentions'], {'count': 1, 'of': 1, 'rate': 1.})
        self.assertEqual(result['exact_read_coverage']['rate'], 0.)
        row = result['per_passage'][0]
        self.assertEqual((row['outcome'], row['true_plate_rank']), ('abstained', 2))
        self.assertEqual(row['failing_stage'], 'acceptance_gate_withheld_rank_2')
        self.assertEqual((result['candidate_recall']['top_1']['count'], result['candidate_recall']['top_5']['count']), (0, 1))

    def test_a_wrong_accepted_reading_counts_against_precision(self):
        result = score([passage()], [track(accepted=[OTHER], candidates=[OTHER, PLATE])])
        self.assertEqual(result['wrong_reads']['count'], 1)
        self.assertEqual(result['accepted_precision'], {'count': 0, 'of': 1, 'rate': 0.})
        self.assertEqual(result['per_passage'][0]['failing_stage'], 'fusion_or_association')

    def test_each_failure_is_attributed_to_its_first_failing_stage(self):
        box = {'position_ms': 2000, 'vehicle': (100., 100., 300., 300.), 'plate': None}
        cases = [
            ([], 'vehicle_detection_or_tracking'),
            ([track(vehicle=[(2000, (600, 600, 800, 800))])], 'vehicle_detection_or_tracking'),
            ([track(observations=0, vehicle=[(2000, (100, 100, 300, 300))])], 'plate_localisation_or_crop'),
            ([track(candidates=[OTHER], vehicle=[(2000, (100, 100, 300, 300))])], 'ocr_candidate_missing'),
        ]
        for tracks, expected in cases:
            row = score([passage(boxes=[box])], tracks)['per_passage'][0]
            self.assertEqual(row['failing_stage'], expected)

    def test_vehicle_and_plate_boxes_use_the_declared_overlap_rule(self):
        boxes = [{'position_ms': 2000, 'vehicle': (100., 100., 300., 300.), 'plate': (150., 250., 250., 280.)},
                 {'position_ms': 3000, 'vehicle': None, 'plate': (160., 250., 260., 280.)}]
        good = track(vehicle=[(2300, (110, 100, 300, 300))], plate=[(2300, (150, 250, 250, 280))])
        result = score([passage(boxes=boxes)], [good])
        self.assertEqual(result['vehicle_coverage'], {'count': 1, 'of': 1, 'rate': 1.})
        self.assertEqual(result['plate_localisation'], {'count': 1, 'of': 2, 'rate': .5})  # 3000 ms has no sample within 500 ms
        late = track(vehicle=[(2600, (110, 100, 300, 300))])
        self.assertEqual(score([passage(boxes=boxes)], [late])['vehicle_coverage']['count'], 0)
        # Without labelled boxes there is no vehicle-coverage claim, only time overlap.
        unboxed = score([passage()], [track()])
        self.assertEqual(unboxed['vehicle_coverage'], {'count': 0, 'of': 0, 'rate': None})
        self.assertEqual(unboxed['time_overlap_coverage']['count'], 1)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1/3)
        self.assertEqual(iou((0, 0, 10, 10), (10, 0, 20, 10)), 0.)

    def test_unverifiable_readings_are_reported_but_kept_out_of_precision(self):
        unreadable = passage('p2', 5000, 8000, readable=False, visible=True, vehicle='v2')
        tracks = [track(accepted=[PLATE], candidates=[PLATE]),
                  track(5500, 7500, accepted=[OTHER]),          # On a passage no person could read.
                  track(20_000, 22_000, accepted=['KA03MN4567'])]  # Outside every labelled passage.
        result = score([passage(), unreadable], tracks)
        self.assertEqual(result['accepted_readings'], {'correct': 1, 'wrong': 0,
                                                       'on_passages_no_person_could_read': 1,
                                                       'outside_labelled_passages': 1})
        self.assertEqual(result['accepted_precision'], {'count': 1, 'of': 1, 'rate': 1.})
        self.assertEqual(result['unreadable_passages'], 1)
        self.assertEqual(result['exact_read_coverage']['of'], 1)  # Unreadable passages are not in this denominator.
        self.assertEqual(result['ocr_observation_yield'], {'count': 2, 'of': 2, 'rate': 1.})

    def test_neighbouring_frames_do_not_multiply_a_success(self):
        many = track(accepted=[PLATE, PLATE], candidates=[PLATE], observations=9)
        many.accepted = [PLATE]
        result = score([passage()], [many])
        self.assertEqual((result['exact_read_coverage']['count'], result['accepted_readings']['correct']), (1, 1))

    def test_totals_are_summed_never_averaged(self):
        first = score([passage()], [track(accepted=[PLATE], candidates=[PLATE])])
        second = score([passage(f'p{i}', vehicle=f'w{i}') for i in range(3)], [])
        totals = merge([first, second])
        self.assertEqual(totals['exact_read_coverage'], {'count': 1, 'of': 4, 'rate': .25})  # Not (1.0 + 0.0) / 2.
        self.assertEqual(totals['passages'], 4)
        self.assertEqual(totals['accepted_precision']['rate'], 1.)
        self.assertIsNone(merge([second])['accepted_precision']['rate'])

    def test_report_rows_carry_masked_plates_only(self):
        result = score([passage()], [track(accepted=[OTHER], candidates=[OTHER, PLATE])])
        text = json.dumps(result)
        self.assertNotIn(PLATE, text); self.assertNotIn(OTHER, text)
        self.assertEqual(result['per_passage'][0]['labelled_plate_masked'], 'GJ******34')
        self.assertEqual(normalise('gj 01-ab 1234'), PLATE)


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'labels.json'

    def write(self, **changes):
        entry = {'recording_id': 'rec-0123456789ab', 'sha256': '0123456789ab'+'0'*52, 'split': 'development',
                 'passages': [{'passage_id': 'p1', 'vehicle_id': 'v1', 'start_ms': 1000, 'end_ms': 4000,
                               'plate': 'GJ 01 AB 1234', 'readable': True,
                               'boxes': [{'position_ms': 2000, 'vehicle': [100, 100, 300, 300], 'plate': [150, 250, 250, 280]}]},
                              {'passage_id': 'p2', 'vehicle_id': 'v2', 'start_ms': 5000, 'end_ms': 6000,
                               'plate': None, 'readable': False}]}
        data = {'schema_version': '1.0', 'labeller': 'A. Reviewer', 'recordings': [entry]}
        for key, value in changes.items():
            if key in data:
                data[key] = value
            elif key == 'more':
                data['recordings'] += value
            else:
                entry['passages'][0][key] = value
        self.path.write_text(json.dumps(data), encoding='utf-8')

    def test_valid_labels_load_for_the_requested_split_only(self):
        self.write()
        labelled, digest = load_labels(self.path, split='development')
        first, second = labelled['rec-0123456789ab']['passages']
        self.assertEqual((first.plate, first.readable, first.plate_visible), ('GJ01AB1234', True, True))
        self.assertEqual((second.plate, second.readable, second.plate_visible), (None, False, False))
        self.assertEqual(len(digest), 64)
        self.assertEqual(load_labels(self.path, split='test')[0], {})

    def test_labels_must_come_from_a_named_person(self):
        self.write(labeller='')
        with self.assertRaisesRegex(LabelError, 'not ground truth'):
            load_labels(self.path, split='development')

    def test_inconsistent_labels_are_rejected(self):
        for change in (dict(plate=None), dict(readable='yes'), dict(start_ms=5000), dict(start_ms=1.5),
                       dict(boxes=[{'position_ms': 9000, 'vehicle': [1, 1, 5, 5]}]),
                       dict(boxes=[{'position_ms': 2000, 'vehicle': [5, 5, 1, 1]}]),
                       dict(plate='AB'), dict(schema_version='2.0')):
            self.write(**change)
            with self.assertRaises(LabelError, msg=str(change)):
                load_labels(self.path, split='development')

    def test_a_vehicle_may_not_appear_in_two_splits(self):
        leak = {'recording_id': 'rec-ffffffffffff', 'sha256': 'f'*64, 'split': 'test',
                'passages': [{'passage_id': 'p1', 'vehicle_id': 'v1', 'start_ms': 0, 'end_ms': 900,
                              'plate': 'GJ01AB1234', 'readable': True}]}
        self.write(more=[leak])
        with self.assertRaisesRegex(LabelError, 'more than one split'):
            load_labels(self.path, split='development')


class EvaluationRunTests(unittest.TestCase):
    """The whole path from file to report, with the recogniser mocked."""
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        (self.root/'recordings').mkdir()
        register(make_video(self.root/'recordings'/'clip.mp4', frames=60, fps=10))
        self.recording = next(iter(RecordingLibrary(self.root/'recordings').scan()[0].values()))
        self.config = SimpleNamespace(ocr_min_confidence=.85)

    def labels(self, plate=TEXT, **entry):
        data = {'schema_version': '1.0', 'labeller': 'A. Reviewer', 'recordings': [dict({
            'recording_id': self.recording.recording_id, 'sha256': self.recording.sha256, 'split': 'development',
            'passages': [{'passage_id': 'p1', 'vehicle_id': 'v1', 'start_ms': 0, 'end_ms': 5900, 'plate': plate,
                          'readable': True, 'boxes': [{'position_ms': 2000, 'vehicle': [0, 0, 160, 120],
                                                       'plate': [5, 5, 150, 35]}]}]}, **entry)]}
        path = self.root/'labels.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        return path

    def test_every_sample_is_analysed_and_the_tracks_are_collected(self):
        tracks, runtime = analyse_recording(self.config, self.recording, *fakes(ocr_delay=.02), detector_profile='general')
        self.assertEqual(runtime['analysed_frames'], 12)
        self.assertTrue(runtime['reached_end'])
        self.assertEqual(runtime['video_seconds'], 6.)
        self.assertEqual(len(tracks), 1)
        self.assertEqual((tracks[0].first_ms, tracks[0].last_ms), (0, 5500))
        self.assertEqual(tracks[0].accepted, [TEXT])
        self.assertEqual(tracks[0].candidates[0], TEXT)
        self.assertEqual(len(tracks[0].vehicle_boxes), 12)

    def test_report_scores_against_the_labels_and_hides_full_plates(self):
        report = evaluate(self.labels(), 'development', self.root/'recordings', self.config, fakes(),
                          detector_profile='general')
        totals = report['totals']
        self.assertEqual(totals['exact_read_coverage'], {'count': 1, 'of': 1, 'rate': 1.})
        self.assertEqual(totals['vehicle_coverage']['count'], 1)
        self.assertEqual(totals['plate_localisation'], {'count': 1, 'of': 1, 'rate': 1.})
        self.assertEqual(report['settings']['acceptance_threshold'], .85)
        self.assertEqual(report['settings']['frames_dropped_by_design'], 0)
        self.assertEqual(len(report['labels_sha256']), 64)
        self.assertNotIn(TEXT, json.dumps(report))
        self.assertNotIn(str(self.root), json.dumps(report))

    def test_a_label_the_mock_does_not_produce_scores_as_a_wrong_read(self):
        report = evaluate(self.labels(plate='GJ01AB1234'), 'development', self.root/'recordings', self.config,
                          fakes(), detector_profile='general')
        self.assertEqual(report['totals']['exact_read_coverage']['count'], 0)
        self.assertEqual(report['totals']['wrong_reads']['count'], 1)
        self.assertEqual(report['totals']['accepted_precision']['rate'], 0.)

    def test_labels_for_a_missing_or_different_file_stop_the_run(self):
        with self.assertRaisesRegex(LabelError, 'different file'):
            evaluate(self.labels(sha256=self.recording.sha256[:12]+'f'*52), 'development', self.root/'recordings',
                     self.config, fakes(), detector_profile='general')
        with self.assertRaisesRegex(LabelError, 'No recordings are labelled'):
            evaluate(self.labels(), 'validation', self.root/'recordings', self.config, fakes(), detector_profile='general')
        with self.assertRaisesRegex(LabelError, 'not present'):
            evaluate(self.labels(), 'development', self.root/'elsewhere', self.config, fakes(), detector_profile='general')
