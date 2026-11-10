import copy
import unittest

from sentinel_edge.full_video_review import score_review, validate_review


HASH, SOURCE = 'a'*64, 'b'*64
PLATE, WRONG = 'MH01AB1234', 'MH01AB1230'
MANUAL = 'manual-0123456789abcdef'


def fixture(ids=('track-s0-1',)):
    manifest = dict(schema_version=1, source=dict(sha256=SOURCE, width=3840, height=2160, duration_ms=30933),
                    run=dict(reached_eof=True, decoded_frames=928, analysed_frames=62), models=dict(ocr='fixed'),
                    tracks=[dict(id=sid, first_ms=0, last_ms=1000, plate_crop_count=1) for sid in ids])
    predictions = dict(schema_version=1, manifest_sha256=HASH, source_sha256=SOURCE,
                       tracks={sid:dict(top_guess=PLATE, candidates=[dict(plate=PLATE)], accepted=[],
                                        last_ms=1000, candidate_ms=1000) for sid in ids})
    labels = dict(schema_version=1, manifest_sha256=HASH, labeller='Human reviewer', inventory_complete=True,
                  revision=2, extras=[], labels=[dict(id=sid, status='readable', plate=PLATE,
                  duplicate_of=None, note='') for sid in ids])
    return manifest, predictions, labels


class FullVideoReviewTests(unittest.TestCase):
    def test_reviewer_discarded_extra_is_not_a_detector_false_positive(self):
        manifest, predictions, labels = fixture()
        labels['extras']=[dict(id=MANUAL,position_ms=100,box=[0,0,50,50],image_path='assets/manual.png')]
        labels['labels'].append(dict(id=MANUAL,status='not_vehicle',plate=None))
        report=score_review(manifest,predictions,labels,HASH)
        self.assertEqual(report['counts']['false_auto_tracks'],0)
        self.assertEqual(report['counts']['vehicle_passages'],1)

    def test_track_without_any_ocr_output_has_no_candidate_timestamp(self):
        manifest, predictions, labels = fixture()
        predictions['tracks']['track-s0-1'].update(top_guess=None,candidates=[],candidate_ms=None)
        result=score_review(manifest,predictions,labels,HASH)
        self.assertEqual(result['metrics']['best_guess_exact_coverage']['value'],0.)
        self.assertEqual(result['counts']['readable_with_guess'],0)

    def test_missed_vehicle_stays_in_denominator_and_unknown_is_separate(self):
        manifest, predictions, labels = fixture(('track-s0-1', 'track-s0-2'))
        labels['labels'][1].update(status='unreadable', plate=None)
        labels['extras'] = [dict(id=MANUAL, position_ms=2000, box=[1, 2, 100, 200], image_path='assets/manual.png')]
        labels['labels'].append(dict(id=MANUAL, status='readable', plate='MH02CD5678'))
        report = score_review(manifest, predictions, labels, HASH)
        self.assertEqual(report['metrics']['best_guess_exact_coverage'], dict(numerator=1, denominator=2, value=.5))
        self.assertEqual(report['metrics']['ocr_exact_on_cropped_readable']['value'], 1.)
        self.assertEqual(report['metrics']['vehicle_detection_coverage']['value'], 2/3)
        self.assertEqual(report['counts']['unreadable_passages'], 1)
        self.assertEqual(report['counts']['missed_vehicle_passages'], 1)

    def test_duplicate_chain_uses_latest_candidate_not_whichever_matches_truth(self):
        manifest, predictions, labels = fixture(('track-s0-1', 'track-s0-2', 'track-s0-3'))
        labels['labels'][1].update(status='duplicate', plate=None, duplicate_of='track-s0-1')
        labels['labels'][2].update(status='duplicate', plate=None, duplicate_of='track-s0-2')
        predictions['tracks']['track-s0-2'].update(top_guess=WRONG, candidates=[dict(plate=WRONG), dict(plate=PLATE)], candidate_ms=2000)
        predictions['tracks']['track-s0-3'].update(top_guess=None, candidates=[], candidate_ms=3000)
        report = score_review(manifest, predictions, labels, HASH)
        self.assertEqual(report['counts']['vehicle_passages'], 1)
        self.assertEqual(report['metrics']['best_guess_exact_coverage']['value'], 0.)
        self.assertEqual(report['metrics']['top5_candidate_recall']['value'], 1.)
        self.assertEqual(report['passages'][0]['top_guess'], WRONG)

    def test_earlier_wrong_acceptance_survives_later_correct_reading(self):
        manifest, predictions, labels = fixture()
        predictions['tracks']['track-s0-1']['accepted'] = [dict(plate=WRONG, pts_ms=100), dict(plate=PLATE, pts_ms=200), dict(plate=PLATE, pts_ms=300)]
        report = score_review(manifest, predictions, labels, HASH)
        self.assertEqual(report['metrics']['accepted_exact_coverage']['value'], 1.)
        self.assertEqual(report['metrics']['accepted_precision']['value'], .5)
        self.assertEqual(report['counts']['passages_with_wrong_acceptance'], 1)

    def test_unreadable_accepts_unverifiable_but_false_vehicle_and_no_plate_are_wrong(self):
        manifest, predictions, labels = fixture(('track-s0-1', 'track-s0-2', 'track-s0-3'))
        for label, status in zip(labels['labels'], ('unreadable', 'not_vehicle', 'no_plate')):
            label.update(status=status, plate=None)
        for pred in predictions['tracks'].values(): pred['accepted'] = [PLATE]
        report = score_review(manifest, predictions, labels, HASH)
        self.assertEqual(report['counts']['accepted_unverifiable'], 1)
        self.assertEqual(report['counts']['accepted_wrong'], 2)
        self.assertEqual(report['metrics']['accepted_precision']['value'], 0.)
        self.assertIsNone(report['metrics']['best_guess_exact_coverage']['value'])

    def test_detection_coverage_not_claimed_without_inventory_attestation(self):
        manifest, predictions, labels = fixture()
        labels['inventory_complete'] = False
        report = score_review(manifest, predictions, labels, HASH)
        self.assertIsNone(report['metrics']['vehicle_detection_coverage']['value'])
        self.assertIn('inventory', report['metrics']['vehicle_detection_coverage']['reason'])
        self.assertIsNone(report['metrics']['accepted_precision']['value'])

    def test_complete_review_and_eof_required_before_scoring(self):
        manifest, predictions, labels = fixture()
        labels['labeller'] = ''; labels['labels'] = []
        draft = validate_review(manifest, labels, HASH)
        self.assertEqual(draft['labels'][0]['status'], 'pending')
        with self.assertRaises(ValueError): score_review(manifest, predictions, labels, HASH)
        _, _, labels = fixture(); manifest['run']['reached_eof'] = False
        with self.assertRaises(ValueError): score_review(manifest, predictions, labels, HASH)

    def test_duplicate_cycles_unknown_ids_and_targets_rejected(self):
        manifest, _, labels = fixture(('track-s0-1', 'track-s0-2'))
        for row, target in zip(labels['labels'], ('track-s0-2', 'track-s0-1')):
            row.update(status='duplicate', plate=None, duplicate_of=target)
        with self.assertRaises(ValueError): validate_review(manifest, labels, HASH)
        _, _, labels = fixture(); labels['labels'].append(copy.deepcopy(labels['labels'][0]))
        with self.assertRaises(ValueError): validate_review(manifest, labels, HASH)
        _, _, labels = fixture(); labels['labels'][0]['id'] = 'unseen-track'
        with self.assertRaises(ValueError): validate_review(manifest, labels, HASH)

    def test_binding_and_candidate_consistency_are_checked(self):
        manifest, predictions, labels = fixture()
        predictions['source_sha256'] = 'c'*64
        with self.assertRaises(ValueError): score_review(manifest, predictions, labels, HASH)
        _, predictions, _ = fixture(); predictions['tracks']['track-s0-1']['top_guess'] = WRONG
        with self.assertRaises(ValueError): score_review(manifest, predictions, labels, HASH)

    def test_native_bounds_paths_and_uncertainty_are_not_silently_corrected(self):
        manifest, _, labels = fixture()
        extra = dict(id=MANUAL, position_ms=2000, box=[1, 2, 100, 200], image_path='assets/manual.png')
        for change in (dict(position_ms=40000), dict(box=[0, 0, 4000, 500]), dict(image_path='assets/../secret.png')):
            labels['extras'] = [dict(extra, **change)]
            with self.assertRaises(ValueError): validate_review(manifest, labels, HASH)
        labels['extras'] = []
        labels['labels'][0]['plate'] = 'mh-01 ab 1234'
        self.assertEqual(validate_review(manifest, labels, HASH, True)['labels'][0]['plate'], PLATE)
        labels['labels'][0]['plate'] = 'MH01AB12?4'
        with self.assertRaises(ValueError): validate_review(manifest, labels, HASH, True)


if __name__ == '__main__': unittest.main()
