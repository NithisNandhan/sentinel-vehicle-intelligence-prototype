"""Three-state scoring, statistics, annotation validation. SYNTHETIC labels/outcomes only."""
import json
import tempfile
import unittest
from pathlib import Path

from sentinel_edge.annotations import SealedLedger, agreement, manifest, validate
from sentinel_edge.evaluation_stats import error_bound, exact_zero_upper, rule_of_three, wilson
from sentinel_edge.recorded_evaluation import (LabelError, Passage, PassageResult, load_labels, release_gates,
                                               score_states)

PLATE = 'MH49BE5557'


def passage(pid='p', start=0, end=2000, plate=PLATE, readable=True, layout='one_row', conditions=()):
    return Passage(pid, 'v-' + pid, start, end, plate if readable else None, readable, readable, (), layout, tuple(conditions))


def outcome(state, first=500, last=1500, plate=None, pattern=None):
    return PassageResult(first, last, state, plate, pattern)


class StatsTests(unittest.TestCase):
    def test_wilson_matches_the_documented_example(self):
        low, high = wilson(30, 50)
        self.assertAlmostEqual(low, .4618, places=3)
        self.assertEqual(wilson(0, 0), (None, None))
        self.assertAlmostEqual(rule_of_three(50), .06)
        self.assertAlmostEqual(exact_zero_upper(50), .0580, places=3)
        self.assertEqual(error_bound(0, 50)['rule_of_three_upper_95'], .06)
        with self.assertRaises(ValueError):
            wilson(5, 3)


class ScoringTests(unittest.TestCase):
    def test_each_state_is_counted_once_per_passage(self):
        passages = [passage('a'), passage('b', 3000, 5000), passage('c', 6000, 8000), passage('d', 9000, 11000),
                    passage('e', 12000, 14000)]
        outcomes = [outcome('confirmed', plate=PLATE), outcome('confirmed', 500, 900, plate=PLATE),  # fragments of one passage
                    outcome('partial', 3500, 4500, pattern='MH49B?5557'),
                    outcome('partial', 6500, 7500, pattern='MH49B?5587'),   # wrong known character
                    outcome('unreadable', 9500, 10500)]
        report = score_states(passages, outcomes)
        self.assertEqual(report['accepted'], {'correct': 1, 'wrong': 0, 'unsupported_on_unreadable_passages': 0})
        self.assertEqual(report['partial']['passages'], 2)
        self.assertEqual(report['partial']['useful']['count'], 1)
        self.assertEqual(report['partial']['with_wrong_known_character'], 1)
        self.assertEqual(report['partial']['wrong_known_characters']['count'], 1)
        self.assertEqual(report['partial']['known_characters'], 18)
        self.assertEqual(report['unreadable_rate']['count'], 2)
        self.assertEqual(report['readable_passages_without_any_outcome'], 1)
        self.assertEqual(report['coverage']['count'], 1)
        self.assertEqual([r['state'] for r in report['per_passage']],
                         ['confirmed_correct', 'partial_useful', 'partial_wrong_known_character', 'unreadable', 'no_outcome'])

    def test_wrong_confirmed_and_unsupported_confirmed_are_failures(self):
        report = score_states([passage('a'), passage('b', 3000, 5000, readable=False)],
                              [outcome('confirmed', plate='MH49BE5587'), outcome('confirmed', 3500, 4500, plate=PLATE)])
        self.assertEqual(report['accepted']['wrong'], 1)
        self.assertEqual(report['accepted']['unsupported_on_unreadable_passages'], 1)
        self.assertEqual(report['accepted_precision']['rate'], 0.)
        gates = release_gates(report, split='sealed_a', previously_opened=False)
        self.assertFalse(gates['checks']['zero_wrong_confirmed'])

    def test_partial_with_wrong_length_counts_every_known_character_wrong(self):
        report = score_states([passage()], [outcome('partial', pattern='MH49B?557')])
        self.assertEqual(report['partial']['with_wrong_known_character'], 1)

    def test_nothing_confirmed_is_undefined_not_perfect(self):
        report = score_states([passage()], [outcome('unreadable')])
        self.assertIsNone(report['accepted_precision']['rate'])
        self.assertIsNone(report['accepted_error'])

    def test_slices_by_layout_and_condition(self):
        passages = [passage('a', layout='one_row'), passage('b', 3000, 5000, layout='two_row', conditions=('blur',))]
        report = score_states(passages, [outcome('confirmed', plate=PLATE)])
        self.assertEqual(report['slices']['layout:one_row']['coverage']['count'], 1)
        self.assertEqual(report['slices']['layout:two_row']['coverage']['count'], 0)
        self.assertEqual(report['slices']['condition:blur']['readable_passages'], 1)
        self.assertNotIn('condition:night', report['slices'], 'untested conditions are absent, not zero')

    def test_gates_need_a_first_opened_sealed_set_of_fifty(self):
        passages = [passage(f'p{i}', i * 5000, i * 5000 + 2000) for i in range(50)]
        outcomes = [outcome('confirmed', i * 5000 + 500, i * 5000 + 1500, plate=PLATE) for i in range(30)]
        report = score_states(passages, outcomes)
        self.assertTrue(release_gates(report, split='sealed_a', previously_opened=False)['passed'])
        self.assertFalse(release_gates(report, split='sealed_a', previously_opened=True)['passed'])
        self.assertFalse(release_gates(report, split='development', previously_opened=False)['passed'])
        self.assertAlmostEqual(report['coverage']['wilson_95'][0], .4618, places=3)


def labels(recordings, version='1.1'):
    return {'schema_version': version, 'labeller': 'A. Person (synthetic fixture)', 'recordings': recordings}


def recording(rid, split, passages, source='blind_human', sha=None):
    suffix = rid[4:]
    return {'recording_id': rid, 'sha256': sha or suffix + '0' * 52, 'split': split, 'label_source': source, 'passages': passages}


def item(pid, plate, vehicle=None, start=0, end=2000, **extra):
    return dict({'passage_id': pid, 'vehicle_id': vehicle or 'v-' + pid, 'start_ms': start, 'end_ms': end,
                 'readable': plate is not None, 'plate': plate}, **extra)


class AnnotationTests(unittest.TestCase):
    def write(self, data):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'labels.json'
        path.write_text(json.dumps(data))
        return path

    def test_sealed_sets_require_blind_human_labels(self):
        for source in ('human_verified_prefill', 'review'):
            path = self.write(labels([recording('rec-aaaaaaaaaaaa', 'sealed_a', [item('p', PLATE)], source)]))
            with self.assertRaises(LabelError):
                load_labels(path, split='sealed_a')
        with self.assertRaises(LabelError):
            load_labels(self.write(labels([recording('rec-aaaaaaaaaaaa', 'test', [item('p', PLATE)], 'review')])), split='test')
        load_labels(self.write(labels([recording('rec-aaaaaaaaaaaa', 'development', [item('p', PLATE)], 'review')])), split='development')

    def test_validation_flags_leaks_duplicates_and_small_sets(self):
        data = labels([
            recording('rec-111111111111', 'development', [item('a', PLATE), item('b', PLATE, start=1000, end=3000)]),
            recording('rec-222222222222', 'sealed_a', [item('c', PLATE, vehicle='other')]),
        ])
        report = validate(self.write(data))
        text = ' '.join(report['errors'])
        self.assertIn('duplicate passage', text)
        self.assertIn('appear in both development and sealed_a', text)
        self.assertIn('at least 2 recordings', text)
        self.assertIn('at least 50 are required', text)
        self.assertFalse(report['valid'])

    def test_clean_set_and_manifest_hash_is_stable(self):
        data = labels([recording('rec-111111111111', 'development', [item('a', PLATE)]),
                       recording('rec-222222222222', 'development', [item('b', 'KA01AB1234')])])
        path = self.write(data)
        self.assertTrue(validate(path)['valid'])
        self.assertEqual(manifest(path)['manifest_sha256'], manifest(path)['manifest_sha256'])
        self.assertEqual(manifest(path)['recordings'][0]['label_source'], 'blind_human')

    def test_second_check_agreement_and_open_disagreements(self):
        data = labels([recording('rec-111111111111', 'development', [
            item('a', PLATE, second_check={'readable': True, 'plate': PLATE}),
            item('b', PLATE, start=3000, end=4000, second_check={'readable': False}),
            item('c', PLATE, start=5000, end=6000, second_check={'readable': False, 'adjudicated_by': 'lead'}),
            item('d', None, start=7000, end=8000)])])
        result = agreement(data)
        self.assertEqual((result['eligible_passages'], result['checked_passages'], result['agreed']), (4, 3, 1))
        self.assertEqual(result['unresolved_disagreements'], ['rec-111111111111/b'])

    def test_ledger_records_every_opening(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        ledger = SealedLedger(Path(directory.name) / 'ledger.json')
        self.assertFalse(ledger.record('sealed_a', 'h1', commit='c1'))
        self.assertTrue(ledger.record('sealed_a', 'h1', commit='c2'), 'second opening is reported as a reopening')
        self.assertEqual([e['opening_number'] for e in ledger.entries()], [1, 2])
        with self.assertRaises(ValueError):
            ledger.record('development', 'h')


if __name__ == '__main__':
    unittest.main()
