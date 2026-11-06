"""Calibration arithmetic on SYNTHETIC records."""
import tempfile
import unittest
from pathlib import Path

from sentinel_edge.calibration import (ThresholdConfig, expected_calibration_error, leave_one_recording_out,
                                       report, risk_coverage, strictest_clean_threshold)


def rec(score, correct, recording='r1'):
    return {'score': score, 'min_char': .9, 'correct': correct, 'recording': recording}


class CalibrationTests(unittest.TestCase):
    def test_floor_cannot_be_loosened(self):
        for bad in (dict(score=.8), dict(min_character=.7), dict(score=1.2)):
            with self.assertRaises(ValueError):
                ThresholdConfig('v', **bad)
        config = ThresholdConfig('v2', score=.95, source='dev-1')
        self.assertTrue(config.admits(.96, .8))
        self.assertFalse(config.admits(.9, .9))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'thresholds.json'
            config.dump(path)
            self.assertEqual(ThresholdConfig.load(path), config)
            path.write_text('{"version": "x", "score": 0.5}')
            with self.assertRaises(ValueError):
                ThresholdConfig.load(path)

    def test_risk_coverage_and_zero_error_bound(self):
        records = [rec(.99, True)] * 5 + [rec(.9, False)] + [rec(.88, True)] * 2
        rows = risk_coverage(records, [.5, .9, .95])
        self.assertEqual([r['threshold'] for r in rows], [.85, .9, .95], 'values below the floor are raised to it')
        self.assertEqual((rows[0]['accepted'], rows[0]['wrong']), (8, 1))
        self.assertEqual((rows[2]['accepted'], rows[2]['wrong'], rows[2]['zero_error_upper_95']), (5, 0, .6))
        self.assertEqual(strictest_clean_threshold(records), .99)

    def test_ece(self):
        self.assertIsNone(expected_calibration_error([]))
        perfect = expected_calibration_error([rec(.95, True)] * 19 + [rec(.95, False)])
        self.assertEqual(perfect['ece'], 0.)
        over = expected_calibration_error([rec(.95, False)] * 10)
        self.assertEqual(over['ece'], .95)

    def test_leave_one_recording_out(self):
        self.assertEqual(leave_one_recording_out([rec(.9, True)])['status'], 'not_run')
        records = ([rec(.97, True, 'a')] * 3 + [rec(.9, False, 'b')] + [rec(.98, True, 'b')] * 2
                   + [rec(.96, True, 'c')] * 3)
        result = leave_one_recording_out(records)
        self.assertEqual(result['status'], 'run')
        self.assertEqual(len(result['folds']), 3)
        # Recording b holds the only error; the other recordings cannot reveal it, so a threshold fitted
        # without b still lets it through. Leave-one-out must expose that, not hide it.
        self.assertEqual({f['held_out']: f['wrong'] for f in result['folds']}, {'a': 0, 'b': 1, 'c': 0})
        self.assertEqual(result['wrong_total'], 1)
        self.assertIn('not a 99.9% guarantee', report(records, version='dev-1')['caution'])


if __name__ == '__main__':
    unittest.main()
