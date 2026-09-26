import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sentinel_edge.hybrid import HybridEngine, layout_hypotheses, layout_probabilities
from sentinel_edge.hybrid_models import HeavyExecutor, verified_asset
from sentinel_edge.reconstruction import Reconstruction

ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_?'


def probabilities(text='GJ01AB1234', score=.97):
    rows = []
    for char in text.ljust(13, '_'):
        row = [(1 - score) / (len(ALPHABET) - 1)] * len(ALPHABET)
        row[ALPHABET.index(char)] = score
        rows.append(row)
    return rows


class HybridTests(unittest.TestCase):
    def engine(self, *, raw_score=.97, processed_text='GJ01AB1234', visuals=False):
        resources = SimpleNamespace(
            ocr=SimpleNamespace(config=SimpleNamespace(alphabet=ALPHABET, pad_char='_', max_plate_slots=13)),
            executor=HeavyExecutor(), aligner=None, restorer=None,
            probabilities=lambda image: probabilities(processed_text if image[0, 0, 0] == 250 else 'GJ01AB1234',
                                                       .99 if image[0, 0, 0] == 250 else raw_score))
        return HybridEngine(resources, retain_visuals=visuals)

    def reconstruction(self, samples):
        original = samples[0].image
        image = np.full_like(original, 250)
        mask = np.ones(original.shape[:2], bool)
        return Reconstruction(image, original, mask, mask.astype(np.uint8) * 3,
                              tuple(s.source_id for s in samples), samples[0].pts_ms, 0, 'fixture')

    def add_three(self, engine):
        value = None
        for index in range(3):
            value = engine.process('7', np.full((24, 120, 3), index + 1, np.uint8), index * 300)
        return value

    def test_raw_supported_consensus_accepts(self):
        engine = self.engine()
        with patch('sentinel_edge.hybrid.reconstruct', side_effect=self.reconstruction):
            result = self.add_three(engine)
        self.assertEqual(result.plate, 'GJ01AB1234')
        self.assertIsNone(engine.tracks['7'].reconstruction)

    def test_restored_high_confidence_cannot_manufacture_acceptance(self):
        engine = self.engine(raw_score=.4, visuals=True)
        with patch('sentinel_edge.hybrid.reconstruct', side_effect=self.reconstruction):
            self.assertIsNone(self.add_three(engine))
        self.assertTrue(engine.tracks['7'].processed_candidates)
        self.assertEqual(engine.tracks['7'].observation_count, 3)

    def test_conflicting_processed_result_blocks_acceptance(self):
        engine = self.engine(processed_text='GJ01AB1235')
        with patch('sentinel_edge.hybrid.reconstruct', side_effect=self.reconstruction):
            self.assertIsNone(self.add_three(engine))
        self.assertEqual(engine.tracks['7'].reason, 'original_processed_disagreement')

    def test_identical_crop_does_not_create_temporal_votes(self):
        engine = self.engine()
        image = np.ones((24, 120, 3), np.uint8)
        for pts in (0, 300, 600):
            self.assertIsNone(engine.process('7', image, pts))
        self.assertEqual(engine.tracks['7'].observation_count, 1)

    def test_reconstruction_failure_preserves_original_ocr(self):
        engine = self.engine()
        with patch('sentinel_edge.hybrid.reconstruct', return_value=None):
            self.assertIsNotNone(self.add_three(engine))

    def test_cleanup_releases_crops_and_readers(self):
        engine = self.engine()
        engine.process('7', np.ones((24, 120, 3), np.uint8), 0)
        engine.retain(set())
        self.assertEqual(engine.bank.bytes_used, 0)
        self.assertEqual(engine.tracks, {})

    def test_no_more_than_two_layout_hypotheses(self):
        image = np.full((48, 100, 3), 255, np.uint8)
        image[5:15, 10:90] = 0
        image[32:42, 10:90] = 0
        self.assertEqual(len(layout_hypotheses(image)), 2)

    def test_executor_failure_is_numeric_and_unlocks(self):
        executor = HeavyExecutor()
        result, reason = executor.run(lambda: 1 / 0)
        self.assertIsNone(result)
        self.assertEqual(reason, 'model_failure')
        self.assertEqual(executor.run(lambda: 42), (42, None))
        self.assertEqual(executor.pending, 0)

    def test_executor_busy_bypasses_instead_of_hanging(self):
        executor = HeavyExecutor()
        executor._lock.acquire()
        try:
            self.assertEqual(executor.run(lambda: 42), (None, 'executor_busy'))
        finally:
            executor._lock.release()

    def test_missing_asset_fails_closed(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            verified_asset(Path('/not-a-model-cache'), 'nafnet-gopro32')

    def test_rows_are_recognised_separately_and_joined_in_order(self):
        cfg = SimpleNamespace(alphabet=ALPHABET, pad_char='_', max_plate_slots=13)
        calls = []
        def read(image):
            calls.append(image.shape)
            return probabilities('GJ01' if len(calls) == 1 else 'AB1234')
        resources = SimpleNamespace(ocr=SimpleNamespace(config=cfg), probabilities=read)
        rows = layout_probabilities(resources, np.ones((48, 100, 3), np.uint8), .5)
        text = ''.join(ALPHABET[max(range(len(row)), key=row.__getitem__)] for row in rows).split('_')[0]
        self.assertEqual(text, 'GJ01AB1234')
        self.assertEqual(calls, [(24, 100, 3), (24, 100, 3)])

    def test_two_row_capacity_overflow_is_not_truncated_into_a_plate(self):
        cfg = SimpleNamespace(alphabet=ALPHABET, pad_char='_', max_plate_slots=13)
        resources = SimpleNamespace(ocr=SimpleNamespace(config=cfg), probabilities=lambda _: probabilities())
        self.assertIsNone(layout_probabilities(resources, np.ones((48, 100, 3), np.uint8), .5))
