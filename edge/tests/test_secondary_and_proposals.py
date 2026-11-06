"""Secondary veto and multi-source proposal pool (SYNTHETIC inputs only)."""
from types import SimpleNamespace
import unittest

import numpy as np

from ctc_fixtures import TOKENS, rows_for
from sentinel_edge.plate_proposals import ProposalPool, plate_like, rank, with_margin
from sentinel_edge.secondary_veto import SecondaryVeto, disputed, evaluate

PLATE = 'MH49BE5557'


class VetoTests(unittest.TestCase):
    def reading(self, text, peak=.95):
        return tuple((c, peak) for c in text)

    def test_only_confident_repeated_aligned_disagreement_vetoes(self):
        self.assertEqual(disputed(PLATE, [self.reading('MH49BF5557')] * 2), frozenset({5}))
        self.assertIsNone(disputed(PLATE, [self.reading('MH49BF5557')]), 'one frame is not enough')
        self.assertIsNone(disputed(PLATE, [self.reading('MH49BF5557', .7)] * 3), 'weak disagreement')
        self.assertIsNone(disputed(PLATE, [self.reading('MH49BF557')] * 3), 'different length')
        self.assertIsNone(disputed(PLATE, [self.reading(PLATE)] * 3))

    def test_reader_integration_and_metrics(self):
        reader = SimpleNamespace(tokens=TOKENS, probabilities=lambda crop: rows_for(list('MH49BF5557')))
        veto = SecondaryVeto(reader)
        crop = np.zeros((20, 80, 3), np.uint8)
        self.assertEqual(veto.disputed_positions(PLATE, [crop, crop]), frozenset({5}))
        self.assertEqual(veto.metrics['vetoes'], 1)
        with self.assertRaises(ValueError):
            SecondaryVeto(reader, threshold=.2)

    def test_evaluation_reports_false_vetoes_and_coverage_cost(self):
        good = {'truth': PLATE, 'primary': PLATE, 'secondary': [list(self.reading('MH49BF5557'))] * 2}
        caught = {'truth': PLATE, 'primary': 'MH49BE5587', 'secondary': [list(self.reading(PLATE))] * 2}
        report = evaluate([good, caught, {'truth': PLATE, 'primary': None}])
        self.assertEqual(report['false_vetoes_blocked_correct'], 1)
        self.assertEqual(report['useful_vetoes_blocked_wrong'], 1)
        self.assertEqual(report['coverage_cost'], {'count': 1, 'of': 1, 'rate': 1.0})
        self.assertEqual(report['recommendation'], 'insufficient data')


class ProposalTests(unittest.TestCase):
    def test_rank_dedupes_prefers_plate_detector_and_filters_generic_text(self):
        shape = (400, 600, 3)
        proposals = [((100, 300, 220, 340), 'plate', .7), ((104, 302, 226, 342), 'tile', None),
                     ((10, 10, 590, 60), 'text', None),            # Shop-sign banner across the top.
                     ((300, 310, 400, 340), 'text', None)]          # Plate-like text region.
        ranked = rank(proposals, shape)
        self.assertEqual(ranked[0].sources, ('plate', 'tile'))
        self.assertEqual(ranked[0].box, (100, 300, 226, 342), 'union keeps both edges')
        self.assertEqual([p.sources for p in ranked[1:]], [('text',)])
        self.assertFalse(plate_like((10, 10, 590, 60), shape))

    def test_margin_never_leaves_the_image(self):
        self.assertEqual(with_margin((0, 0, 100, 20), (30, 100, 3)), (0, 0, 100, 22))

    def test_pool_respects_budget_and_falls_back_to_primary(self):
        primary = SimpleNamespace(box=lambda region: (100, 300, 220, 340), last_calls=1, last_confidence=.8)
        calls = []
        tiles = SimpleNamespace(boxes=lambda tile: calls.append(tile.shape) or [])
        clock = iter([0., 0., .5, .5, .5, .5, .5]).__next__
        pool = ProposalPool(primary, tile_detector=tiles, min_tile_region=300, frame_budget_ms=100, clock=clock)
        pool.begin_frame()
        box = pool.box(np.zeros((400, 1000, 3), np.uint8))
        self.assertEqual(box, (92, 297, 228, 343))
        self.assertGreaterEqual(pool.skipped_for_budget, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(pool.last_confidence, .8)
        empty = ProposalPool(SimpleNamespace(box=lambda region: None))
        self.assertIsNone(empty.box(np.zeros((100, 200, 3), np.uint8)))


if __name__ == '__main__':
    unittest.main()
