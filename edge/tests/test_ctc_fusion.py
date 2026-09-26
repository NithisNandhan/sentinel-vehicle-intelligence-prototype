import itertools
import math
import unittest

from sentinel_edge.ctc_fusion import (CTCPlateReader, greedy_text, prefix_candidates,
                                      sequence_score, validate_ctc)


class CTCTests(unittest.TestCase):
    def test_repeated_symbols_require_separator(self):
        tokens = ('', 'A', 'a', ' ')
        self.assertEqual(greedy_text([[0, 1, 0, 0], [0, 1, 0, 0]], tokens), 'A')
        rows = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]
        self.assertEqual(prefix_candidates(rows, tokens)[0][0], 'AA')
        self.assertEqual(sequence_score(rows, tokens, 'AA')[0], 0)
        # Do not fold case before collapse and lose a character.
        rows = [[0, 1, 0, 0], [0, 0, 1, 0]]
        self.assertEqual(prefix_candidates(rows, tokens)[0][0], 'AA')
        self.assertEqual(sequence_score(rows, tokens, 'AA')[0], 0)

    def test_forward_score_matches_exhaustive_paths(self):
        tokens = ('', 'A', 'a', 'B', ' ')
        rows = [[.2, .3, .1, .2, .2], [.3, .2, .2, .2, .1], [.1, .2, .1, .4, .2]]
        for text in ('A', 'AA', 'AB', 'ABA', 'BA'):
            expected = 0.
            for path in itertools.product(range(len(tokens)), repeat=len(rows)):
                previous, emitted = None, []
                for token in path:
                    if token and token != previous:
                        emitted.append(tokens[token])
                    previous = token
                if ''.join(emitted).replace(' ', '').upper() == text:
                    expected += math.prod(rows[i][token] for i, token in enumerate(path))
            actual, _ = sequence_score(rows, tokens, text)
            self.assertAlmostEqual(math.exp(actual), expected, places=10)

    def test_stable_prefix_beam_matches_exhaustive_path_mass(self):
        tokens = ('', 'A', 'a', 'B', ' ')
        rows = [[.2, .3, .1, .2, .2], [.3, .2, .2, .2, .1], [.1, .2, .1, .4, .2]]
        expected = {}
        for path in itertools.product(range(len(tokens)), repeat=len(rows)):
            previous, emitted = None, []
            for token in path:
                if token and token != previous:
                    emitted.append(tokens[token])
                previous = token
            text = ''.join(emitted).replace(' ', '').upper()
            if text:
                expected[text] = expected.get(text, 0.) + math.prod(rows[i][token] for i, token in enumerate(path))
        actual = dict(prefix_candidates(rows, tokens, beam_width=200))
        self.assertEqual(set(actual), set(expected))
        for text in expected:
            self.assertAlmostEqual(math.exp(actual[text]), expected[text], places=10)

    def test_invalid_scores_and_bounds_fail(self):
        for rows in ([], [[.2, .2]], [[float('nan'), 1]], [[-.1, 1.1]], [[.1, .1]]):
            with self.assertRaises(ValueError):
                validate_ctc(rows, ('', 'A'))

    def test_three_distinct_observations_not_three_reused_views(self):
        tokens = ('', *'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        text = 'GJ01AB1234'
        rows = []
        for char in text:
            blank, character = [0.] * len(tokens), [0.] * len(tokens)
            blank[0], character[tokens.index(char)] = 1., 1.
            rows.extend([character, blank])
        reader = CTCPlateReader(tokens)
        self.assertTrue(reader.add(rows, timestamp_ms=0, source_id='frame-a'))
        self.assertFalse(reader.add(rows, timestamp_ms=400, source_id='frame-a'))
        self.assertIsNone(reader.accepted())
        self.assertTrue(reader.add(rows, timestamp_ms=400, source_id='frame-b'))
        self.assertTrue(reader.add(rows, timestamp_ms=800, source_id='frame-c'))
        self.assertEqual(reader.accepted().plate, text)
        self.assertEqual(reader.observation_count, 3)

    def test_model_ambiguity_does_not_become_certainty(self):
        tokens = ('', *'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        reader = CTCPlateReader(tokens)
        rows = []
        for char in 'GJ01AB1234':
            row = [0.] * len(tokens)
            row[tokens.index(char)] = 1.
            if char == 'B':
                row[tokens.index('B')] = .51
                row[tokens.index('C')] = .49
            rows.extend([row, [1.] + [0.] * (len(tokens) - 1)])
        for index in range(3):
            reader.add(rows, timestamp_ms=index * 400, source_id=str(index))
        self.assertIsNone(reader.accepted())
        self.assertEqual(len(reader.candidates()), 2)
