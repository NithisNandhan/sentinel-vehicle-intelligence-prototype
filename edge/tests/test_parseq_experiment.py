import unittest

from sentinel_edge.parseq_experiment import ALPHABET, map_probabilities


class AdapterTests(unittest.TestCase):
    def test_case_merging_keeps_unknown_mass_and_end_uncertainty(self):
        tokens = ("[E]", "A", "a", "!")
        result = map_probabilities([[.1, .3, .2, .4], [.7, .1, .1, .1]], tokens)
        self.assertAlmostEqual(result[0][ALPHABET.index("A")], .5)
        self.assertAlmostEqual(result[0][ALPHABET.index("?")], .4)
        self.assertAlmostEqual(result[1][ALPHABET.index("_")], .7)
        self.assertEqual(len(result), 13)
        self.assertTrue(all(abs(sum(row) - 1) < 1e-9 for row in result))

    def test_does_not_fabricate_end_token_for_truncated_output(self):
        with self.assertRaises(ValueError):
            map_probabilities([[.1, .9]], ("[E]", "A"))

    def test_invalid_distributions_are_rejected(self):
        for rows in ([], [[float("nan"), 1]], [[-1, 2]], [[0, 0]], [[.1, .1]]):
            with self.assertRaises(ValueError):
                map_probabilities(rows, ("[E]", "A"))
