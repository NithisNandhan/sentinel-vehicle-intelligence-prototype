import unittest

from sentinel_edge.ocr_refinement import choose_suggestion, refine


class RefinementTests(unittest.TestCase):
    tokens = ("", *"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def rows(self, text, alternative=None, alternative_score=.46):
        rows = []
        for index, char in enumerate(text):
            row = [0.] * len(self.tokens)
            row[self.tokens.index(char)] = 1.
            if alternative and index == len(text) - 1:
                row[self.tokens.index(char)] = 1. - alternative_score
                row[self.tokens.index(alternative)] = alternative_score
            rows += [row, [1., *([0.] * (len(self.tokens) - 1))]]
        return rows

    def test_agreement_can_recover_close_alternative_without_new_votes(self):
        native = self.rows("MH12AB1230", "4")
        alternative = self.rows("MH12AB1234")
        result = choose_suggestion(native, alternative, alternative, self.tokens)
        self.assertEqual(result["reading"], "MH12AB1234")
        self.assertTrue(result["changed"])
        self.assertEqual(result["independent_observations_added"], 0)
        self.assertEqual(native, self.rows("MH12AB1230", "4"))

    def test_agreeing_views_cannot_override_weak_original_support(self):
        native = self.rows("MH12AB1230", "4", .01)
        alternative = self.rows("MH12AB1234")
        result = choose_suggestion(native, alternative, alternative, self.tokens)
        self.assertFalse(result["changed"])

    def test_disagreement_and_multiple_edits_keep_native(self):
        native = self.rows("MH12AB1230", "4")
        for a, b in [("MH12AB1234", "MH12AB1235"), ("MH12AB1254", "MH12AB1254")]:
            result = choose_suggestion(native, self.rows(a), self.rows(b), self.tokens)
            self.assertFalse(result["changed"])

    def test_agreement_with_weak_character_scores_keeps_native(self):
        native = self.rows("MH12AB1230", "4")
        alternative = self.rows("MH12AB1234")
        alternative[-2][self.tokens.index("4")] = .4
        alternative[-2][self.tokens.index("0")] = .3
        alternative[-2][0] = .3
        result = choose_suggestion(native, alternative, alternative, self.tokens)
        self.assertFalse(result["changed"])

    def test_strong_native_avoids_retry(self):
        test = self

        class Reader:
            tokens = test.tokens
            calls = 0

            def probabilities(self, image):
                self.calls += 1
                return test.rows("MH12AB1234")

        reader = Reader()
        result = refine(reader, None)
        self.assertEqual(reader.calls, 1)
        self.assertEqual(result["reason"], "native_strong")


if __name__ == "__main__":
    unittest.main()
