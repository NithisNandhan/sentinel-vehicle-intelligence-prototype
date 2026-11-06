"""Result states on SYNTHETIC CTC posteriors (tests/ctc_fixtures.py). Not accuracy evidence."""
import unittest

from sentinel_edge.ctc_fusion import CTCPlateReader
from sentinel_edge.layout import CONFLICT, TWO_ROW, passage_layout
from sentinel_edge.plate_format import completions, pattern_compatible, strict_status, TooManyCompletions
from sentinel_edge.result_state import (CONFIRMED, PARTIAL, UNREADABLE, RecognitionResult, ResultPolicy,
                                        classify)
from ctc_fixtures import TOKENS, reader_with

PLATE = 'MH49BE5557'


def clean(text=PLATE):
    return list(text)


def weak_at(position, rival=None, text=PLATE, peak=.45):
    spec = list(text)
    spec[position] = (text[position], peak, rival) if rival else (text[position], peak)
    return spec


class ConfirmedTests(unittest.TestCase):
    def test_three_clean_frames_confirm_a_strictly_valid_plate(self):
        result = classify(reader_with([clean()] * 3))
        self.assertEqual(result.state, CONFIRMED)
        self.assertEqual(result.plate, PLATE)
        self.assertEqual(result.masked(), 'MH******57')
        self.assertEqual(result.known_characters, 10)
        self.assertNotIn(PLATE, repr(result))

    def test_safety_floor_is_unchanged_two_frames_never_confirm(self):
        result = classify(reader_with([clean()] * 2))
        self.assertNotEqual(result.state, CONFIRMED)
        self.assertEqual(result.state, PARTIAL)
        self.assertEqual(result.reason, 'insufficient_observations')
        self.assertTrue(result.review_required)
        self.assertEqual(result.unknown_positions, ())

    def test_lookalike_reading_is_never_fixed_or_confirmed(self):
        # 'MH49BE555O' only fits if O becomes 0: strict policy sends it to review.
        self.assertEqual(strict_status('MH49BE555O'), 'lookalike')
        for options in ({}, {'validate_format': True}):
            result = classify(reader_with([clean('MH49BE555O')] * 3, **options))
            self.assertNotEqual(result.state, CONFIRMED)
            if result.state == PARTIAL:
                self.assertNotEqual(result.pattern[-1], '0', 'format must never choose a character')

    def test_unknown_state_code_blocks_confirmation(self):
        result = classify(reader_with([clean('XX49BE5557')] * 3))
        self.assertNotEqual(result.state, CONFIRMED)

    def test_layout_conflict_and_low_resolution_block_confirmation(self):
        conflict = classify(reader_with([clean()] * 3), layout=CONFLICT)
        self.assertEqual((conflict.state, conflict.reason), (PARTIAL, 'layout_conflict'))
        small = classify(reader_with([clean()] * 3), max_plate_height=12)
        self.assertEqual((small.state, small.reason), (PARTIAL, 'insufficient_resolution'))
        two_row = classify(reader_with([clean()] * 3), layout=TWO_ROW, max_plate_height=20)
        self.assertEqual(two_row.reason, 'insufficient_resolution')
        self.assertEqual(classify(reader_with([clean()] * 3), max_plate_height=40).state, CONFIRMED)

    def test_secondary_veto_sends_reading_to_review_with_disputed_position_unknown(self):
        result = classify(reader_with([clean()] * 3), veto_positions={5})
        self.assertEqual((result.state, result.reason), (PARTIAL, 'secondary_reader_veto'))
        self.assertEqual(result.unknown_positions, (5,))
        self.assertEqual(result.pattern, 'MH49B?5557')


class PartialTests(unittest.TestCase):
    def test_one_disputed_position_becomes_unknown(self):
        frames = [weak_at(5, 'F'), weak_at(5, 'F'), clean('MH49BF5557')]
        result = classify(reader_with(frames))
        self.assertEqual(result.state, PARTIAL)
        self.assertEqual(result.pattern, 'MH49B?5557')
        self.assertEqual(result.unknown_positions, (5,))
        self.assertEqual(result.masked(), 'MH******57')
        for position in result.positions:
            if position.index != 5:
                self.assertTrue(position.known)
                self.assertGreaterEqual(position.support_frames, 2)

    def test_known_characters_are_always_visually_supported(self):
        frames = [weak_at(2, '8', peak=.5), weak_at(2, '8', peak=.5), weak_at(2, '8', peak=.5)]
        result = classify(reader_with(frames))
        self.assertIn(result.state, (PARTIAL, UNREADABLE))
        if result.state == PARTIAL:
            self.assertEqual(result.pattern[2], '?')

    def test_wrong_character_in_one_frame_with_confident_peak_is_unknown(self):
        frames = [clean(), clean(), clean('MH49BE5587')]
        result = classify(reader_with(frames))
        self.assertNotEqual(result.state, CONFIRMED)
        self.assertEqual(result.state, PARTIAL)
        self.assertEqual(result.pattern[8], '?')

    def test_missing_or_extra_character_is_never_confirmed(self):
        for other in ('MH49BE557', 'MH49BE55557'):
            result = classify(reader_with([clean(), clean(other), clean(other)]))
            self.assertNotEqual(result.state, CONFIRMED)
            if result.state == PARTIAL:
                self.assertFalse(result.length_certain)
                self.assertTrue(result.review_required)

    def test_split_length_vote_is_unreadable(self):
        frames = [clean(), clean(), clean('MH49BE557'), clean('MH49BE557')]
        result = classify(reader_with(frames))
        self.assertEqual((result.state, result.reason), (UNREADABLE, 'length_uncertain'))

    def test_existing_gate_deletion_path_is_blocked(self):
        # Every frame reads ten characters, yet CTC can rank a nine-character
        # deletion first. The existing accepted() gate confirms it; result states must not.
        reader = reader_with([clean('MH49BE555O')] * 3)
        self.assertEqual(reader.accepted().plate, 'MH49BE555')
        result = classify(reader)
        self.assertNotEqual(result.state, CONFIRMED)

    def test_too_many_unknown_positions_is_unreadable(self):
        frames = []
        for _ in range(3):
            spec = list(PLATE)
            for position in (2, 3, 5, 7):
                spec[position] = (PLATE[position], .4)
            frames.append(spec)
        result = classify(reader_with(frames))
        self.assertEqual(result.state, UNREADABLE)

    def test_format_impossible_pattern_is_unreadable(self):
        self.assertFalse(pattern_compatible('QQ49B?5557'))
        self.assertTrue(pattern_compatible('MH49B?5557'))


class UnreadableTests(unittest.TestCase):
    def test_no_evidence_keeps_the_given_reason(self):
        self.assertEqual(classify(None, empty_reason='plate_too_small').reason, 'plate_too_small')
        empty = CTCPlateReader(TOKENS)
        result = classify(empty)
        self.assertEqual((result.state, result.reason, result.masked()), (UNREADABLE, 'no_plate_found', None))

    def test_single_frame_is_insufficient(self):
        self.assertEqual(classify(reader_with([clean()])).reason, 'insufficient_observations')

    def test_result_invariants_are_enforced(self):
        with self.assertRaises(ValueError):
            RecognitionResult(CONFIRMED, 'layout_conflict', 3, plate=PLATE, length=10)
        with self.assertRaises(ValueError):
            RecognitionResult(PARTIAL, 'uncertain_characters', 3, pattern='MH49B?5557', length=9)
        with self.assertRaises(ValueError):
            RecognitionResult(UNREADABLE, None, 0)
        with self.assertRaises(ValueError):
            RecognitionResult(UNREADABLE, 'not-a-reason', 0)


class FormatSearchTests(unittest.TestCase):
    def test_completions_respect_known_characters_and_bounds(self):
        found = completions('MH49B?5557')
        self.assertEqual(len(found), 26)
        self.assertTrue(all(text[:5] == 'MH49B' and text[6:] == '5557' for text in found))
        self.assertEqual(completions('MH49BE5557'), ['MH49BE5557'])
        with self.assertRaises(TooManyCompletions):
            completions('??????????', limit=100)
        with self.assertRaises(ValueError):
            completions('mh49b?5557')

    def test_passage_layout(self):
        self.assertEqual(passage_layout(['one_row', 'two_row']), CONFLICT)
        self.assertEqual(passage_layout(['uncertain', 'two_row']), TWO_ROW)
        self.assertEqual(passage_layout(['uncertain']), 'undetermined')
        self.assertEqual(passage_layout([]), 'unknown')

    def test_policy_is_versioned(self):
        self.assertEqual(ResultPolicy().as_dict()['version'], 'result-policy-1')


if __name__ == '__main__':
    unittest.main()
