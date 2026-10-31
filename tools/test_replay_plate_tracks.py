"""Track labels and fusion replay in tools/replay_plate_tracks.py (no OCR model needed)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'edge'), str(ROOT.parent / 'edge' / 'tests')]

from replay_plate_tracks import ADJUSTED, one_crop, replay, replay_with_notes, track_label  # noqa: E402
from test_plate_format import TOKENS, frame  # noqa: E402


class TrackLabel(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(track_label(['KL07CD4321', 'UNREADABLE']), 'KL07CD4321')
        self.assertEqual(track_label(['KL07CD4321', 'KL07CD4327']), 'UNREADABLE')   # Conflict.
        self.assertEqual(track_label(['NO_PLATE', 'UNREADABLE']), 'UNREADABLE')
        self.assertEqual(track_label(['NO_PLATE', 'NO_PLATE']), 'NO_PLATE')


class Replay(unittest.TestCase):
    def test_accepts_after_three_frames_and_resets_on_gaps(self):
        rows = frame('MH12AB1234')
        steady = [(i * 500, f's{i}', rows) for i in range(3)]
        self.assertEqual(replay(steady, TOKENS, {}), ('MH12AB1234', 'MH12AB1234'))
        gapped = [(0, 'a', rows), (500, 'b', rows), (4000, 'c', rows)]   # 3,500 ms gap: new reader.
        self.assertEqual(replay(gapped, TOKENS, {}), ('', ''))
        close = [(0, 'a', rows), (100, 'b', rows), (500, 'c', rows)]     # 100 ms apart: skipped.
        self.assertEqual(replay(close, TOKENS, {}), ('', ''))

    def test_settings_reach_the_reader(self):
        rows = frame('XX12AB1234')
        observations = [(i * 500, f's{i}', rows) for i in range(3)]
        self.assertEqual(replay(observations, TOKENS, {})[0], 'XX12AB1234')
        self.assertEqual(replay(observations, TOKENS, {'validate_format': True}), ('', ''))

    def test_validator_made_readings_are_noted(self):
        created = [(i * 500, f's{i}', frame('0D02AB1234')) for i in range(3)]
        result = replay_with_notes(created, TOKENS, {'validate_format': True})
        self.assertEqual((result['top'], result['top_note']), ('OD02AB1234', ADJUSTED))
        self.assertEqual((result['accepted'], result['accepted_note']), ('', ''))   # Gate rejects it.
        self.assertEqual(one_crop(frame('0D02AB1234'), TOKENS, validate_format=True), ('OD02AB1234', ADJUSTED))
        as_read = replay_with_notes([(i * 500, f's{i}', frame('MH12AB1234')) for i in range(3)], TOKENS,
                                    {'validate_format': True})
        self.assertEqual((as_read['accepted'], as_read['accepted_note'], as_read['top_note']), ('MH12AB1234', '', ''))


if __name__ == '__main__':
    unittest.main()
