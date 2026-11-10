import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'edge'))
from compare_temporal_approaches import evaluate_segment, match_reference, quarantine_conflicts, replay_reader
from sentinel_edge.ctc_fusion import CTCPlateReader

TOKENS = ('', *'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
PLATE = 'MH01AB1234'


def observation(index, text=PLATE, quality=1.):
    rows = []
    for char in text:
        row = [.001/(len(TOKENS)-1)]*len(TOKENS)
        row[TOKENS.index(char)] = .999
        rows.extend([row, [1.] + [0.]*(len(TOKENS)-1)])
    reader = CTCPlateReader(TOKENS)
    reader.add(rows, timestamp_ms=index*500, source_id=str(index), quality=quality)
    return dict(position_ms=index*500, timestamp_ms=index*500, source_id=str(index), quality=quality,
                rows=rows, frame_candidates=reader.samples[-1][3], box=[0,0,100,40],
                native_path=f'assets/{index}.png', current_candidates=[dict(plate=text)], current_accepted=None)


class TemporalComparisonTests(unittest.TestCase):
    def test_replay_preserves_actual_ctc_scores(self):
        original = CTCPlateReader(TOKENS)
        frames = [observation(i) for i in range(3)]
        for frame in frames:
            original.add(frame['rows'], timestamp_ms=frame['timestamp_ms'],
                         source_id=frame['source_id'], quality=frame['quality'])
        replay = replay_reader(TOKENS, frames)
        self.assertEqual(original.candidates(), replay.candidates())
        self.assertEqual(original.accepted(), replay.accepted())

    def test_single_choice_uses_image_quality_and_not_truth_or_ocr_confidence(self):
        frames = [observation(0, quality=.9), observation(1, text='MH01AB1235', quality=1.)]
        result = evaluate_segment(TOKENS, {'observations':frames}, .85)
        self.assertEqual(result['single_quality']['top'], 'MH01AB1235')
        self.assertIsNone(result['single_quality']['accepted'])

    def test_no_observations_are_combined_across_reader_segments(self):
        for text in (PLATE, 'MH01AB1235'):
            result = evaluate_segment(TOKENS, {'observations':[observation(0, text=text)]}, .85)
            self.assertEqual(result['multi_originals']['top'], text)
            self.assertEqual(result['multi_originals']['accepted'], [])

    def test_conflicting_frames_can_overturn_a_correct_sharper_frame(self):
        frames = [observation(0), observation(1, 'MH01AB1235', .9),
                  observation(2, 'MH01AB1235', .9)]
        result = evaluate_segment(TOKENS, {'observations': frames}, .85)
        self.assertEqual(result['single_quality']['top'], PLATE)
        self.assertEqual(result['multi_originals']['top'], 'MH01AB1235')

    def test_geometry_requires_the_same_timestamp_and_rejects_ambiguity(self):
        sample = dict(id='x', position_ms=0, box=[0,0,100,40])
        segments = [dict(id='a', observations=[observation(0)])]
        self.assertEqual(match_reference(sample, segments, {})['segment'], 'a')
        self.assertIsNone(match_reference({**sample, 'position_ms':200}, segments, {}))
        segments.append(dict(id='b', observations=[observation(0)]))
        self.assertTrue(match_reference(sample, segments, {})['ambiguous'])

    def test_conflicting_labels_do_not_choose_the_favourable_truth(self):
        rows = [dict(id=str(i), expected=text, association={'segment':'same', 'ambiguous':False},
                     observations=4, methods={'single_quality': {'top':PLATE,'exact':text==PLATE}})
                for i, text in enumerate((PLATE, 'MH01AB1235'))]
        self.assertEqual(quarantine_conflicts(rows), ['same'])
        for row in rows:
            self.assertTrue(row['association']['ambiguous'])
            self.assertEqual(row['observations'], 0)
            self.assertIsNone(row['methods']['single_quality']['top'])


if __name__ == '__main__':
    unittest.main()
