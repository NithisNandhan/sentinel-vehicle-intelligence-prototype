"""Opt-in per-position, confidence-weighted vote in CTC fusion (SENTINEL_POSITION_VOTE)."""
from types import SimpleNamespace
import unittest

import numpy as np

from sentinel_edge.ctc_fusion import CTCPlateReader, greedy_reading, position_vote
from sentinel_edge.plate_options import PlateOptions
from sentinel_edge.tracked_ocr import TrackedOCR

from test_plate_format import TOKENS, frame, fused

PLATE = 'MH12AB1234'


def misread(position, wrong):
    """A frame that reads PLATE except at `position`, where `wrong` beats the true character."""
    return frame(PLATE, {position: {wrong: .55, PLATE[position]: .40}})


def reading(text, peak=.9):
    return tuple((character, peak) for character in text)


class VoteFunction(unittest.TestCase):
    def test_majority_per_position(self):
        self.assertEqual(position_vote([reading('MH12AB1234'), reading('MH12A81234'),
                                        reading('MH17AB1234')]), 'MH12AB1234')

    def test_confidence_outweighs_count(self):
        sure, unsure = reading('MH12AB1234', .95), reading('MH12A81234', .3)
        self.assertEqual(position_vote([sure, unsure, unsure]), 'MH12AB1234')   # .95 > .3 + .3

    def test_frame_weights_scale_votes(self):
        votes = [reading('MH12AB1234'), reading('MH12A81234')]
        self.assertEqual(position_vote(votes, [1., .2]), 'MH12AB1234')
        self.assertEqual(position_vote(votes, [.2, 1.]), 'MH12A81234')

    def test_only_the_common_length_votes(self):
        # The 9-character frame cannot vote: its positions are not the same slots.
        self.assertEqual(position_vote([reading('MH12AB1234'), reading('MH12AB1234'),
                                        reading('MH2AB1234', .99)]), 'MH12AB1234')

    def test_no_guess(self):
        self.assertIsNone(position_vote([]))
        self.assertIsNone(position_vote([reading('MH12AB1234')]))                       # One frame.
        self.assertIsNone(position_vote([reading('MH12AB1234'), reading('MH12A81234')]))  # Position tie.
        self.assertIsNone(position_vote([reading('MH12AB1234'), reading('MH12AB1234'),     # Length tie.
                                         reading('MH2AB1234'), reading('MH2AB1234')]))
        self.assertIsNone(position_vote([(), ()]))

    def test_greedy_reading_keeps_each_characters_peak(self):
        tokens = ('', 'A', 'B', ' ')
        def one(index, peak):
            values = [(1. - peak) / 3] * 4
            values[index] = peak
            return values
        # A A (one run: peak .9), space (dropped), blank, A (a second A), B.
        rows = [one(1, .6), one(1, .9), one(3, .8), one(0, .9), one(1, .7), one(2, .8)]
        self.assertEqual(greedy_reading(rows, tokens), (('A', .9), ('A', .7), ('B', .8)))


class ReaderVote(unittest.TestCase):
    def frames(self):
        # No frame reads the whole plate right: each misreads a different position.
        return [misread(2, '7'), misread(5, '8'), misread(8, '0')]

    def test_off_by_default_and_unchanged(self):
        self.assertFalse(PlateOptions().position_vote)
        self.assertFalse(PlateOptions.from_env({}).position_vote)
        self.assertTrue(PlateOptions.from_env({'SENTINEL_POSITION_VOTE': 'true'}).position_vote)
        default, explicit = fused(self.frames(), beam_width=1), fused(self.frames(), beam_width=1,
                                                                      position_vote=False)
        self.assertEqual([(c.plate, c.log_score) for c in default.candidates()],
                         [(c.plate, c.log_score) for c in explicit.candidates()])
        self.assertIsNone(default.voted)

    def test_vote_supplies_a_reading_no_single_frame_made(self):
        # Beam width 1 keeps only each frame's own reading, so without the vote the
        # correct plate is never a candidate and nothing passes the gate.
        off = fused(self.frames(), beam_width=1)
        self.assertNotIn(PLATE, [c.plate for c in off.candidates()])
        self.assertIsNone(off.accepted())
        on = fused(self.frames(), beam_width=1, position_vote=True)
        self.assertEqual(on.voted, PLATE)
        self.assertEqual(on.candidates()[0].plate, PLATE)
        self.assertEqual(on.accepted().plate, PLATE)   # Same gate: 3 frames, .85, .75, margin .5.

    def test_vote_cannot_bypass_the_three_frame_rule(self):
        reader = fused(self.frames()[:2], beam_width=1, position_vote=True)
        self.assertIsNone(reader.accepted())

    def test_vote_cannot_bypass_the_evidence_gate(self):
        # Position 7 is a near coin-flip in every frame. The vote still picks 7
        # (.5 + .5 + .45 against .45 + .45 + .5), but that character's support is far below
        # the unchanged .75 weakest-character gate, so the reader abstains.
        weak = [frame(PLATE, {7: {'7': .5, '2': .45}}), frame(PLATE, {7: {'7': .5, '2': .45}}),
                frame(PLATE, {7: {'2': .5, '7': .45}})]
        reader = fused(weak, position_vote=True)
        self.assertEqual(reader.voted, 'MH12AB1734')
        self.assertIsNone(reader.accepted())

    def test_vote_respects_the_format_setting(self):
        frames = [frame('XX12AB1234') for _ in range(3)]
        self.assertEqual(fused(frames, position_vote=True).voted, 'XX12AB1234')
        self.assertIsNone(fused(frames, position_vote=True, validate_format=True).voted)
        fixed = [frame('MH12A81234') for _ in range(3)]
        self.assertEqual(fused(fixed, position_vote=True, validate_format=True).voted, PLATE)

    def test_tracked_pipeline_passes_the_setting_on(self):
        ocr = SimpleNamespace(tokens=TOKENS, probabilities=lambda image: frame(PLATE))
        detector = SimpleNamespace(box=lambda image: (5, 5, 150, 35), last_confidence=.9)
        track = SimpleNamespace(track_id='1', box=(0, 0, 180, 60), vehicle_class=2)
        pipeline = TrackedOCR(ocr, detector, options=PlateOptions(position_vote=True))
        for i in range(3):
            image = np.full((60, 180, 3), 100, np.uint8)
            image[10, 10] = i
            pipeline.process(image, [track], i * 500)
        reader = pipeline.histories['1'].reader
        self.assertTrue(reader.use_position_vote)
        self.assertEqual(reader.voted, PLATE)
        self.assertEqual(reader.accepted().plate, PLATE)


class VoteReport(unittest.TestCase):
    """Diagnostic only: why the vote did or did not add a candidate."""

    def test_setting_off(self):
        self.assertEqual(fused([frame(PLATE)] * 3).vote_report()['outcome'], 'setting_off')

    def test_one_reading_cannot_vote(self):
        report = fused([frame(PLATE)], position_vote=True).vote_report()
        self.assertEqual((report['outcome'], report['readings'], report['voters']), ('too_few_same_length', 1, 1))

    def test_readings_of_different_lengths(self):
        # Three lengths, one frame each: no common length to vote on.
        report = fused([frame(PLATE), frame('MH12AB123'), frame('MH2AB123')], position_vote=True).vote_report()
        self.assertEqual((report['outcome'], report['lengths']), ('length_tie', {8: 1, 9: 1, 10: 1}))
        # Two frames agree on 9 characters: only they vote; the 10-character frame does not.
        report = fused([frame(PLATE), frame('MH12AB123'), frame('MH12AB123')], position_vote=True).vote_report()
        self.assertEqual((report['outcome'], report['voters'], report['admitted']), ('already_in_beam', 2, 'MH12AB123'))
        # Two 9-character frames that disagree everywhere tie on every position.
        report = fused([frame(PLATE), frame('MH12AB123'), frame('MH2AB1234')], position_vote=True).vote_report()
        self.assertEqual(report['outcome'], 'position_tie')

    def test_vote_already_in_a_beam(self):
        report = fused([frame(PLATE)] * 3, position_vote=True).vote_report()
        self.assertEqual((report['outcome'], report['admitted'], report['differs_from_top']),
                         ('already_in_beam', PLATE, False))

    def test_vote_adds_a_new_candidate(self):
        frames = [misread(2, '7'), misread(5, '8'), misread(8, '0')]
        report = fused(frames, beam_width=1, position_vote=True).vote_report()
        self.assertEqual((report['outcome'], report['admitted'], report['differs_from_top']),
                         ('new_candidate', PLATE, True))
        self.assertNotEqual(report['top_without_vote'], PLATE)

    def test_vote_that_is_not_a_plate(self):
        report = fused([frame('XX12AB1234')] * 3, position_vote=True, validate_format=True).vote_report()
        self.assertEqual(report['outcome'], 'not_plate_format')

    def test_report_does_not_change_fusion(self):
        frames = [misread(2, '7'), misread(5, '8'), misread(8, '0')]
        reader = fused(frames, beam_width=1, position_vote=True)
        before = [(c.plate, c.log_score) for c in reader.candidates()]
        reader.vote_report()
        self.assertEqual([(c.plate, c.log_score) for c in reader.candidates()], before)
        self.assertEqual(reader.accepted().plate, PLATE)


if __name__ == '__main__':
    unittest.main()
