"""Opt-in Indian plate format validator (SENTINEL_PLATE_FORMAT_VALIDATOR, off by default)."""
from types import SimpleNamespace
import unittest

import numpy as np

from sentinel_edge.ctc_fusion import CTCPlateReader
from sentinel_edge.plate_format import STATE_CODES, check_plate, is_valid_plate
from sentinel_edge.plate_options import PlateOptions
from sentinel_edge.tracked_ocr import TrackedOCR

TOKENS = ('', *'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
EPSILON = 1e-4


def row(**mass):
    """One CTC row: the given characters' mass, a little on every token, rest on blank."""
    values = [EPSILON] * len(TOKENS)
    for character, probability in mass.items():
        values[TOKENS.index(character.lstrip('_'))] = probability
    values[0] = 0.
    values[0] = 1. - sum(values)
    return values


def frame(text, overrides=None):
    """Rows reading `text`, with a blank between characters; `overrides` maps position -> mass."""
    rows = []
    for position, character in enumerate(text):
        mass = (overrides or {}).get(position, {character: .95})
        rows.append(row(**{f'_{key}': value for key, value in mass.items()}))
        rows.append(row())
    return rows


def fused(frames, **options):
    reader = CTCPlateReader(TOKENS, **options)
    for i, rows in enumerate(frames):
        reader.add(rows, timestamp_ms=i * 500, source_id=f'frame-{i}')
    return reader


class ValidPlatesUnchanged(unittest.TestCase):
    def test_valid_plates_are_never_changed(self):
        # Each holds look-alike characters in positions where they are already right.
        for plate in ('MH12AB1234', 'KL07CD4321', 'MH12SB5678', 'GJ05BZ6789', 'OD02GS1080',
                      'KA01Z1234', 'TS09T4567', 'TG09OB2222', 'DL3CAB1234', 'DL10CA0001',
                      'MH43BE1110', '22BH1234AA', '23BH0042B'):
            with self.subTest(plate=plate):
                result = check_plate(plate)
                self.assertEqual((result.status, result.plate, result.swaps), ('valid', plate, 0))

    def test_state_code_list(self):
        self.assertIn('TG', STATE_CODES)   # Telangana since 2024; TS kept for older plates.
        self.assertIn('OD', STATE_CODES)
        self.assertEqual(len(STATE_CODES), 40)
        self.assertFalse(is_valid_plate('XX12AB1234'))
        self.assertFalse(is_valid_plate('QQ01A1234'))


class LookAlikeFixes(unittest.TestCase):
    def assertFixed(self, reading, plate):
        result = check_plate(reading)
        self.assertEqual((result.status, result.plate), ('corrected', plate), reading)

    def test_o_and_zero(self):
        self.assertFixed('0D02AB1234', 'OD02AB1234')   # State code needs letters.
        self.assertFixed('MH12AB1O34', 'MH12AB1034')   # Number needs digits.
        self.assertFixed('MH1OAB1234', 'MH10AB1234')   # District needs digits.

    def test_i_and_one(self):
        self.assertFixed('MHI2AB1234', 'MH12AB1234')
        self.assertFixed('MH12AB12I4', 'MH12AB1214')
        # A five-digit number is impossible, so the 1 after the series must be a letter.
        self.assertFixed('MH12AB12345', 'MH12ABI2345')

    def test_b_and_eight(self):
        self.assertFixed('MH12A81234', 'MH12AB1234')   # Series letter.
        self.assertFixed('MH12AB12B4', 'MH12AB1284')   # Number digit.

    def test_s_and_five(self):
        self.assertFixed('MH1SAB1234', 'MH15AB1234')
        self.assertFixed('MH12AB12S4', 'MH12AB1254')

    def test_z_and_two(self):
        self.assertFixed('MHZ2AB1234', 'MH22AB1234')
        self.assertFixed('MH12AB1Z34', 'MH12AB1234')

    def test_g_and_six(self):
        self.assertFixed('6J05AB1234', 'GJ05AB1234')
        self.assertFixed('MH12AB12G4', 'MH12AB1264')

    def test_several_swaps_when_each_is_forced(self):
        self.assertFixed('0DO2AB1234', 'OD02AB1234')
        self.assertEqual(check_plate('0DO2AB1234').swaps, 2)

    def test_other_confusions_are_not_fixed(self):
        # D/0 and Q/0 are not in the approved list, so these stay invalid.
        self.assertEqual(check_plate('MH1DAB1234').status, 'invalid')
        self.assertEqual(check_plate('MH12AB1Q34').status, 'invalid')


class NotPlates(unittest.TestCase):
    def test_impossible_readings_are_invalid(self):
        for reading in ('XX12AB1234',     # Unknown state code.
                        'MH00AB1234',     # District 00.
                        'MH12AB0000',     # Number 0000.
                        'MH1AB1234',      # One-digit district outside Delhi.
                        'MH12ABCD1234',   # Four series letters.
                        'MH12AB34734',    # Five-digit number, no look-alike to fix.
                        '20BH1234AA',     # Bharat series began in 2021.
                        '1234567890', 'ABCDEFGHIJ', 'MH12', '', 'MH-12-AB'):
            with self.subTest(reading=reading):
                result = check_plate(reading)
                self.assertEqual(result.status, 'invalid')
                self.assertIsNone(result.plate)
                self.assertFalse(result.usable)

    def test_two_equally_small_fixes_are_not_guessed(self):
        # 5->S gives MH99SB84 and B->8 gives MH995884: one swap each, both valid.
        result = check_plate('MH995B84')
        self.assertEqual((result.status, result.plate), ('ambiguous', None))

    def test_fewest_swaps_wins(self):
        # 8->B alone fixes it; the two-swap MH12ABI234 (1->I as well) is not considered.
        self.assertEqual(check_plate('MH12A81234').swaps, 1)


class ReaderIntegration(unittest.TestCase):
    def test_off_by_default_everywhere(self):
        self.assertFalse(PlateOptions().format_validator)
        self.assertFalse(PlateOptions.from_env({}).format_validator)
        self.assertTrue(PlateOptions.from_env({'SENTINEL_PLATE_FORMAT_VALIDATOR': 'true'}).format_validator)
        self.assertFalse(CTCPlateReader(TOKENS).validate_format)

    def test_default_reader_unchanged(self):
        frames = [frame('XX12AB1234') for _ in range(3)]
        default, explicit = fused(frames), fused(frames, validate_format=False)
        self.assertEqual([c.plate for c in default.candidates()], [c.plate for c in explicit.candidates()])
        self.assertEqual(default.accepted().plate, 'XX12AB1234')   # The broad hint lets it through.

    def test_reading_that_cannot_be_a_plate_is_dropped(self):
        reader = fused([frame('XX12AB1234') for _ in range(3)], validate_format=True)
        self.assertNotIn('XX12AB1234', [c.plate for c in reader.candidates()])
        self.assertIsNone(reader.accepted())

    def test_valid_plate_still_accepted(self):
        reader = fused([frame('MH12AB1234') for _ in range(3)], validate_format=True)
        self.assertEqual(reader.accepted().plate, 'MH12AB1234')

    def test_fix_is_rescored_and_needs_pixel_evidence(self):
        # Every frame clearly shows 8 where a series letter belongs. The fix MH12AB1234 is
        # only a candidate: its B has almost no support, so the unchanged gate rejects it.
        reader = fused([frame('MH12A81234') for _ in range(3)], validate_format=True)
        plates = [c.plate for c in reader.candidates()]
        self.assertIn('MH12AB1234', plates)
        self.assertNotIn('MH12A81234', plates)
        self.assertIsNone(reader.accepted())

    def test_valid_reading_with_look_alike_alternative_accepted(self):
        # B is the clear reading at position 5; the 8 is only a weak alternative.
        rows = frame('MH12AB1234', {5: {'B': .9, '8': .05}})
        reader = fused([rows, rows, rows], validate_format=True)
        self.assertEqual(reader.accepted().plate, 'MH12AB1234')

    def test_three_frame_rule_unchanged(self):
        reader = fused([frame('MH12AB1234') for _ in range(2)], validate_format=True)
        self.assertIsNone(reader.accepted())

    def test_tracked_pipeline_passes_the_setting_on(self):
        ocr = SimpleNamespace(tokens=TOKENS, probabilities=lambda image: frame('XX12AB1234'))
        detector = SimpleNamespace(box=lambda image: (5, 5, 150, 35), last_confidence=.9)
        track = SimpleNamespace(track_id='1', box=(0, 0, 180, 60), vehicle_class=2)
        results = {}
        for enabled in (False, True):
            pipeline = TrackedOCR(ocr, detector, options=PlateOptions(format_validator=enabled))
            for i in range(4):
                image = np.full((60, 180, 3), 100, np.uint8)
                image[10, 10] = i
                pipeline.process(image, [track], i * 500)
            reader = pipeline.histories['1'].reader
            results[enabled] = (reader.validate_format, reader.accepted() is not None)
        self.assertEqual(results, {False: (False, True), True: (True, False)})


class FormatAdjustedMark(unittest.TestCase):
    """Readings the validator corrected or created are marked "format-adjusted, unverified"."""

    # Every frame clearly reads 0 where a state-code letter belongs. The default reader shows
    # nothing (not a plate format); the validator creates OD02AB1234 by swapping 0 -> O,
    # the same way a corrected look-alike surfaced a new reading on the Mumbai clip.
    CREATED_FROM = '0D02AB1234'
    CREATED = 'OD02AB1234'

    def test_created_reading_is_marked_and_rejected_by_the_frame_gate(self):
        frames = [frame(self.CREATED_FROM) for _ in range(3)]
        self.assertEqual(fused(frames).candidates(), [])                 # Default: nothing shown.
        reader = fused(frames, validate_format=True)
        top = reader.candidates()[0]
        self.assertEqual((top.plate, top.format_adjusted), (self.CREATED, True))
        # Three frames, but the O has almost no support in any of them: the unchanged
        # weakest-character and two-frame support rules reject it.
        self.assertLess(top.minimum_character_score, .75)
        self.assertIsNone(reader.accepted())

    def test_created_reading_cannot_be_accepted_on_two_frames(self):
        # Even with the pixels supporting O, the 3-frame rule is not bypassed.
        rows = frame(self.CREATED, {0: {'O': .9, '0': .05}})
        self.assertIsNone(fused([rows, rows], validate_format=True).accepted())

    def test_gate_reasons_say_format_adjusted(self):
        from sentinel_edge.recognition_debug import FORMAT_ADJUSTED_REASON, gate_reasons
        reader = fused([frame(self.CREATED_FROM) for _ in range(3)], validate_format=True)
        reasons = gate_reasons(reader, reader.candidates(), .85)
        self.assertIn(FORMAT_ADJUSTED_REASON, reasons)
        self.assertIn('format-adjusted, unverified', FORMAT_ADJUSTED_REASON)
        self.assertIn('Weakest character score is below 0.75', reasons)
        self.assertNotIn('Model gates passed; correctness has not been independently verified', reasons)

    def test_readings_read_as_is_are_not_marked(self):
        reader = fused([frame('MH12AB1234') for _ in range(3)], validate_format=True)
        self.assertFalse(reader.candidates()[0].format_adjusted)
        self.assertFalse(any(c.format_adjusted for c in fused([frame('MH12AB1234')] * 3).candidates()))

    def test_corrected_reading_is_marked(self):
        reader = fused([frame('MH12A81234') for _ in range(3)], validate_format=True)
        flags = {c.plate: c.format_adjusted for c in reader.candidates()}
        self.assertTrue(flags['MH12AB1234'])

    def test_frame_callout_and_display_carry_the_mark(self):
        from unittest.mock import patch
        from sentinel_edge.overlay import draw_plate_callout, scale_for
        ocr = SimpleNamespace(tokens=TOKENS, probabilities=lambda image: frame(self.CREATED_FROM))
        detector = SimpleNamespace(box=lambda image: (5, 5, 150, 35), last_confidence=.9)
        track = SimpleNamespace(track_id='1', box=(0, 0, 180, 60), vehicle_class=2)
        pipeline = TrackedOCR(ocr, detector, options=PlateOptions(format_validator=True))
        for i in range(3):
            image = np.full((60, 180, 3), 100, np.uint8)
            image[10, 10] = i
            displays = pipeline.process(image, [track], i * 500)
        display = displays[-1]
        self.assertTrue(display['format_adjusted'])
        self.assertNotEqual(display['ocr_status'], 'accepted by model — NOT verified')
        view = np.zeros((480, 640, 3), np.uint8)
        with patch('sentinel_edge.overlay._banner', return_value=(0, 0, 10, 10)) as banner:
            draw_plate_callout(view, (100, 100, 400, 300), (150, 250, 300, 280), display['crop'], 'OD02****34',
                               display['ocr_status'], 3, scale_for(view), format_adjusted=display['format_adjusted'])
        self.assertIn('FORMAT-ADJUSTED, UNVERIFIED', banner.call_args_list[-1].args[1])


if __name__ == '__main__':
    unittest.main()
