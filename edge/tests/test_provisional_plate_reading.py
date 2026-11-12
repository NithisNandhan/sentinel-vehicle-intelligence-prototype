import unittest
from unittest.mock import patch
from sentinel_edge.provisional_plate_reading import check_reading,ocr_allowed,unconstrained_top


class ProvisionalReadingTests(unittest.TestCase):
    def test_uncertain_can_attempt_ocr_without_becoming_a_verified_region(self):
        verdict={'status':'uncertain','classifier_score':.94}
        self.assertFalse(ocr_allowed(verdict))
        self.assertTrue(ocr_allowed(verdict,allow_uncertain=True))
        result=check_reading(verdict,'MH02AB1234')
        self.assertEqual(result['display_plate'],'MH02AB1234')
        self.assertEqual(result['state'],'provisional_plate')
        self.assertFalse(result['operational_eligible'])
        self.assertEqual(verdict['status'],'uncertain')

    def test_unknown_prefix_can_resemble_a_plate_but_is_not_a_registered_identity(self):
        result=check_reading({'status':'uncertain'},'XX02AB1234')
        self.assertEqual(result['format_status'],'registration_shape_only')
        self.assertEqual(result['display_plate'],'XX02AB1234')
        self.assertFalse(result['operational_eligible'])

    def test_formatting_normalisation_preserves_raw_text(self):
        result=check_reading({'status':'uncertain'},'mh 02-ab 1234')
        self.assertEqual(result['raw_reading'],'mh 02-ab 1234')
        self.assertEqual(result['display_plate'],'MH02AB1234')

    def test_no_character_swaps_or_missing_prefixes_are_invented(self):
        for text in ('MHO2AB1234','02AB1234','TOYOTA','000000','MH02AB0000'):
            with self.subTest(text=text):
                result=check_reading({'status':'uncertain'},text)
                # A zero-only serial can look like a registration even though
                # it is not recognised by the stricter project validator.
                if text=='MH02AB0000':
                    self.assertEqual(result['format_status'],'registration_shape_only')
                else:self.assertIsNone(result['display_plate'])
                self.assertEqual(result['raw_reading'],text)

    def test_rejected_region_never_becomes_ocr_eligible_or_a_result(self):
        self.assertFalse(ocr_allowed({'status':'likely_non_plate'},allow_uncertain=True))
        self.assertIsNone(check_reading({'status':'likely_non_plate'},'MH02AB1234')['display_plate'])

    def test_bharat_reading_and_empty_hypothesis(self):
        self.assertEqual(check_reading({'status':'likely_plate'},'22BH1234AA')['format_status'],'recognized_format')
        self.assertEqual(check_reading({'status':'uncertain'},None)['format_status'],'not_read')

    def test_format_check_does_not_select_a_lower_ranked_plate_shaped_hypothesis(self):
        with patch('sentinel_edge.ctc_fusion.prefix_candidates',return_value=[('TOYOTA',-.1),('MH02AB1234',-5.)]):
            top=unconstrained_top([],())
        self.assertEqual(top,'TOYOTA')
        self.assertIsNone(check_reading({'status':'uncertain'},top)['display_plate'])

    def test_unfiltered_fusion_cannot_auto_accept_even_confident_text(self):
        from sentinel_edge.ctc_fusion import CTCPlateReader
        from ctc_fixtures import TOKENS,rows_for
        reader=CTCPlateReader(TOKENS,allow_any_text=True)
        for index in range(3):
            reader.add(rows_for(list('TOYOTA')),timestamp_ms=index*250,source_id=str(index))
        self.assertEqual(reader.candidates()[0].plate,'TOYOTA')
        self.assertIsNone(reader.accepted())


if __name__=='__main__':unittest.main()
