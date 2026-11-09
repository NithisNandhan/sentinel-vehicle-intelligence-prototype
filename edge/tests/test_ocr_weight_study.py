import unittest

from sentinel_edge.ocr_weight_study import character_errors, disjoint_training, metrics, promotion_decision
from sentinel_edge.awiros_experiment import adaptation_keys


class WeightStudyTests(unittest.TestCase):
    def test_expanded_scope_keeps_auxiliary_decoder_frozen(self):
        state = dict.fromkeys(['backbone.conv.weight','backbone.norm._mean',
                              'head.ctc_encoder.weight','head.ctc_head.fc.weight',
                              'head.nrtr_head.weight'])
        keys = adaptation_keys(state,'full_image_encoder_and_ctc')
        self.assertNotIn('head.nrtr_head.weight',keys)
        self.assertEqual(len(keys),4)
        with self.assertRaises(ValueError):
            adaptation_keys({'head.ctc_head.fc.weight':None},'full_image_encoder_and_ctc')
    def test_missing_and_repeated_characters(self):
        self.assertEqual(character_errors('MH011234', 'MH01234'),
                         dict(total=1, missing=1, extra=0, substituted=0))
        self.assertEqual(character_errors('AB12', 'AB112')['extra'], 1)
        self.assertEqual(character_errors('AB12', 'AB17')['substituted'], 1)

    def test_metrics_do_not_treat_model_confidence_as_correctness(self):
        result = metrics([dict(expected='AB12', reading='AB12'), dict(expected='AB12', reading='AB2')])
        self.assertEqual((result['exact'], result['missing'], result['cer']), (1, 1, .125))

    def test_train_evaluation_registration_and_pixel_overlap_rejected(self):
        training = [dict(transcription='AB1234', crop_sha256='a')]
        for row in (dict(transcription='AB1234', crop_sha256='b'), dict(transcription='CD5678', crop_sha256='a')):
            with self.assertRaises(ValueError):
                disjoint_training(training, [row])
        disjoint_training(training, [dict(transcription='CD5678', crop_sha256='b')])

    def test_no_promotion_for_validation_tie_or_external_regression(self):
        base = {name: dict(exact=10, character_errors=12) for name in ('validation','manual','external','corpus_test')}
        self.assertFalse(promotion_decision(base, base)[0])
        candidate = {**base, 'validation':dict(exact=11, character_errors=10)}
        self.assertTrue(promotion_decision(base, candidate)[0])
        candidate['external'] = dict(exact=9, character_errors=10)
        self.assertFalse(promotion_decision(base, candidate)[0])


if __name__ == '__main__':
    unittest.main()
