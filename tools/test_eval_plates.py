import csv
import tempfile
import unittest
from pathlib import Path

from eval_plates import edit_distance, load_labels, load_predictions, normalise, score


def write(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class EvalPlatesTests(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance('', 'MH01'), 4)
        self.assertEqual(edit_distance('MH01AB1234', 'MH01AB1234'), 0)
        self.assertEqual(edit_distance('MP01AB1234', 'MH01AB1234'), 1)
        self.assertEqual(edit_distance('MH01AB234', 'MH01AB1234'), 1)

    def test_normalise_keeps_capitals_and_digits(self):
        self.assertEqual(normalise(' mh 01-ab 1234 '), 'MH01AB1234')
        self.assertEqual(normalise(None), '')

    def test_scores_each_metric_with_its_denominator(self):
        labels = {'a': 'MH01AB1234', 'b': 'KA05CD9876', 'n1': 'NO_PLATE', 'n2': 'NO_PLATE', 'u': 'UNREADABLE'}
        predictions = {'a': 'MH01AB1234', 'b': 'KA05CD987', 'n1': 'A661', 'n2': '', 'u': 'XYZ'}
        result = score(labels, predictions)
        self.assertEqual((result['readable_n'], result['exact_matches'], result['exact_match_rate']), (2, 1, .5))
        self.assertEqual((result['character_errors'], result['label_characters']), (1, 20))
        self.assertEqual(result['character_error_rate'], .05)
        self.assertEqual((result['no_plate_n'], result['false_reads_on_no_plate'], result['false_read_rate']), (2, 1, .5))
        self.assertEqual((result['unreadable_n'], result['readings_on_unreadable']), (1, 1))

    def test_an_abstention_counts_every_character_as_an_error(self):
        result = score({'a': 'MH01'}, {'a': ''})
        self.assertEqual((result['abstained_on_readable'], result['character_error_rate']), (1, 1.0))

    def test_rates_are_undefined_without_a_denominator(self):
        result = score({'u': 'UNREADABLE'}, {'u': 'MH01'})
        self.assertIsNone(result['exact_match_rate'])
        self.assertIsNone(result['character_error_rate'])
        self.assertIsNone(result['false_read_rate'])

    def test_a_missing_prediction_is_an_error_not_a_pass(self):
        with self.assertRaises(ValueError):
            score({'a': 'MH01', 'b': 'NO_PLATE'}, {'a': 'MH01'})

    def test_label_file_is_validated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.csv'
            write(path, ['crop_id', 'label', 'confidence'], [['a', 'MH01AB1234', 'high'], ['b', 'NO_PLATE', 'high']])
            self.assertEqual(load_labels(path), {'a': 'MH01AB1234', 'b': 'NO_PLATE'})
            for bad in ('mh01', 'MH 01', 'maybe'):
                write(path, ['crop_id', 'label'], [['a', bad]])
                with self.assertRaises(ValueError):
                    load_labels(path)
            write(path, ['crop_id', 'label'], [['a', 'MH01'], ['a', 'MH02']])
            with self.assertRaises(ValueError):
                load_labels(path)

    def test_predictions_file_has_one_column_per_variant(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'predictions.csv'
            write(path, ['crop_id', 'baseline', 'two_line'], [['a', 'mh01 ab', 'MH01AB'], ['b', '', 'X']])
            variants, predictions = load_predictions(path)
            self.assertEqual(variants, ['baseline', 'two_line'])
            self.assertEqual(predictions['baseline'], {'a': 'MH01AB', 'b': ''})


if __name__ == '__main__':
    unittest.main()
