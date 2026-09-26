import unittest

from sentinel_edge.sequence_fusion import SequencePlateReader, token_alignment

ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_?'


def matrix(text, confidence=.97):
    output = []
    for char in text.ljust(13, '_'):
        row = [(1 - confidence) / (len(ALPHABET) - 1)] * len(ALPHABET)
        row[ALPHABET.index(char)] = confidence
        output.append(row)
    return output


class SequenceTests(unittest.TestCase):
    def reader(self):
        return SequencePlateReader(ALPHABET, '_')

    def test_deleted_character_does_not_shift_the_suffix(self):
        mapping, cost = token_alignment('GJ01AB1234', 'GJ01A1234')
        self.assertEqual(cost, 1)
        self.assertIsNone(mapping[5])
        self.assertEqual(mapping[6:], [5, 6, 7, 8])

    def test_one_deleted_read_does_not_count_as_character_confirmation(self):
        reader = self.reader()
        for index, text in enumerate(('GJ01AB1234', 'GJ01A1234', 'GJ01AB1234')):
            reader.add(matrix(text), timestamp_ms=index * 300)
        self.assertEqual(reader.candidates()[0].plate, 'GJ01AB1234')
        # Missing vote remains unknown mass; do not renormalise it away.
        self.assertIsNone(reader.accepted(.85))
        reader.add(matrix('GJ01AB1234'), timestamp_ms=900)
        self.assertIsNone(reader.accepted(.85))  # .97 * 3 / 4 < weakest gate
        reader.add(matrix('GJ01AB1234'), timestamp_ms=1200)
        self.assertEqual(reader.accepted(.85).plate, 'GJ01AB1234')

    def test_inserted_character_is_aligned_not_shifted(self):
        reader = self.reader()
        for index, text in enumerate(('GJ01AB1234', 'GJ01ABX1234', 'GJ01AB1234')):
            reader.add(matrix(text), timestamp_ms=index * 300)
        self.assertEqual(reader.candidates()[0].plate, 'GJ01AB1234')
        self.assertIsNone(reader.accepted(.85))  # Conflicting EOS has no confirmation.

    def test_consistent_sequence_passes_without_score_inflation(self):
        reader = self.reader()
        for index in range(3):
            reader.add(matrix('GJ01AB1234'), timestamp_ms=index * 300)
        self.assertAlmostEqual(reader.accepted(.85).score, .97)

    def test_same_time_and_restored_copy_cannot_add_support(self):
        reader = self.reader()
        self.assertTrue(reader.add(matrix('GJ01AB1234'), timestamp_ms=1000))
        self.assertFalse(reader.add(matrix('GJ01AB1234'), timestamp_ms=1000))
        self.assertEqual(reader.observation_count, 1)
        self.assertIsNone(reader.accepted(.85))

    def test_unknown_and_empty_return_partial_not_fake_plate(self):
        reader = self.reader()
        for index in range(3):
            reader.add(matrix('GJ01??1234'), timestamp_ms=index * 300)
        self.assertIsNone(reader.accepted(.85))
        self.assertIn('?', reader.partial())
        self.assertEqual(self.reader().partial(), '')
