import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image
from sentinel_edge.plate_corpus import Corpus, SOURCES, indian_style, normalized_label, orient_native, pixel_hash, safe_member, split_for, verify_corpus


def provenance(source='bapatla', origin='photograph_publisher_declared'):
    return {'source': source, 'origin': origin, 'member': 'example.png',
            'license': SOURCES[source]['license'], 'source_url': SOURCES[source]['url'],
            'attribution': SOURCES[source]['attribution'], 'original_sha256': 'source'}


class CorpusTests(unittest.TestCase):
    def test_format_filter_does_not_correct_characters(self):
        self.assertEqual(normalized_label('mh 02-ek0837'), 'MH02EK0837')
        self.assertTrue(indian_style('MH02EK0837'))
        self.assertTrue(indian_style('23BH1234AB'))
        self.assertFalse(indian_style('MH02EKO837'))
        self.assertFalse(indian_style('XX02AB1234'))
        self.assertFalse(indian_style('MH02EK837'))

    def test_archive_escape_symlink_and_size(self):
        for name in ('../escape.png', '/root.png', 'C:/root.png', '..\\escape.png'):
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                safe_member(zipfile.ZipInfo(name))
        info = zipfile.ZipInfo('safe.png')
        info.external_attr = (0o120777 << 16)
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            safe_member(info)
        info = zipfile.ZipInfo('safe.png')
        info.file_size = 20 * 1024**2
        with self.assertRaisesRegex(ValueError, 'bounds'):
            safe_member(info)

    def test_pixel_identity_independent_of_png_encoding(self):
        image = Image.new('RGB', (80, 20), 'white')
        first, second = io.BytesIO(), io.BytesIO()
        image.save(first, 'PNG', compress_level=0)
        image.save(second, 'PNG', compress_level=9)
        self.assertNotEqual(first.getvalue(), second.getvalue())
        self.assertEqual(pixel_hash(Image.open(first)), pixel_hash(Image.open(second)))

    def test_source_overlap_and_generator_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(tmp, min_free=0)
            image = Image.new('RGB', (80, 20), 'white')
            corpus.add(image, 'AP05BY7799', provenance('umar', 'mixed_origin_unverified'))
            corpus.add(image, 'AP05BY7799', provenance('abtexp', 'synthetic'), review='synthetic_generated')
            summary = corpus.finish()
            self.assertEqual(summary['unique_crops'], 1)
            self.assertEqual(summary['observations']['exact_decoded_duplicates'], 1)
            row = next(iter(corpus.rows.values()))
            self.assertEqual(row['origin'], 'synthetic')
            self.assertEqual(row['review'], 'synthetic_generated')
            self.assertEqual(len(row['provenance']), 2)
            with Image.open(Path(tmp) / row['crop_path']) as crop:
                self.assertEqual(crop.tobytes(), image.tobytes())

    def test_conflict_quarantines_both_labels_and_other_encodings(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(tmp, min_free=0)
            white = Image.new('RGB', (80, 20), 'white')
            corpus.add(white, 'AP05BY7799', provenance())
            corpus.add(white, 'AP05BY7789', provenance('umar'))
            corpus.add(Image.new('RGB', (80, 20), 'black'), 'AP05BY7799', provenance())
            corpus.finish()
            self.assertTrue(all(r['quarantined'] and not r['training_eligible'] for r in corpus.rows.values()))

    def test_legacy_and_temporal_registrations_never_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(tmp, min_free=0, legacy={'AP05BY7799': 'test'}, reserved={'MH02EK0837'})
            corpus.add(Image.new('RGB', (80, 20), 'white'), 'AP05BY7799', provenance())
            corpus.add(Image.new('RGB', (80, 20), 'black'), 'MH02EK0837', provenance('abtexp', 'synthetic'))
            corpus.finish()
            self.assertEqual({r['split'] for r in corpus.rows.values()}, {'legacy_test', 'regression_only'})
            self.assertFalse(any(r['training_eligible'] for r in corpus.rows.values()))

    def test_source_independent_registration_split(self):
        for number in range(1000, 1020):
            label = f'AP05BY{number}'
            self.assertEqual(split_for(label, synthetic=True), split_for(label, synthetic=False))

    def test_publisher_review_not_promoted_and_holdout_not_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(tmp, min_free=0)
            corpus.add(Image.new('RGB', (80, 20), 'white'), 'AP05BY7799', provenance())
            corpus.finish()
            row = json.loads((Path(tmp) / 'corpus.jsonl').read_text())
            self.assertEqual(row['review'], 'publisher_unverified')
            self.assertFalse(row['independent_evaluation_eligible'])
            self.assertNotEqual((Path(tmp) / 'review_queue.jsonl').read_text(), '')

    def test_modified_cached_pixels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Image.new('RGB', (80, 20), 'white')
            corpus = Corpus(tmp, min_free=0)
            corpus.add(image, 'AP05BY7799', provenance())
            row = next(iter(corpus.rows.values()))
            Image.new('RGB', (80, 20), 'black').save(Path(tmp) / row['crop_path'])
            with self.assertRaisesRegex(ValueError, 'modified'):
                Corpus(tmp, min_free=0).add(image, 'AP05BY7799', provenance())

    def test_unknown_format_kept_in_rejection_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(tmp, min_free=0)
            corpus.add(Image.new('RGB', (80, 20), 'white'), 'PARTIAL', provenance())
            summary = corpus.finish()
            self.assertEqual(summary['unique_crops'], 0)
            self.assertIn('unsupported_or_non_indian_label', (Path(tmp) / 'rejected.jsonl').read_text())

    def test_extension_preserves_prior_observations_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Corpus(tmp, min_free=0)
            first.add(Image.new('RGB', (80, 20), 'white'), 'AP05BY7799', provenance())
            first.finish()
            second = Corpus(tmp, min_free=0)
            self.assertEqual(second.load_existing(), {'bapatla'})
            second.add(Image.new('RGB', (80, 20), 'black'), 'MH02EK0837', provenance('umar'))
            self.assertEqual(second.finish()['unique_crops'], 2)
            (Path(tmp) / 'corpus.jsonl').write_text('modified')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                Corpus(tmp, min_free=0).load_existing()

    def test_final_integrity_checks_pixels_and_training_indexes_not_label_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Corpus(tmp, min_free=0)
            corpus.add(Image.new('RGB', (80, 20), 'white'), 'AP05BY7799', provenance())
            corpus.finish()
            result = verify_corpus(tmp)
            self.assertEqual(result['verified_native_crops'], 1)
            self.assertFalse(result['label_truth_verified_by_this_check'])
            row = next(iter(corpus.rows.values()))
            Image.new('RGB', (80, 20), 'black').save(Path(tmp) / row['crop_path'])
            with self.assertRaisesRegex(ValueError, 'modified'):
                verify_corpus(tmp)

    def test_exif_normalisation_permutates_without_changing_pixels(self):
        image = Image.new('RGB', (3, 2))
        image.putdata([(i, i + 10, i + 20) for i in range(6)])
        original = sorted(image.getpixel((x, y)) for y in range(image.height) for x in range(image.width))
        for orientation in range(1, 9):
            exif = image.getexif()
            exif[274] = orientation
            oriented, actual = orient_native(image)
            self.assertEqual(actual, orientation)
            pixels = sorted(oriented.getpixel((x, y)) for y in range(oriented.height) for x in range(oriented.width))
            self.assertEqual(pixels, original)
            self.assertEqual(image.size, (3, 2))
        image.getexif()[274] = 99
        with self.assertRaises(ValueError):
            orient_native(image)


if __name__ == '__main__':
    unittest.main()
