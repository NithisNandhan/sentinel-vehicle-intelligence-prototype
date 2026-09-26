import json
import tempfile
import unittest
from pathlib import Path

from sentinel_edge.awiros_experiment import WEIGHTS_SHA256, checkpoint_metadata, file_hash
from sentinel_edge.indian_ocr_finetune import edit_distance, validate_samples


class AdaptationGuardsTests(unittest.TestCase):
    def test_character_error_distance(self):
        self.assertEqual(edit_distance('AP05BY7799', 'AP05BY7799'), 0)
        self.assertEqual(edit_distance('AP05BY779', 'AP05BY7799'), 1)
        self.assertEqual(edit_distance('AB', 'BA'), 2)

    def fixture(self, root):
        rows = []
        for index in range(8):
            image, crop = root / f'image-{index}', root / f'crop-{index}'
            image.write_bytes(f'original-{index}'.encode())
            crop.write_bytes(f'crop-{index}'.encode())
            rows.append({'path': image.name, 'crop_path': crop.name,
                         'sha256': file_hash(image), 'crop_sha256': file_hash(crop),
                         'transcription': f'AP05BY{7790 + index}', 'review': 'visually_verified',
                         'split': 'train' if index < 4 else 'validation' if index < 6 else 'test'})
        return {'dataset': {'license': 'CC-BY-4.0'}, 'samples': rows}

    def test_clean_split_and_label_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            doc = self.fixture(root)
            self.assertEqual(len(validate_samples(doc, root)), 8)
            doc['samples'][-1]['transcription'] = doc['samples'][0]['transcription']
            with self.assertRaisesRegex(ValueError, 'leakage'):
                validate_samples(doc, root)

    def test_review_hash_and_duplicate_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mutation in ('review', 'hash', 'duplicate', 'escape'):
                doc = self.fixture(root)
                if mutation == 'review':
                    doc['samples'][0]['review'] = 'pending'
                elif mutation == 'hash':
                    (root / doc['samples'][0]['crop_path']).write_bytes(b'modified')
                elif mutation == 'duplicate':
                    doc['samples'][1] = dict(doc['samples'][0])
                else:
                    doc['samples'][0]['crop_path'] = '../outside'
                with self.assertRaises(ValueError):
                    validate_samples(doc, root)

    def test_explicit_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'en_dict.txt').write_text('dictionary')
            weights = root / 'head.safetensors'
            weights.write_bytes(b'weights')
            metadata = {'parent_sha256': WEIGHTS_SHA256,
                        'dictionary_sha256': file_hash(root / 'en_dict.txt'),
                        'weights_sha256': file_hash(weights), 'training_scope': 'ctc_linear_head_only',
                        'frozen_state_unchanged': True}
            weights.with_suffix('.json').write_text(json.dumps(metadata))
            self.assertEqual(checkpoint_metadata(weights, root), metadata)
            weights.write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                checkpoint_metadata(weights, root)
