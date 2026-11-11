"""Saved-output viewer checks with synthetic images; no model or OpenCV needed."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from render_crop_comparison import ROOT, sha
from render_low_contrast_comparison import render, sources_for_sample
import test_crop_comparison as fixtures
from test_crop_comparison import group, write


class LowContrastComparisonTests(unittest.TestCase):
    def fixture(self, root):
        old, new, review = fixtures.CropComparisonTests().make_fixture(root)
        document = json.loads((new/'predictions.json').read_bytes())
        document['low_contrast_search'] = dict(character_guided=True, contrast_pass=True)
        document['search_diagnostics'] = dict(low_contrast=dict(contrast_calls=7, character_proposals=4))
        first = document['groups'][0]
        first['samples'][0]['proposal_sources'] = ['contrast_plate', 'character_group']
        first['observed'][0]['proposal_sources'] = ['contrast_plate']
        for i in range(10):
            extra = group('extra-'+str(i), '</script><script>bad</script>' if i == 0 else 'UNKNOWN'+str(i), excluded=i == 1)
            extra['samples'][0]['source_id'] = first['samples'][0]['source_id']
            extra['samples'][0]['proposal_sources'] = ['character_group']
            extra['observed'][0]['proposal_sources'] = ['character_group']
            document['groups'].append(extra)
        self.save(new, document)
        return old, new, review

    def save(self, capture, document):
        write(capture/'predictions.json', document)
        accuracy = json.loads((capture/'accuracy.json').read_bytes())
        accuracy['predictions_sha256'] = sha(capture/'predictions.json')
        write(capture/'accuracy.json', accuracy)

    def temporary(self):
        (ROOT/'data'/'benchmarks').mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='contrast-view-unit-', dir=ROOT/'data'/'benchmarks')

    def test_selected_source_attribution_does_not_borrow_unrelated_observation(self):
        item = dict(observed=[dict(source_id='a', proposal_sources=['character_group']),
                              dict(source_id='b', proposal_sources=['contrast_plate'])])
        self.assertEqual(sources_for_sample(item, dict(source_id='b')), ['contrast_plate'])
        self.assertEqual(sources_for_sample(item, dict(source_id='c')), [])
        self.assertEqual(sources_for_sample(item, dict(source_id='b', proposal_sources=['existing_plate'])), ['existing_plate'])

    def test_render_preserves_scores_inputs_and_native_pixels_and_labels_unknown_candidates(self):
        with self.temporary() as folder:
            root = Path(folder)
            old, new, review = self.fixture(root)
            original = {p: sha(p) for p in root.rglob('*') if p.is_file()}
            # Image transform integration is tested in the detector suite. The
            # viewer test needs no local OpenCV installation or model imports.
            with patch('render_low_contrast_comparison.contrast_preview', side_effect=shutil.copyfile) as preview:
                data = render(old, new, review, root/'output')
            self.assertEqual(preview.call_count, 1)
            self.assertEqual(original, {p: sha(p) for p in original})
            self.assertEqual(data['scores']['previous']['correct'], 0)
            self.assertEqual(data['scores']['current']['correct'], 1)
            self.assertEqual(data['scores']['current']['total'], 1)
            self.assertEqual(data['results'][0]['current']['proposal_sources'], ['character_group', 'contrast_plate'])
            self.assertEqual(data['experiment']['candidate_histories'], 11)
            self.assertFalse(data['experiment']['scoring_changed'])
            self.assertEqual(data['experiment']['diagnostics']['contrast_calls'], 7)
            self.assertTrue(data['new_candidates'][2]['excluded'])
            self.assertEqual(data['new_candidates'][1]['correctness'], 'Unlabelled candidate: correctness unknown.')
            for candidate in data['new_candidates']:
                copied = root/'output'/candidate['crop']
                self.assertEqual(sha(copied), sha(new/'assets/crop.png'))
                self.assertEqual(candidate['crop_sha256'], sha(copied))
            page = (root/'output/index.html').read_text(encoding='utf-8')
            for expected in ('Choose reference', 'Previous automatic crop', 'Additional-source candidate review',
                             'const size=8', "rows.slice(page*size,(page+1)*size)", 'not the OCR input',
                             'surrounding pixels differ', 'Unlabelled candidate: correctness unknown.'):
                self.assertIn(expected, page)
            self.assertNotIn('</script><script>bad</script>', page)
            self.assertIn('\\u003c/script\\u003e', page)
            self.assertNotIn('https://', page)

    def test_new_candidate_unsafe_asset_fails_before_output_creation(self):
        with self.temporary() as folder:
            root = Path(folder)
            old, new, review = self.fixture(root)
            document = json.loads((new/'predictions.json').read_bytes())
            document['groups'][1]['samples'][0]['path'] = '../review/ref.png'
            self.save(new, document)
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                render(old, new, review, root/'output')
            self.assertFalse((root/'output').exists())

    def test_new_candidate_tampered_native_hash_fails_before_output_creation(self):
        with self.temporary() as folder:
            root = Path(folder)
            old, new, review = self.fixture(root)
            document = json.loads((new/'predictions.json').read_bytes())
            document['groups'][1]['samples'][0]['source_id'] = 'changed'
            self.save(new, document)
            with self.assertRaisesRegex(ValueError, 'native source hash'):
                render(old, new, review, root/'output')
            self.assertFalse((root/'output').exists())

    def test_changed_models_or_missing_explicit_experiment_are_rejected(self):
        for mode in ('models', 'weights', 'flags'):
            with self.subTest(mode=mode), self.temporary() as folder:
                root = Path(folder)
                old, new, review = self.fixture(root)
                document = json.loads((new/'predictions.json').read_bytes())
                if mode == 'models':
                    document['models'] = dict(plate='replacement-model')
                elif mode == 'weights':
                    before = json.loads((old/'predictions.json').read_bytes())
                    before['models'] = dict(plate_model='same-model', plate_model_sha256='original', plate_device='gpu')
                    self.save(old, before)
                    document['models'] = dict(plate_model='same-model', plate_model_sha256='different', plate_device='cpu')
                else:
                    document.pop('low_contrast_search')
                self.save(new, document)
                with self.assertRaises(ValueError):
                    render(old, new, review, root/'output')
                self.assertFalse((root/'output').exists())

    def test_device_only_difference_is_allowed_and_visible_without_claiming_weight_change(self):
        with self.temporary() as folder:
            root = Path(folder)
            old, new, review = self.fixture(root)
            for capture, device in [(old, 'gpu'), (new, 'cpu')]:
                document = json.loads((capture/'predictions.json').read_bytes())
                document['models'] = dict(ocr_model='same-ocr', ocr_checkpoint_sha256='same-weights',
                    plate_model='same-detector', plate_model_sha256='same-plate-weights', detector_profile='general',
                    ocr_device=device, plate_device=device, vehicle_device=device, detector_device=device)
                self.save(capture, document)
            with patch('render_low_contrast_comparison.contrast_preview', side_effect=shutil.copyfile):
                data = render(old, new, review, root/'output')
            runtime = data['experiment']['model_runtime']
            self.assertTrue(runtime['devices_changed'])
            self.assertTrue(runtime['model_identity_equal'])
            self.assertFalse(data['experiment']['model_weights_changed'])
            self.assertEqual(runtime['previous']['ocr_device'], 'gpu')
            self.assertEqual(runtime['current']['ocr_device'], 'cpu')
            page = (root/'output/index.html').read_text(encoding='utf-8')
            self.assertIn('Previous devices', page)
            self.assertIn('New devices', page)
            self.assertIn('not a controlled device-performance comparison', page)


if __name__ == '__main__':
    unittest.main()
