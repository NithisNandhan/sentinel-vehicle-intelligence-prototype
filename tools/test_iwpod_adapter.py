"""Geometry and pinned official decoder parity without requiring TensorFlow."""
import unittest
from unittest.mock import patch
from io import BytesIO
import hashlib
from pathlib import Path
import sys
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iwpod_adapter import settings, native_predictions, checked_modules
import iwpod_adapter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/benchmarks/detector-corners-20261112/iwpod-source'


class Label:
    def __init__(self, points, confidence=.9):
        self.pts = np.array(points)
        self.confidence = confidence
    def prob(self):
        return self.confidence


class IWPODAdapterTests(unittest.TestCase):
    def test_prepare_pins_download_and_reuses_verified_cache(self):
        content = b'fixed official source'
        expected = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as folder, patch.object(iwpod_adapter, 'HASHES', {'src/test.py': expected}), \
                patch.object(iwpod_adapter, 'DOCUMENT_HASHES', {}), \
                patch('urllib.request.urlopen', return_value=BytesIO(content)) as download:
            receipt = iwpod_adapter.prepare_assets(Path(folder))
            self.assertEqual(receipt['files'][0]['sha256'], expected)
            request = download.call_args.args[0]
            self.assertIn('/' + iwpod_adapter.REVISION + '/src/test.py', request.full_url)
            download.reset_mock()
            iwpod_adapter.prepare_assets(Path(folder))
            download.assert_not_called()
            self.assertTrue((Path(folder) / 'receipt.json').exists())

    def test_prepare_hash_failure_preserves_existing_asset_and_cleans_partial(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(iwpod_adapter, 'HASHES', {'test.bin': '0' * 64}), \
                patch.object(iwpod_adapter, 'DOCUMENT_HASHES', {}), \
                patch('urllib.request.urlopen', return_value=BytesIO(b'changed bytes')):
            destination = Path(folder) / 'test.bin'
            destination.write_bytes(b'existing unverified cache')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                iwpod_adapter.prepare_assets(Path(folder))
            self.assertEqual(destination.read_bytes(), b'existing unverified cache')
            self.assertFalse(list(Path(folder).rglob('*.partial')))
            self.assertFalse((Path(folder) / 'receipt.json').exists())

    def test_prepare_size_failure_cleans_partial_without_publishing(self):
        content = b'too large'
        with tempfile.TemporaryDirectory() as folder, patch.object(iwpod_adapter, 'HASHES',
                {'test.bin': hashlib.sha256(content).hexdigest()}), \
                patch.object(iwpod_adapter, 'DOCUMENT_HASHES', {}), \
                patch.object(iwpod_adapter, '_asset_limit', return_value=4), \
                patch('urllib.request.urlopen', return_value=BytesIO(content)):
            with self.assertRaisesRegex(ValueError, 'bounded download size'):
                iwpod_adapter.prepare_assets(Path(folder))
            self.assertFalse((Path(folder) / 'test.bin').exists())
            self.assertFalse(list(Path(folder).rglob('*.partial')))

    def test_settings_match_author_example(self):
        self.assertEqual(settings((100, 200, 3), 'car'), (512., (240, 80)))
        self.assertEqual(settings((100, 400, 3), 'truck'), (704., (240, 80)))
        self.assertEqual(settings((300, 100, 3), 'bus'), (256., (240, 80)))
        self.assertEqual(settings((300, 100, 3), 'bike'), (208, (120, 80)))
        self.assertEqual(settings((300, 100, 3)), (480, (240, 80)))

    def test_native_mapping_and_bgr_uint8(self):
        image = np.zeros((100, 200, 3), np.uint8)
        quad = [[.1, .8, .8, .1], [.2, .2, .6, .6]]
        working = np.ones((80, 240, 3), np.float32) * .5
        result = native_predictions([Label(quad)], [working], image)[0]
        np.testing.assert_allclose(result['box'], [20, 20, 160, 60])
        np.testing.assert_allclose(result['quad'], [[20, 20], [160, 20], [160, 60], [20, 60]])
        self.assertEqual(result['rectified'].dtype, np.uint8)
        self.assertEqual(int(result['rectified'][0, 0, 0]), 128)
        self.assertFalse(result['boundary_clipped'])

    def test_outside_corners_remain_measured_not_silently_moved(self):
        image = np.zeros((100, 200, 3), np.uint8)
        result = native_predictions([Label([[-.1, .8, .8, -.1], [.2, .2, .6, .6]])],
            [np.zeros((80, 240, 3))], image)[0]
        self.assertEqual(result['box'][0], 0)
        self.assertEqual(result['quad'][0][0], -20)
        self.assertTrue(result['boundary_clipped'])

    def test_invalid_quads_and_empty_intersection_rejected(self):
        image = np.zeros((100, 200, 3), np.uint8)
        bad = [Label([[0, 1, 0, 1], [0, 1, 1, 0]]),
               Label([[2, 3, 3, 2], [.2, .2, .6, .6]]),
               Label([[0, 1, 1, 0], [0, 0, float('nan'), 1]])]
        self.assertEqual(native_predictions(bad, [np.zeros((80, 240, 3))] * 3, image), [])

    @unittest.skipUnless(SOURCE.exists(), 'Pinned official assets not cached')
    def test_official_decoder_synthetic_geometry_and_preprocess(self):
        upstream, utils = checked_modules(SOURCE)
        image = np.zeros((100, 200, 3), np.uint8)
        image[:, :, 0] = 255
        seen = []
        class FakeModel:
            def predict(self, tensor):
                seen.append(tensor.copy())
                result = np.zeros((1, 7, 13, 7), np.float32)
                result[0, 3, 5, 0] = .9
                result[0, 3, 5, 1:] = [1, 0, 0, 0, .3, 0]
                return result
        labels, rectified, _ = upstream.detect_lp_width(FakeModel(), utils.im2single(image),
            480, 16, (240, 80), .35)
        self.assertEqual(seen[0].shape, (1, 112, 208, 3))
        self.assertAlmostEqual(float(seen[0][0, 0, 0, 0]), 1., places=6)
        self.assertEqual(float(seen[0][0, 0, 0, 2]), 0.)
        self.assertEqual(len(labels), 1)
        # Official affine decode: cell centre(5.5,3.5), side7.75, grid(13,7).
        expected = np.array([[5.5-3.875, 5.5+3.875, 5.5+3.875, 5.5-3.875],
                             [3.5-1.1625, 3.5-1.1625, 3.5+1.1625, 3.5+1.1625]])
        expected /= np.array([[13], [7]])
        np.testing.assert_allclose(labels[0].pts, expected, atol=1e-7)
        output = native_predictions(labels, rectified, image)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]['rectified'].shape, (80, 240, 3))


if __name__ == '__main__':
    unittest.main()
