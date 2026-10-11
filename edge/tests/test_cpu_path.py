"""CPU wiring only. These tests prove the device is passed through explicitly;
they do not run a real model and say nothing about CPU accuracy or speed."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sentinel_edge.live_detection_viewer import load_models, model_identity, serve


def launcher():
    path = Path(__file__).resolve().parents[2]/'tools'/'run_ocr_research.py'
    spec = importlib.util.spec_from_file_location('run_ocr_research', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CpuPathTests(unittest.TestCase):
    def load(self, **kwargs):
        seen = {}
        def reader(cache, *, device, checkpoint):
            seen.update(device=device, checkpoint=checkpoint)
            return SimpleNamespace(probabilities=lambda image: [], device=device, adaptation=None)
        with patch('sentinel_edge.awiros_experiment.AwirosExperiment', side_effect=reader), \
                patch('sentinel_edge.experimental_detectors.PlateDetector', return_value='plate'), \
                patch('sentinel_edge.experimental_detectors.VehicleTracker', return_value='vehicle'):
            result = load_models(SimpleNamespace(model_cache_dir=Path('/models')), Path('/cache'), None,
                                 detector_profile='general', **kwargs)
        return seen, result

    def test_requested_device_reaches_the_ocr_reader(self):
        seen, result = self.load(device='cpu')
        self.assertEqual(seen['device'], 'cpu')
        self.assertEqual(result[1:], ('plate', 'vehicle'))
        self.assertEqual(self.load()[0]['device'], 'gpu')  # Existing launchers keep their default.

    def test_unknown_device_is_rejected_before_any_model_loads(self):
        with patch('sentinel_edge.awiros_experiment.AwirosExperiment') as reader:
            for device in ('auto', '', None, 'cuda'):
                with self.assertRaises(ValueError):
                    load_models(SimpleNamespace(model_cache_dir=Path('/models')), Path('/cache'), None, device=device)
            reader.assert_not_called()
        with self.assertRaises(ValueError):
            serve(None, None, None, device='auto', operator_key='k')

    def test_a_gpu_failure_is_not_retried_on_cpu(self):
        calls = []
        def reader(cache, *, device, checkpoint):
            calls.append(device)
            raise RuntimeError('Paddle GPU build is unavailable; no silent device fallback')
        with patch('sentinel_edge.awiros_experiment.AwirosExperiment', side_effect=reader):
            with self.assertRaises(RuntimeError):
                load_models(SimpleNamespace(model_cache_dir=Path('/models')), Path('/cache'), None, device='gpu')
        self.assertEqual(calls, ['gpu'])

    def test_identity_reports_device_and_verified_checkpoint_hash_without_paths(self):
        reader = SimpleNamespace(device='cpu', adaptation={'weights_sha256': 'c2ba' + '0'*60})
        identity = model_identity(reader, 'general')
        self.assertEqual(identity['ocr_device'], 'cpu')
        self.assertEqual(identity['ocr_checkpoint_sha256'], 'c2ba' + '0'*60)
        self.assertEqual((identity['vehicle_model'], identity['plate_model']), ('yolo26s', 'parent'))
        self.assertEqual(len(identity['vehicle_model_sha256']), 64)
        self.assertNotIn('/', ''.join(str(value) for value in identity.values()))
        parent = model_identity(SimpleNamespace(device='gpu', adaptation=None))
        self.assertEqual(parent['ocr_model'], 'awiros-ctc-parent')
        self.assertIsNone(parent['ocr_checkpoint_sha256'])

    def test_cpu_launcher_environment_has_no_gpu_libraries(self):
        cache = Path('/models/experiments/awiros')
        cpu = launcher().research_environment({'SENTINEL_OCR_DEVICE': 'cpu', 'LD_LIBRARY_PATH': '/keep'}, cache)
        self.assertEqual(cpu['PYTHONPATH'].split(':'), [str(cache/'deps'), '/workspace/edge'])
        self.assertNotIn('gpu', cpu['PYTHONPATH'])
        self.assertEqual(cpu['LD_LIBRARY_PATH'], '/keep')
        self.assertEqual(cpu['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(cpu['OMP_NUM_THREADS'], '4')
        self.assertNotIn('NVIDIA_TF32_OVERRIDE', cpu)

    def test_gpu_launcher_environment_is_unchanged_by_default(self):
        cache = Path('/models/experiments/awiros')
        gpu = launcher().research_environment({}, cache)
        self.assertEqual(gpu['PYTHONPATH'].split(':')[0], str(cache/'gpu-deps'))
        self.assertEqual(gpu['OMP_NUM_THREADS'], '2')
        self.assertEqual(gpu['NVIDIA_TF32_OVERRIDE'], '0')
        self.assertIn('LD_LIBRARY_PATH', gpu)

    def test_launcher_rejects_ambiguous_settings(self):
        for environment in ({'SENTINEL_OCR_DEVICE': 'auto'}, {'SENTINEL_CPU_THREADS': '0'},
                            {'SENTINEL_CPU_THREADS': 'many'}):
            with self.assertRaises(SystemExit):
                launcher().research_environment(environment)
