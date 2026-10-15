"""Mocked device-routing tests; these are not GPU speed/accuracy measurements."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sentinel_edge.experimental_detectors import PlateDetector, VehicleTracker


class GpuDetectorTests(unittest.TestCase):
    @contextmanager
    def vehicle_runtime(self, available=True):
        model = Mock(names=dict.fromkeys(range(80)))
        model.track.return_value = [SimpleNamespace(boxes=[])]
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available),
                                set_num_threads=Mock())
        yolo = Mock(return_value=model)
        with patch.dict('sys.modules', {'torch': torch,
                                      'ultralytics': SimpleNamespace(YOLO=yolo)}), \
                patch('sentinel_edge.experimental_detectors.verified', side_effect=lambda path, _: path):
            yield model, yolo

    @contextmanager
    def plate_runtime(self, *, available=True, actual_gpu=True):
        calls = []
        class Session:
            def __init__(self, path=None, options=None, *, providers=None):
                self.providers = providers or ['CPUExecutionProvider']
                self.fallback_disabled = False
                calls.append(self)
            def get_providers(self):
                return (self.providers if actual_gpu else ['CPUExecutionProvider'])
            def disable_fallback(self):
                self.fallback_disabled = True
            def get_inputs(self):
                return [SimpleNamespace(shape=[1, 3, 512, 512], name='images')]
            def get_outputs(self):
                return [SimpleNamespace(shape=[1, 5, 5376])]
        runtime = SimpleNamespace(
            SessionOptions=SimpleNamespace, InferenceSession=Session,
            get_available_providers=lambda: (['CUDAExecutionProvider', 'CPUExecutionProvider']
                                            if available else ['CPUExecutionProvider']))
        def parent(name, cache, device):
            providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                         if device == 'cuda' else ['CPUExecutionProvider'])
            session = Session(providers=providers)
            # Exercise the nested wrapper layout supported by PlateModel.
            return SimpleNamespace(providers=session.get_providers(),
                                   model=SimpleNamespace(backend=SimpleNamespace(session=session)))
        parent_factory = Mock(side_effect=parent)
        receipt = {'input_shape': [1, 3, 512, 512], 'output_shape': [1, 5, 5376]}
        with patch.dict('sys.modules', {'onnxruntime': runtime}), \
                patch('sentinel_edge.experimental_detectors.verified', side_effect=lambda path, _: path), \
                patch('sentinel_edge.experimental_detectors.PlateModel', parent_factory), \
                patch.object(Path, 'read_text', return_value='{}'), \
                patch('sentinel_edge.experimental_detectors.json.loads', return_value=receipt):
            yield calls, parent_factory

    def test_vehicle_default_remains_cpu(self):
        with self.vehicle_runtime() as (model, _):
            tracker = VehicleTracker('yolo26s', '/models', '/training')
            self.assertEqual(tracker.device, 'cpu')
            self.assertEqual(tracker.track(None), [])
            self.assertEqual(model.track.call_args.kwargs['device'], 'cpu')
            model.to.assert_not_called()

    def test_vehicle_gpu_moves_model_and_routes_each_inference(self):
        with self.vehicle_runtime() as (model, _):
            tracker = VehicleTracker('yolo26s', '/models', '/training', device='gpu')
            tracker.track(None)
            self.assertEqual(tracker.device, 'gpu')
            model.to.assert_called_once_with('cuda:0')
            self.assertEqual(model.track.call_args.kwargs['device'], 0)

    def test_vehicle_gpu_unavailable_is_not_retried_on_cpu(self):
        with self.vehicle_runtime(available=False) as (_, factory):
            with self.assertRaisesRegex(RuntimeError, 'CUDA runtime is unavailable'):
                VehicleTracker('yolo26s', '/models', '/training', device='gpu')
            factory.assert_not_called()

    def test_parent_plate_default_remains_cpu(self):
        with self.plate_runtime() as (sessions, factory):
            detector = PlateDetector('parent', '/models')
            self.assertEqual(detector.device, 'cpu')
            self.assertEqual(factory.call_args.args[-1], 'cpu')
            self.assertEqual(detector.providers, ['CPUExecutionProvider'])
            self.assertFalse(sessions[0].fallback_disabled)

    def test_parent_plate_gpu_routes_and_disables_nested_session_fallback(self):
        with self.plate_runtime() as (sessions, factory):
            detector = PlateDetector('parent', '/models', device='gpu')
            self.assertEqual(detector.device, 'gpu')
            self.assertEqual(factory.call_args.args[-1], 'cuda')
            self.assertIn('CUDAExecutionProvider', detector.providers)
            self.assertTrue(sessions[0].fallback_disabled)

    def test_adapted_plate_cpu_and_gpu_providers_are_explicit(self):
        for device in ('cpu', 'gpu'):
            with self.subTest(device=device), self.plate_runtime() as (sessions, _):
                detector = PlateDetector('adapted', '/models', device=device)
                self.assertEqual(detector.device, device)
                self.assertEqual(sessions[0].providers,
                                 ['CUDAExecutionProvider', 'CPUExecutionProvider']
                                 if device == 'gpu' else ['CPUExecutionProvider'])
                self.assertEqual(sessions[0].fallback_disabled, device == 'gpu')

    def test_plate_gpu_unavailable_fails_before_creating_models(self):
        for name in ('parent', 'adapted'):
            with self.subTest(name=name), self.plate_runtime(available=False) as (sessions, factory):
                with self.assertRaisesRegex(RuntimeError, 'CUDA provider is unavailable'):
                    PlateDetector(name, '/models', device='gpu')
                self.assertEqual(sessions, [])
                factory.assert_not_called()

    def test_plate_gpu_actual_cpu_fallback_is_rejected(self):
        for name in ('parent', 'adapted'):
            with self.subTest(name=name), self.plate_runtime(actual_gpu=False):
                with self.assertRaisesRegex(RuntimeError, 'fell back to CPU'):
                    PlateDetector(name, '/models', device='gpu')

    def test_invalid_devices_rejected_before_importing_runtimes(self):
        for device in ('cuda', 'auto', None, '', 0):
            with self.subTest(device=device):
                with self.assertRaises(ValueError):
                    VehicleTracker('yolo26s', '/models', '/training', device=device)
                with self.assertRaises(ValueError):
                    PlateDetector('parent', '/models', device=device)


if __name__ == '__main__':
    unittest.main()
