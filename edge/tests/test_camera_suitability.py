import contextlib
import io
import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sentinel_edge.camera_suitability import run


@dataclass
class Config:
    media_relay_base_url: str = 'rtsp://mediamtx:8554'


class CameraScreeningTests(unittest.TestCase):
    def test_invalid_scope_rejected_before_loading_models(self):
        with self.assertRaises(ValueError): run(None, None, None, [], 45, None)
        with patch('sentinel_edge.camera_suitability.discover', return_value=[]):
            with self.assertRaises(ValueError): run(Config(), None, None, ['cam12'], 45, None)

    def test_report_contains_only_metrics_and_cameras_processed_sequentially(self):
        output, seen = [], []
        report = SimpleNamespace(resolve=lambda: Path('/workspace/data/benchmarks/reports/test.json'),
                                 suffix='.json', write_text=output.append)
        cameras = [SimpleNamespace(camera_id=c, live=True) for c in ('cam12', 'cam14')]
        def observe(config, camera, model, detector, vehicle, state, stop, duration_seconds):
            self.assertEqual(config.media_relay_base_url, '')
            seen.append(camera.camera_id)
            state.publish({'camera_id': camera.camera_id, 'decoded_frames': 12, 'analysed_frames': 3,
                           'frame': 'genuine pixels', 'crop': {'image': 'private image'},
                           'plate': 'AP01AB1234', 'resolution': [1920, 1080]})
        with patch('sentinel_edge.camera_suitability.discover', return_value=cameras), \
                patch('sentinel_edge.camera_suitability.load_models', return_value=(None, None, None)), \
                patch('sentinel_edge.camera_suitability.observe', side_effect=observe), \
                contextlib.redirect_stdout(io.StringIO()):
            run(Config(), None, None, ['cam12', 'cam14'], 45, report)
        self.assertEqual(seen, ['cam12', 'cam14'])
        document = json.loads(output[-1])
        self.assertIsNone(document['accuracy'])
        self.assertEqual(len(document['cameras']), 2)
        self.assertNotIn('genuine pixels', output[-1])
        self.assertNotIn('AP01AB1234', output[-1])
        self.assertFalse(document['recordings_saved'])


if __name__ == '__main__':
    unittest.main()
