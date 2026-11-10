"""Synthetic wiring checks; no models, real recording or real labels are used."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'edge'))
import run_video_review as runner
import sentinel_edge.config
import sentinel_edge.live_detection_viewer
import sentinel_edge.plate_options
import sentinel_edge.tracked_live_viewer


class ObserverIntegrationTests(unittest.TestCase):
    def run_fixture(self, directory, summary_acceptances=1):
        recording = SimpleNamespace(path=Path('synthetic.mp4'), sha256='b'*64,
                                    recording_id='synthetic', duration_ms=1000, start_pts_ms=0)
        context = MagicMock()
        context.__enter__.return_value.streams.video = [SimpleNamespace(
            width=640, height=480, average_rate=30, frames=30,
            codec_context=SimpleNamespace(name='h264'))]
        args = SimpleNamespace(output=Path(directory)/'fixture', recordings_dir=Path(directory),
                               recording_id='synthetic', cache=Path(directory),
                               detector_profile='general', commit='synthetic-fixture')

        def observe(config, camera, reader, detector, vehicle, state, stop, **kwargs):
            # The callback must reach the collector, without any central client.
            delivery = kwargs.get('delivery')
            self.assertIs(delivery, state)
            delivery.accept('synthetic', 7, SimpleNamespace(plate='MH01AB1234'), 500,
                            recorded=kwargs['recorded'])
            counted = kwargs['recorded']
            counted.decoded = 30
            counted.reached_eof = True
            counted.last_pts = 1000
            state.samples = 2
            return {'unique_model_accepted_unverified': summary_acceptances}

        with patch.object(runner.RecordingLibrary, 'scan', return_value=({'synthetic': recording}, [])), \
             patch.object(runner, 'file_hash', return_value='c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d'), \
             patch.dict(sys.modules, {'av': SimpleNamespace(open=MagicMock(return_value=context))}), \
             patch('sentinel_edge.config.Config.from_env', return_value=SimpleNamespace(ocr_min_confidence=.85)), \
             patch('sentinel_edge.live_detection_viewer.load_models', return_value=(None, None, None)), \
             patch('sentinel_edge.live_detection_viewer.model_identity', return_value={'model': 'synthetic'}), \
             patch('sentinel_edge.tracked_live_viewer.observe_tracked', side_effect=observe), \
             patch.object(runner.ReviewObserver, 'export_assets'):
            runner.main(args)
        return args.output

    def test_model_acceptance_is_captured_by_the_frozen_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.run_fixture(directory)
            predictions = json.loads((root/'predictions.json').read_text())
            self.assertEqual(predictions['tracks']['track-s0-7']['accepted'],
                             [{'plate': 'MH01AB1234', 'pts_ms': 500}])

    def test_incomplete_acceptance_capture_cannot_be_published_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Acceptance observer'):
                self.run_fixture(directory, summary_acceptances=2)
            self.assertFalse((Path(directory)/'fixture'/'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
