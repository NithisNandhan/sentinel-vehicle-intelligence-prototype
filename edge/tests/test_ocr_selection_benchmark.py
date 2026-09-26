import json
import io
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sentinel_edge.ocr_selection_benchmark import run_selection_benchmark


class SelectionBenchmarkTests(unittest.TestCase):
    def test_processed_view_cannot_supply_missing_original_observations(self):
        import numpy as np
        class FakeOCR:
            tokens = ('', *'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            device = 'cpu'
            def probabilities(self, crop):
                rows = []
                for char in 'GJ01AB1234':
                    row = [0.] * len(self.tokens)
                    row[self.tokens.index(char)] = 1.
                    rows.extend([row, [1.] + [0.] * (len(self.tokens) - 1)])
                return rows
        class TwoFrameDetector:
            calls = 0
            def box(self, image):
                self.calls += 1
                return (10, 10, 80, 40) if self.calls % 3 else None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.fixture(root)
            output = root / 'data/benchmarks/reports'
            with patch('sentinel_edge.awiros_experiment.AwirosExperiment', return_value=FakeOCR()), \
                    patch('sentinel_edge.ocr_selection_benchmark.PlateDetectorExperiment', return_value=TwoFrameDetector()), \
                    patch('sentinel_edge.plate_quality.deskew_view', return_value=(np.zeros((30, 70, 3), np.uint8), 5.)), \
                    patch('sentinel_edge.ocr_selection_benchmark.deskew_view', return_value=(np.zeros((30, 70, 3), np.uint8), 5.)), \
                    redirect_stdout(io.StringIO()):
                result = run_selection_benchmark(manifest, root, family='motion', automatic_crops=True,
                                                 deskew=True, report_path=output / 'two.json', viewer_path=output / 'two.html')
            self.assertEqual(result['correct'], 0)
            self.assertTrue(all(row['observations'] == 2 for row in result['rows']))
            self.assertTrue(all(row['processed_diagnostics'][0]['greedy_exact'] for row in result['rows']))
            self.assertNotIn('GJ01AB1234', (output / 'two.json').read_text())

    def test_automatic_no_detections_reports_abstention_not_oracle_fallback(self):
        class FakeOCR:
            tokens = ('', 'A')
            device = 'cpu'
            def probabilities(self, crop):
                return [[1., 0.]]
        class MissingDetector:
            def box(self, image):
                return None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.fixture(root)
            output = root / 'data/benchmarks/reports'
            with patch('sentinel_edge.awiros_experiment.AwirosExperiment', return_value=FakeOCR()), \
                    patch('sentinel_edge.ocr_selection_benchmark.PlateDetectorExperiment', return_value=MissingDetector()), \
                    redirect_stdout(io.StringIO()):
                result = run_selection_benchmark(manifest, root, family='motion', automatic_crops=True,
                                                 report_path=output / 'auto.json', viewer_path=output / 'auto.html')
            self.assertEqual(result['correct'], 0)
            self.assertEqual(result['abstain'], len(result['rows']))
            self.assertTrue(result['automatic_detection_evaluated'])
            self.assertTrue(all(row['selected_crops'] == 0 for row in result['rows']))
            self.assertTrue(all(row['failure_reason'] == 'automatic_localisation_insufficient' for row in result['rows']))

    def fixture(self, root):
        import cv2
        import numpy as np
        dataset = {'name': 'Synthetic unit fixture', 'license': 'CC-BY-4.0',
                   'source_url': 'https://example.org/fixture',
                   'license_url': 'https://creativecommons.org/licenses/by/4.0/',
                   'permitted_use': 'noncommercial_benchmark'}
        frames = []
        for index in range(3):
            name = f'frame-{index}.png'
            cv2.imwrite(str(root / name), np.full((64, 128, 3), index, np.uint8))
            frames.append({'path': name, 'pts_ms': 400 * index,
                           'plate_box': [10, 10, 80, 40], 'vehicle_box': [0, 0, 100, 60]})
        document = {'dataset': dataset, 'passages': [{'id': 'one', 'vehicle_id': 'vehicle-one',
                    'split': 'development', 'country': 'IN', 'kind': 'sequence',
                    'timing_mode': 'assigned_replay', 'ground_truth': 'GJ01AB1234', 'frames': frames}]}
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps(document))
        return manifest, document

    def test_labels_are_for_scoring_not_model_inputs_and_reports_are_numeric(self):
        import numpy as np
        class FakeOCR:
            tokens = ('', *'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            device = 'cpu'
            inputs = []
            def probabilities(self, crop):
                self.inputs.append(crop.copy())
                rows = []
                for char in 'GJ01AB1234':
                    row = [0.] * len(self.tokens)
                    row[self.tokens.index(char)] = 1.
                    rows.extend([row, [1.] + [0.] * (len(self.tokens) - 1)])
                return rows
        fake = FakeOCR()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.fixture(root)
            output = root / 'data/benchmarks/reports'
            with patch('sentinel_edge.awiros_experiment.AwirosExperiment', return_value=fake), redirect_stdout(io.StringIO()):
                result = run_selection_benchmark(manifest, root, family='resolution',
                                                 report_path=output / 'result.json', viewer_path=output / 'result.html')
            self.assertEqual(result['correct'], 7)
            self.assertEqual(result['wrong'], 0)
            self.assertFalse(result['further_fine_tuned_here'])
            self.assertNotIn('GJ01AB1234', (output / 'result.json').read_text())
            self.assertIn('GJ01AB1234', (output / 'result.html').read_text())
            self.assertTrue(np.array_equal(fake.inputs[1], np.zeros((30, 70, 3), np.uint8)))

    def test_live_provenance_is_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, document = self.fixture(root)
            document['dataset']['source_url'] = 'https://cctv.corp8.cloud/cam01'
            manifest.write_text(json.dumps(document))
            output = root / 'data/benchmarks/reports'
            with patch('sentinel_edge.awiros_experiment.AwirosExperiment') as loader:
                with self.assertRaises(ValueError):
                    run_selection_benchmark(manifest, root, report_path=output / 'result.json',
                                             viewer_path=output / 'result.html')
                loader.assert_not_called()
