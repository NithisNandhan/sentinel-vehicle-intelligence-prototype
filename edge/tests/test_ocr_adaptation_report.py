import tempfile
import unittest
from pathlib import Path

from sentinel_edge.ocr_adaptation_report import render_summary

DATASET = {'name': '<script>dataset</script>', 'license': 'CC-BY-4.0',
           'source_url': 'https://example.org/data', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
           'permitted_use': 'noncommercial_benchmark', 'attribution': 'Research fixture'}


class SummaryTests(unittest.TestCase):
    def fixture(self):
        report = {'manifest_sha256': 'a', 'backend': 'awiros', 'stage': 'automatic_plate_crop_ocr',
                  'device': 'cpu', 'reconstruction_requested': False, 'restoration_requested': False,
                  'quality_weighted': True, 'deskew_requested': True, 'family': 'all',
                  'further_fine_tuned_here': False, 'correct': 1, 'wrong': 0, 'abstain': 0,
                  'rows': [{'condition': '<unsafe>', 'passage_index': 0, 'outcome': 'correct', 'truth_rank': 1}]}
        tuned = dict(report, further_fine_tuned_here=True, adapted_checkpoint_sha256='b')
        stills = {'checkpoint_sha256': 'b', 'models': {'Original': {'exact': 1, 'samples': 2,
                    'character_errors': 1, 'characters': 20, 'cer': .05, 'candidate_recall_counts': {'10': 2}}}}
        training = {'trainable_parameters': 7744, 'selected_epoch': 55,
                    'split_counts': {'train': 28, 'validation': 10, 'test': 10}}
        return report, tuned, stills, training

    def test_static_escaped_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data/benchmarks/reports/summary.html'
            output.parent.mkdir(parents=True)
            render_summary(output, DATASET, *self.fixture())
            text = output.read_text()
            self.assertNotIn('<script>', text)
            self.assertIn('&lt;unsafe&gt;', text)
            self.assertIn('does not establish 90%', text)
            self.assertIn("default-src 'none'", text)

    def test_configuration_or_checkpoint_mismatch_rejected(self):
        for field in ('manifest_sha256', 'backend', 'device', 'stage', 'quality_weighted',
                      'reconstruction_requested', 'restoration_requested', 'deskew_requested', 'family'):
            parent, tuned, stills, training = self.fixture()
            tuned[field] = 'mismatch'
            with self.assertRaises(ValueError):
                render_summary(Path('data/benchmarks/reports/x.html'), DATASET, parent, tuned, stills, training)
        parent, tuned, stills, training = self.fixture()
        stills['checkpoint_sha256'] = 'different'
        with self.assertRaises(ValueError):
            render_summary(Path('data/benchmarks/reports/x.html'), DATASET, parent, tuned, stills, training)
