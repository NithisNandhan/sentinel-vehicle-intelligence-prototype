import tempfile
import unittest
from pathlib import Path

from sentinel_edge.benchmark_viewer import render_viewer, ranked_candidates, validate_viewer_scope
from sentinel_edge.temporal import PlateCandidate

DATASET = {'name': 'Permitted fixture', 'license': 'CC-BY-4.0',
           'source_url': 'https://example.org/data', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
           'permitted_use': 'noncommercial_benchmark'}


class ViewerTests(unittest.TestCase):
    def test_experiment_metadata_does_not_claim_reconstruction_or_verified_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data/benchmarks/reports/view.html'
            dataset = dict(DATASET,modifications='Native automatic crops; no reconstruction <script>')
            case = {'source_mode':'benchmark','expected':'MH02EK0837','source_count':3,
                    'label_description':'Visually reviewed development label <script>',
                    'source_count_label':'Distinct original observations'}
            render_viewer(output,dataset,[case])
            text = output.read_text()
            self.assertIn('Distinct original observations: 3',text)
            self.assertIn('Visually reviewed development label',text)
            self.assertIn('Native automatic crops; no reconstruction',text)
            self.assertNotIn('Manually verified evaluation label',text)
            self.assertNotIn('Reconstruction sources',text)
            self.assertNotIn('<script>',text)

    def test_still_diagnostic_never_claims_temporal_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data/benchmarks/reports/view.html'
            case = {'source_mode': 'benchmark', 'diagnostic_only': True, 'outcome': 'raw-exact'}
            render_viewer(output, DATASET, [case])
            self.assertIn('RAW MATCH', output.read_text())
            case['accepted'] = 'GJ01AB1234'
            with self.assertRaises(ValueError):
                render_viewer(output, DATASET, [case])

    def test_processed_readings_are_correlated_escaped_and_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data/benchmarks/reports/view.html'
            case = {'source_mode': 'benchmark', 'localisation_note': '<script>location</script>',
                    'processed_readings': [{'origin': '<script>processed</script>',
                                            'reading': '<script>reading</script>', 'ocr_ms': 1.}]}
            render_viewer(output, DATASET, [case])
            text = output.read_text()
            self.assertNotIn('<script>', text)
            self.assertIn('not additional votes', text)
            case['processed_readings'] *= 4
            with self.assertRaises(ValueError):
                render_viewer(output, DATASET, [case])

    def candidate(self, text, score=.5):
        return PlateCandidate(text, score, .2, -5., 3, 'state')

    def test_rank_is_bounded_unique_and_original_takes_precedence(self):
        values = [self.candidate(f'GJ01AB{i:04d}') for i in range(250)]
        ranked = ranked_candidates(values, [self.candidate(values[0].plate, .99)])
        self.assertEqual(len(ranked), 200)
        self.assertEqual(len({c.plate for c, _ in ranked}), 200)
        self.assertEqual(ranked[0][1], 'original-frame fusion')

    def test_live_and_bad_output_are_rejected(self):
        for source, output in (('live', Path('data/benchmarks/reports/test.html')),
                               ('benchmark', Path('output/test.html'))):
            with self.assertRaises(ValueError):
                validate_viewer_scope(output, DATASET, source_mode=source)
        dataset = dict(DATASET, source_url='https://cctv.corp8.cloud/cam01')
        with self.assertRaises(ValueError):
            validate_viewer_scope(Path('data/benchmarks/reports/test.html'), dataset)

    def test_uncertain_view_escapes_text_and_has_no_external_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data/benchmarks/reports/view.html'
            cases = [{'source_mode': 'benchmark', 'title': '<script>unsafe</script>',
                      'candidates': [self.candidate('GJ01AB1234')], 'accepted': None}]
            render_viewer(output, DATASET, cases)
            text = output.read_text()
            self.assertIn('UNCERTAIN', text)
            self.assertIn('GJ01AB1234', text)
            self.assertNotIn('<script>', text)
            self.assertNotIn('src="http', text)
            self.assertIn("default-src 'none'", text)

    def test_empty_evidence_outputs_no_read_not_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data/benchmarks/reports/view.html'
            render_viewer(output, DATASET, [{'source_mode': 'benchmark', 'partial': 'GJ01??12?4'}])
            text = output.read_text()
            self.assertIn('NO-READ', text)
            self.assertIn('GJ01??12?4', text)
            self.assertIn('No retained complete candidate', text)

    def test_each_case_must_be_benchmark(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                render_viewer(Path(temp) / 'data/benchmarks/reports/view.html', DATASET,
                              [{'source_mode': 'live'}])

    def test_frame_diagnostics_are_escaped_and_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data/benchmarks/reports/view.html'
            case = {'source_mode': 'benchmark', 'expected': '<script>label</script>',
                    'score_note': '<script>score</script>',
                    'frame_diagnostics': [{'index': 0, 'reading': '<script>reading</script>'}]}
            render_viewer(output, DATASET, [case])
            text = output.read_text()
            self.assertNotIn('<script>', text)
            self.assertIn('Raw OCR', text)
            case['frame_diagnostics'] *= 41
            with self.assertRaises(ValueError):
                render_viewer(output, DATASET, [case])
