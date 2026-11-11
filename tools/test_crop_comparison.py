"""Focused saved-output scoring, binding and rendering tests; no inference."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from render_crop_comparison import ROOT, build_data, geometry_group, html, render, sha


def group(name, reading, *, excluded=False, position=500, box=None):
    return dict(id=name, methods=dict(reconstruction=dict(top=reading, candidates=[])),
                samples=[dict(position_ms=position, quality=.5, path='assets/crop.png')],
                observed=[dict(position_ms=position, box=box or [0, 0, 100, 40])],
                plate_validation=dict(excluded_from_fusion=excluded))


def case(reference='reference', expected='AB123', associated=None):
    return dict(id=reference, expected=expected, group=associated, position_ms=0)


def write(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


class CropComparisonTests(unittest.TestCase):
    def test_only_active_reconstructed_top_anywhere_counts_each_reference_once(self):
        wrong = group('wrong', 'WRONG')
        wrong['methods']['single'] = dict(top='AB123')
        wrong['methods']['multi'] = dict(top='AB123')
        wrong['methods']['reconstruction']['candidates'] = [dict(plate='AB123')]
        excluded = group('excluded', 'AB123', excluded=True)
        old = dict(groups=[wrong, excluded])
        # The exact matches are far from the reference timestamp, with no
        # accepted output or confidence: neither condition gates this metric.
        new = dict(groups=[group('first', 'AB123', position=90000), group('second', 'AB123', position=95000)])
        score = dict(cases=[case(associated='wrong')])
        manifest = dict(source_sha256='source', samples=[dict(id='reference', crop='ref.png')])
        result = build_data(old, new, score, dict(cases=[case()]), manifest)
        self.assertEqual(result['scores']['previous']['correct'], 0)
        self.assertEqual(result['scores']['current']['correct'], 1)
        self.assertEqual(result['scores']['current']['total'], 1)
        self.assertEqual(result['results'][0]['current']['group_id'], 'first')
        self.assertEqual(result['results'][0]['previous']['reading'], 'WRONG')
        self.assertEqual(result['results'][0]['number'], 1)

    def test_duplicate_reference_ids_cannot_inflate_denominator(self):
        score = dict(cases=[case(), case()])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            build_data(dict(groups=[]), dict(groups=[]), score, score, dict(samples=[]))

    def test_missing_association_stays_missing_without_geometry(self):
        score = dict(cases=[case()])
        manifest = dict(source_sha256='source', samples=[dict(id='reference', crop='ref.png')])
        result = build_data(dict(groups=[]), dict(groups=[group('different', 'WRONG')]), score, score, manifest)
        current = result['results'][0]['current']
        self.assertIsNone(current['crop'])
        self.assertIsNone(current['reading'])
        self.assertFalse(current['found'])

    def test_geometry_uses_equal_source_time_and_overlap_not_ids_or_text(self):
        anchor = group('old-id', 'AB123')
        mapped = group('new-id', 'WRONG', box=[2, 0, 102, 40])
        off_time = group('old-id', 'AB123', position=501)
        self.assertIs(geometry_group(anchor, [off_time, mapped]), mapped)
        self.assertIsNone(geometry_group(anchor, [off_time]))
        self.assertIsNone(geometry_group(anchor, [mapped, copy.deepcopy(mapped)]))
        self.assertIsNone(geometry_group(anchor, [mapped, group('almost-same', 'OTHER', box=[3,0,103,40])]))
        self.assertIsNone(geometry_group(anchor, [group('excluded', 'AB123', excluded=True)]))
        score = dict(cases=[case(associated='old-id')])
        manifest = dict(source_sha256='source', samples=[dict(id='reference', crop='ref.png')])
        result = build_data(dict(groups=[anchor]), dict(groups=[mapped]), score, dict(cases=[case()]), manifest)
        self.assertFalse(result['results'][0]['current']['found'])
        self.assertIn('not proof', result['results'][0]['current']['correspondence'])

    def make_fixture(self, root):
        review = root/'review'
        review.mkdir()
        Image.new('RGB', (80, 30), 'white').save(review/'ref.png')
        manifest = dict(source_sha256='source', samples=[dict(id='reference', crop='ref.png', crop_sha256=sha(review/'ref.png'))])
        write(review/'manifest.json', manifest)
        labels = dict(source_sha256='source', review_manifest_sha256=sha(review/'manifest.json'),
                      labels=[dict(id='reference', status='readable', plate='AB123')])
        write(review/'labels.json', labels)
        captures = []
        for name, reading in [('old', 'WRONG'), ('new', 'AB123')]:
            capture = root/name
            (capture/'assets').mkdir(parents=True)
            Image.new('RGB', (100, 40), '#0f766e').save(capture/'assets'/'crop.png')
            evidence = group(name, reading)
            with Image.open(capture/'assets'/'crop.png') as image:
                evidence['samples'][0]['source_id'] = hashlib.sha256(image.convert('RGB').tobytes('raw', 'BGR')).hexdigest()
            write(capture/'predictions.json', dict(source_sha256='source', groups=[evidence]))
            accuracy = dict(predictions_sha256=sha(capture/'predictions.json'), labels_sha256=sha(review/'labels.json'),
                            cases=[case(associated=name)])
            write(capture/'accuracy.json', accuracy)
            captures.append(capture)
        return *captures, review

    def temporary(self):
        (ROOT/'data'/'benchmarks').mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='comparison-unit-', dir=ROOT/'data'/'benchmarks')

    def test_render_preserves_every_input_and_outputs_standalone_html_and_sheet(self):
        with self.temporary() as directory:
            root = Path(directory)
            old, new, review = self.make_fixture(root)
            original = {path: sha(path) for path in root.rglob('*') if path.is_file()}
            output = root/'output'
            data = render(old, new, review, output)
            self.assertEqual({path: sha(path) for path in original}, original)
            self.assertEqual(data['scores']['current']['correct'], 1)
            page = (output/'index.html').read_text(encoding='utf-8')
            self.assertIn('data:image/png;base64,', page)
            self.assertIn('Choose reference', page)
            self.assertNotIn('https://', page)
            self.assertTrue((output/'comparison-01.png').is_file())
            for row in data['results']:
                self.assertEqual(row['reference_sha256'], sha(output/row['reference_crop']))
                for side in ('previous', 'current'):
                    self.assertEqual(row[side]['crop_sha256'], sha(output/row[side]['crop']))
                    self.assertTrue(row[side]['native_source_id_verified'])
            with self.assertRaisesRegex(ValueError, 'fresh'):
                render(old, new, review, output)

    def test_tampered_bindings_fail_before_output_creation(self):
        for tamper, expected in [('prediction', 'hash binding'), ('source', 'Source hash'),
                                 ('reference', 'Reference image'), ('labels_manifest', 'review manifest'),
                                 ('answer', 'actual readable'), ('automatic', 'native source hash')]:
            with self.subTest(tamper=tamper), self.temporary() as directory:
                root = Path(directory)
                old, new, review = self.make_fixture(root)
                prediction = json.loads((new/'predictions.json').read_bytes())
                accuracy = json.loads((new/'accuracy.json').read_bytes())
                if tamper == 'prediction':
                    (new/'predictions.json').write_text('{}')
                elif tamper == 'source':
                    prediction['source_sha256'] = 'other-source'
                    write(new/'predictions.json', prediction)
                    accuracy['predictions_sha256'] = sha(new/'predictions.json')
                    write(new/'accuracy.json', accuracy)
                elif tamper == 'reference':
                    Image.new('RGB', (80, 30), 'black').save(review/'ref.png')
                elif tamper == 'labels_manifest':
                    manifest = json.loads((review/'manifest.json').read_bytes())
                    manifest['selection'] = 'changed'
                    write(review/'manifest.json', manifest)
                elif tamper == 'automatic':
                    Image.new('RGB', (100, 40), 'red').save(new/'assets'/'crop.png')
                else:
                    accuracy['cases'][0]['expected'] = 'WRONG-ANSWER'
                    write(new/'accuracy.json', accuracy)
                with self.assertRaisesRegex(ValueError, expected):
                    render(old, new, review, root/'output')
                self.assertFalse((root/'output').exists())

    def test_unsafe_automatic_path_is_rejected_without_output(self):
        with self.temporary() as directory:
            root = Path(directory)
            old, new, review = self.make_fixture(root)
            prediction = json.loads((new/'predictions.json').read_bytes())
            prediction['groups'][0]['samples'][0]['path'] = '../review/ref.png'
            write(new/'predictions.json', prediction)
            accuracy = json.loads((new/'accuracy.json').read_bytes())
            accuracy['predictions_sha256'] = sha(new/'predictions.json')
            write(new/'accuracy.json', accuracy)
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                render(old, new, review, root/'output')
            self.assertFalse((root/'output').exists())

    def test_html_escapes_saved_text_and_has_no_external_assets(self):
        data = dict(results=[], scores={}, rule='</script><script>alert(1)</script>')
        page = html(data, Path('.'))
        self.assertNotIn('</script><script>alert(1)</script>', page)
        self.assertIn('\\u003c/script\\u003e', page)

    def test_selection_priority_precedes_quality_without_changing_readings(self):
        preferred = group('new', 'AB123')
        preferred['samples'].append(dict(position_ms=700, quality=.1, selection_priority=1, path='assets/preferred.png'))
        preferred['samples'][0]['quality'] = .99
        score = dict(cases=[case()])
        manifest = dict(source_sha256='source', samples=[dict(id='reference', crop='ref.png')])
        result = build_data(dict(groups=[]), dict(groups=[preferred]), score, score, manifest)
        selected = result['results'][0]['current']
        self.assertEqual(selected['crop'], 'assets/preferred.png')
        self.assertEqual(selected['reading'], 'AB123')
        self.assertTrue(selected['found'])


if __name__ == '__main__':
    unittest.main()
