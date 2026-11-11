"""Synthetic saved-audit viewer checks; no inference or real plate labels."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from render_crop_comparison import ROOT,sha
from render_text_corner_audit import associate,render,validate_audit,_quad
import test_crop_comparison as fixtures


def write(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


def native_hash(path):
    with Image.open(path) as image:
        return hashlib.sha256(image.convert('RGB').tobytes('raw','BGR')).hexdigest()


class TextCornerAuditTests(unittest.TestCase):
    def temporary(self):
        (ROOT/'data'/'benchmarks').mkdir(parents=True,exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='text-audit-unit-',dir=ROOT/'data'/'benchmarks')

    def fixture(self,root):
        baseline,unused,review=fixtures.CropComparisonTests().make_fixture(root)
        manifest=json.loads((review/'manifest.json').read_bytes())
        manifest['samples'][0].update(position_ms=500,box=[10,30,60,50])
        write(review/'manifest.json',manifest)
        labels=json.loads((review/'labels.json').read_bytes())
        labels['review_manifest_sha256']=sha(review/'manifest.json')
        write(review/'labels.json',labels)
        accuracy=json.loads((baseline/'accuracy.json').read_bytes())
        accuracy['labels_sha256']=sha(review/'labels.json')
        write(baseline/'accuracy.json',accuracy)
        capture=root/'audit';(capture/'assets').mkdir(parents=True)
        for name,size in [('region',(120,70)),('native',(50,20)),('rectified',(60,24))]:
            Image.new('RGB',size,'#dce8e1').save(capture/'assets'/(name+'.png'))
        native=capture/'assets/native.png';rectified=capture/'assets/rectified.png';region=capture/'assets/region.png'
        proposal=dict(id='P1',box=[10,30,60,50],quad=[[10,30],[60,30],[60,50],[10,50]],
            seed_quad=[[12,32],[58,32],[58,48],[12,48]],text_members=[0],character_components=4,
            geometry=dict(boundary='estimated_text_enclosure'),reasons=['Text seed'],
            verdict=dict(status='uncertain',classifier_score=.5,reasons=['Needs more evidence'],model_sha256='model'),
            native_path='assets/native.png',rectified_path='assets/rectified.png',native_sha256=sha(native),
            rectified_sha256=sha(rectified),native_source_id=native_hash(native),ocr_ran=False,top=None)
        audit=dict(source_sha256='source',scope='Synthetic capture; reference matching only after inference',
            labels_used_during_inference=False,elapsed_seconds=1.,schedule_sha256='schedule',summary=dict(proposals=1),
            verifier_metadata=dict(model='test-only'),frames=[dict(position_ms=500,regions=[dict(id='V1',
            box=[0,0,120,70],path='assets/region.png',sha256=sha(region),native_source_id=native_hash(region),
            polygons=[proposal['seed_quad'],[[80,10],[110,10],[110,20],[80,20]]],proposals=[proposal])])])
        write(capture/'audit.json',audit)
        return capture,baseline,review,audit

    def test_render_is_standalone_preserves_inputs_and_keeps_uncertain_evidence(self):
        with self.temporary() as directory:
            root=Path(directory);capture,baseline,review,audit=self.fixture(root)
            before={p:sha(p) for p in root.rglob('*') if p.is_file()}
            data=render(capture,baseline,review,root/'output')
            self.assertEqual(before,{p:sha(p) for p in before})
            self.assertEqual(len(data['references']),1)
            self.assertEqual(data['references'][0]['match'],[0,0,0])
            self.assertEqual(data['baseline_coverage']['correct'],0)
            self.assertNotIn('new_accuracy',data)
            self.assertFalse(data['frames'][0]['regions'][0]['proposals'][0]['ocr_ran'])
            self.assertEqual(len(data['frames'][0]['regions'][0]['polygons']),2)
            page=(root/'output/index.html').read_text(encoding='utf-8')
            for phrase in ('estimated text enclosure','not verified real-world probabilities','diagnostic single-frame',
                           'OCR skipped','not retained by bounded geometry search','data:image/png;base64,'):
                self.assertIn(phrase,page)
            self.assertNotIn('https://',page)
            self.assertTrue((root/'output/audit-view.json').is_file())
            with self.assertRaisesRegex(ValueError,'fresh'):
                render(capture,baseline,review,root/'output')

    def test_sources_paths_and_hashes_fail_before_output_creation(self):
        for change,message in [('source','source differs'),('path','Unsafe'),('png','PNG hash'),
                               ('native','native BGR'),('corners','outside native')]:
            with self.subTest(change=change),self.temporary() as directory:
                root=Path(directory);capture,baseline,review,audit=self.fixture(root)
                region=audit['frames'][0]['regions'][0];p=region['proposals'][0]
                if change=='source':audit['source_sha256']='other'
                elif change=='path':p['native_path']='../escape.png'
                elif change=='png':p['rectified_sha256']='changed'
                elif change=='native':p['native_source_id']='changed'
                elif change=='corners':p['quad'][0][0]=-1
                write(capture/'audit.json',audit)
                with self.assertRaisesRegex(ValueError,message):render(capture,baseline,review,root/'output')
                self.assertFalse((root/'output').exists())

    def test_ocr_is_blocked_for_uncertain_or_non_plate_states(self):
        for status in ('uncertain','likely_non_plate'):
            with self.subTest(status=status),self.temporary() as directory:
                root=Path(directory);capture,_,_,audit=self.fixture(root)
                p=audit['frames'][0]['regions'][0]['proposals'][0]
                p['verdict']['status']=status;p['ocr_ran']=True;p['top']='AB123'
                write(capture/'audit.json',audit)
                with self.assertRaisesRegex(ValueError,'must not run OCR'):validate_audit(capture)

    def test_label_free_capture_must_be_explicit_and_skipped_ocr_has_no_string(self):
        for value in (True,None):
            with self.subTest(value=value),self.temporary() as directory:
                root=Path(directory);capture,_,_,audit=self.fixture(root)
                audit['labels_used_during_inference']=value;write(capture/'audit.json',audit)
                with self.assertRaisesRegex(ValueError,'label-free'):validate_audit(capture)
        with self.temporary() as directory:
            root=Path(directory);capture,_,_,audit=self.fixture(root)
            audit['frames'][0]['regions'][0]['proposals'][0]['top']='AB123'
            write(capture/'audit.json',audit)
            with self.assertRaisesRegex(ValueError,'Skipped OCR'):validate_audit(capture)

    def test_association_uses_geometry_not_a_correct_diagnostic_string(self):
        with self.temporary() as directory:
            root=Path(directory);capture,_,_,audit=self.fixture(root)
            ref=dict(id='r',box=[10,30,60,50],position_ms=500)
            self.assertEqual(associate(ref,audit,{})['match'],[0,0,0])
            p=copy.deepcopy(audit['frames'][0]['regions'][0]['proposals'][0]);p['id']='P2'
            audit['frames'][0]['regions'][0]['proposals'].append(p)
            linked=associate(ref,audit,{})
            self.assertIsNotNone(linked['match'])
            self.assertIn('Overlapping crop alternatives',linked['correspondence'])
            self.assertIn('identity is not verified',linked['correspondence'])
            ref['position_ms']=1000
            self.assertEqual(associate(ref,audit,{})['frame_indexes'],[])

    def test_tied_candidates_from_different_vehicle_regions_remain_uncertain(self):
        with self.temporary() as directory:
            root=Path(directory);_,_,_,audit=self.fixture(root)
            extra=copy.deepcopy(audit['frames'][0]['regions'][0]);extra['id']='V2'
            audit['frames'][0]['regions'].append(extra)
            ref=dict(id='r',box=[10,30,60,50],position_ms=500)
            result=associate(ref,audit,{})
            self.assertIsNone(result['match'])
            self.assertIsNotNone(result['inspection'])
            self.assertIn('inspection only',result['correspondence'])

    def test_high_overlap_alternatives_use_geometric_overlap_not_classifier_or_ocr(self):
        with self.temporary() as directory:
            root=Path(directory);_,_,_,audit=self.fixture(root)
            proposals=audit['frames'][0]['regions'][0]['proposals']
            second=copy.deepcopy(proposals[0])
            second.update(id='P2',box=[11,30,61,50],ocr_ran=True,top='AB123')
            second['verdict'].update(status='likely_plate',classifier_score=.999)
            proposals.append(second)
            ref=dict(id='r',box=[10,30,60,50],position_ms=500)
            result=associate(ref,audit,{})
            self.assertEqual(result['match'],[0,0,0])
            self.assertIn('highest geometric overlap',result['correspondence'])

    def test_spatially_distinct_same_region_ties_remain_uncertain(self):
        with self.temporary() as directory:
            root=Path(directory);_,_,_,audit=self.fixture(root)
            proposals=audit['frames'][0]['regions'][0]['proposals']
            proposals[0]['box']=[0,30,50,50]
            second=copy.deepcopy(proposals[0]);second.update(id='P2',box=[20,30,70,50])
            proposals.append(second)
            ref=dict(id='r',box=[10,30,60,50],position_ms=500)
            self.assertIsNone(associate(ref,audit,{})['match'])

    def test_unsafe_ocr_text_is_json_escaped(self):
        with self.temporary() as directory:
            root=Path(directory);capture,baseline,review,audit=self.fixture(root)
            p=audit['frames'][0]['regions'][0]['proposals'][0]
            p['verdict']['status']='likely_plate';p['ocr_ran']=True;p['top']='</script><script>bad()</script>'
            write(capture/'audit.json',audit)
            render(capture,baseline,review,root/'output')
            page=(root/'output/index.html').read_text(encoding='utf-8')
            self.assertNotIn('</script><script>bad()',page)
            self.assertIn('\\u003c/script\\u003e',page)


class BoundaryRoundoffTests(unittest.TestCase):
    def test_subpixel_roundoff_is_display_normalised_but_real_escape_is_rejected(self):
        result=_quad([[-2.7e-16,0],[30,0],[30,10],[0,10]],(100,50))
        self.assertEqual(result[0],[0,0])
        with self.assertRaises(ValueError):
            _quad([[-.1,0],[30,0],[30,10],[0,10]],(100,50))


if __name__=='__main__':
    unittest.main()
