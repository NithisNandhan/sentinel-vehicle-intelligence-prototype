import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_owned_multiframe import build_vehicle_results,score,sha


class EvidenceIntegrationTests(unittest.TestCase):
    def test_parent_reader_excludes_suspected_window_but_retains_real_plate(self):
        tokens=('',*'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        def rows_for(text):
            rows=[]
            for char in text:
                row=[.001]*len(tokens);row[0]=.001;row[tokens.index(char)]=1-.001*(len(tokens)-1)
                blank=[.001]*len(tokens);blank[0]=1-.001*(len(tokens)-1)
                rows.extend([row,blank])
            return rows
        def sample(text,t,source):
            return dict(rows=rows_for(text),position_ms=t,source_id=source,quality=.9,vehicle_id='VT1')
        predictions=[dict(id='PT1',parent_vehicle_id='VT1',plate_validation=dict(excluded_from_fusion=True)),
                     dict(id='PT2',parent_vehicle_id='VT1')]
        posts=[dict(id='PT1',observations=[sample('MH02XX9999',0,'window'),sample('MH02XX9999',250,'window2')]),
               dict(id='PT2',observations=[sample('MH01AB1234',0,'plate'),sample('MH01AB1234',250,'plate2')])]
        result=build_vehicle_results(predictions,posts,tokens)
        self.assertEqual(result[0]['children'],['PT2'])
        self.assertEqual(result[0]['methods']['top'],'MH01AB1234')
        self.assertEqual({s['source_id'] for s in result[0]['samples']},{'plate','plate2'})
        self.assertEqual(build_vehicle_results(predictions[:1],posts[:1],tokens),[])
        # The same plausible OCR evidence may remain visible provisionally,
        # but uncertainty must not become an accepted parent/watchlist result.
        posts[1]['observations'].append(sample('MH01AB1234',500,'plate3'))
        self.assertIsNotNone(build_vehicle_results(predictions,posts,tokens)[0]['methods']['accepted'])
        predictions[1]['provisional_region']=True
        provisional=build_vehicle_results(predictions,posts,tokens)[0]
        self.assertEqual(provisional['methods']['top'],'MH01AB1234')
        self.assertIsNone(provisional['methods']['accepted'])

    def test_rejected_true_plate_counts_as_miss_even_if_raw_hypothesis_is_correct(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);review=root/'review';review.mkdir();(review/'ocr-model-dev').mkdir()
            (review/'ocr-model-dev/rectify.py').write_text('CORNERS = {}')
            (review/'manifest.json').write_text(json.dumps(dict(source_sha256='source',
                samples=[dict(id='01',position_ms=250,box=[0,0,100,30])])))
            (review/'labels.json').write_text(json.dumps(dict(source_sha256='source',
                review_manifest_sha256=sha(review/'manifest.json'),labels=[dict(id='01',status='readable',plate='MH01AB1234')])) )
            doc=dict(source_sha256='source',groups=[dict(id='PT1',observations=3,
                plate_validation=dict(excluded_from_fusion=True),observed=[dict(position_ms=250,box=[0,0,100,30])],
                methods={m:dict(top='MH01AB1234') for m in ('single','multi','reconstruction')})])
            (root/'predictions.json').write_text(json.dumps(doc))
            with patch('run_owned_multiframe.render'),contextlib.redirect_stdout(io.StringIO()):
                score(SimpleNamespace(output=root,review=review))
            accuracy=json.loads((root/'accuracy.json').read_text())
            self.assertEqual(accuracy['summary']['multi']['targets'],1)
            self.assertEqual(accuracy['summary']['multi']['exact'],0)
            self.assertIsNone(accuracy['cases'][0]['group'])


if __name__=='__main__':unittest.main()
