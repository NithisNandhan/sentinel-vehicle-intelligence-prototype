"""Controlled fixtures only: persistence, blindness and local media protocol."""
import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from PIL import Image

spec=importlib.util.spec_from_file_location('review_server',Path(__file__).with_name('serve_video_review.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ReviewServerTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        (self.root/'assets').mkdir();self.video=self.root/'source.mp4';self.video.write_bytes(b'0123456789')
        self.manifest=dict(schema_version=1,source=dict(name='controlled fixture',sha256=module.digest(self.video),
            width=640,height=480,duration_ms=1000),run=dict(reached_eof=True,decoded_frames=30,analysed_frames=2),
            models=dict(ocr='synthetic fixture only'),tracks=[dict(id='track-s0-1',first_ms=0,last_ms=500,plate_crop_count=0,samples=[],vehicle_preview=None)])
        module.write_json(self.root/'manifest.json',self.manifest);self.sha=module.digest(self.root/'manifest.json')
        module.write_json(self.root/'predictions.json',dict(schema_version=1,manifest_sha256=self.sha,
            source_sha256=self.manifest['source']['sha256'],tracks={'track-s0-1':dict(first_ms=0,last_ms=500,
                top_guess=None,candidates=[],candidate_ms=None,accepted=[],observations=0)}))
        self.store=module.ReviewStore(self.root,self.video)

    def tearDown(self):self.directory.cleanup()

    def test_draft_survives_restart_and_stale_revision_is_rejected(self):
        body=copy.deepcopy(self.store.labels);body['labeller']='Fixture reviewer'
        body['labels'][0].update(status='readable',plate='mh-01 ab 1234')
        self.assertEqual(self.store.save(body)['revision'],1)
        restored=module.ReviewStore(self.root,self.video)
        self.assertEqual(restored.labels['labels'][0]['plate'],'MH01AB1234')
        with self.assertRaises(module.Conflict):restored.save(body)

    def test_finalisation_is_durable_and_read_only(self):
        body=copy.deepcopy(self.store.labels);body['labeller']='Fixture reviewer'
        body['labels'][0].update(status='readable',plate='MH01AB1234')
        result=self.store.save(body,final=True)
        self.assertEqual(result['report']['metrics']['best_guess_exact_coverage']['value'],0.)
        restored=module.ReviewStore(self.root,self.video)
        self.assertTrue(restored.finalized)
        with self.assertRaises(module.Conflict):restored.save(restored.labels)

    def test_missed_vehicle_native_png_and_metadata_are_owned_by_server(self):
        stream=io.BytesIO();Image.new('RGB',(20,10),'white').save(stream,format='PNG')
        body=dict(revision=0,position_ms=250,box=[10,20,30,30],image_png=base64.b64encode(stream.getvalue()).decode())
        result=self.store.add_extra(body)
        self.assertEqual(result['revision'],1)
        self.assertTrue(self.store.asset_path(result['extra']['image_path']).exists())
        altered=copy.deepcopy(self.store.labels);altered['extras'][0]['position_ms']=999
        with self.assertRaises(ValueError):self.store.save(altered)
        body['revision']=1;body['box']=[10,20,31,30]
        with self.assertRaises(ValueError):self.store.add_extra(body)

    def test_source_or_frozen_predictions_cannot_silently_change(self):
        document=json.loads((self.root/'predictions.json').read_text());document['manifest_sha256']='a'*64
        module.write_json(self.root/'predictions.json',document)
        with self.assertRaises(ValueError):module.ReviewStore(self.root,self.video)

    def test_http_hides_predictions_supports_ranges_and_rejects_cross_origin(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),module.make_handler(self.store,0))
        port=server.server_address[1];server.RequestHandlerClass=module.make_handler(self.store,port)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{port}'
        try:
            with urllib.request.urlopen(base+'/api/review') as r:public=json.load(r)
            self.assertNotIn('predictions',public);self.assertNotIn('top_guess',json.dumps(public))
            for path in ('/api/report','/predictions.json','/assets/../predictions.json'):
                with self.assertRaises(urllib.error.HTTPError) as failure:urllib.request.urlopen(base+path)
                self.assertIn(failure.exception.code,(400,404))
            request=urllib.request.Request(base+'/video',headers={'Range':'bytes=2-5'})
            with urllib.request.urlopen(request) as r:
                self.assertEqual(r.status,206);self.assertEqual(r.read(),b'2345')
            request=urllib.request.Request(base+'/video',headers={'Range':'bytes=100-'})
            with self.assertRaises(urllib.error.HTTPError) as failure:urllib.request.urlopen(request)
            self.assertEqual(failure.exception.code,416)
            request=urllib.request.Request(base+'/api/labels',data=json.dumps(self.store.labels).encode(),
                headers={'Content-Type':'application/json','Origin':'https://unrelated.example'})
            with self.assertRaises(urllib.error.HTTPError) as failure:urllib.request.urlopen(request)
            self.assertEqual(failure.exception.code,403)
        finally:server.shutdown();server.server_close();thread.join(timeout=3)


if __name__=='__main__':unittest.main()
