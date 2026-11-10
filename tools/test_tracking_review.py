"""Synthetic fixtures only; never write real tracking review labels."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
import shutil
import subprocess
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from benchmark_vehicle_tracking import ownership
from serve_tracking_review import ReviewStore, make_handler


class TrackingReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.video=self.root/'source.mp4';self.video.write_bytes(b'0123456789')
        report=dict(source_sha256=hashlib.sha256(self.video.read_bytes()).hexdigest(),
                    strategies={'screened':{'records':[{'passage_id':'V001','preview_paths':[]}]}})
        (self.root/'report.json').write_text(json.dumps(report))
        (self.root/'timeline.json').write_text(json.dumps({'frames':[]}))
        self.store=ReviewStore(self.root,self.video)

    def tearDown(self):self.tmp.cleanup()

    def test_labels_survive_restart_and_stale_revision_rejected(self):
        value=copy.deepcopy(self.store.review);value['labeller']='Fixture reviewer'
        value['labels']['V001']['status']='non_vehicle'
        result=self.store.save(value)
        self.assertEqual(result['revision'],1)
        self.assertEqual(ReviewStore(self.root,self.video).review['labels']['V001']['status'],'non_vehicle')
        with self.assertRaises(RuntimeError):self.store.save(value)

    def test_review_is_bound_to_report_and_known_inventory(self):
        for change in ('report','inventory','status'):
            value=copy.deepcopy(self.store.review)
            if change=='report':value['report_sha256']='a'*64
            elif change=='inventory':value['labels']['V002']={'status':'pending','note':''}
            else:value['labels']['V001']['status']='verified_by_model'
            with self.assertRaises(ValueError):self.store.save(value)

    def test_http_range_origin_and_path_boundaries(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.store,0));port=server.server_address[1]
        server.RequestHandlerClass=make_handler(self.store,port)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            base=f'http://127.0.0.1:{port}'
            request=urllib.request.Request(base+'/video',headers={'Range':'bytes=2-5'})
            with urllib.request.urlopen(request) as r:self.assertEqual((r.status,r.read()),(206,b'2345'))
            for path in ('/.env','/assets/../report.json','/capture-script.py'):
                with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(base+path)
            request=urllib.request.Request(base+'/api/review',data=json.dumps(self.store.review).encode(),
                headers={'Content-Type':'application/json','Origin':'https://unrelated.example'})
            with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(request)
            self.assertEqual(error.exception.code,403)
        finally:server.shutdown();server.server_close();thread.join(timeout=3)

    def test_overlapping_vehicle_boxes_are_ambiguous_not_arbitrarily_assigned(self):
        plate=[30,30,50,40]
        tracks=[{'passage_id':'a','box':[0,0,100,100]},{'passage_id':'b','box':[20,20,80,80]}]
        self.assertEqual(ownership(plate,tracks)['status'],'ambiguous')
        self.assertEqual(ownership(plate,tracks[:1])['status'],'unique')
        self.assertEqual(ownership(plate,[])['status'],'unassigned')

    @unittest.skipUnless(shutil.which('node'),'Node is needed for the browser timing regression')
    def test_browser_fractional_frame_times_choose_the_correct_rounded_pts(self):
        page=Path(__file__).with_name('tracking_review.html').read_text(encoding='utf-8')
        function=page[page.index('function frameIndex(ms)'):page.index('function visible(f)')]
        javascript='const timeline={frames:Array.from({length:928},(_,i)=>({t:Math.round(i*1000/30)}))};'+function
        javascript+='const wrong=[];for(let i=0;i<928;i++){if(frameIndex(i*1000/30)!==i)wrong.push(i);}console.log(JSON.stringify(wrong));'
        result=subprocess.run([shutil.which('node'),'-e',javascript],check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(result.stdout),[])


if __name__=='__main__':unittest.main()
