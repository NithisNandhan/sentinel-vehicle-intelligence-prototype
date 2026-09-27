import base64
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from sentinel_edge.live_detection_viewer import LiveState
from sentinel_edge.recognition_contract import operator_snapshot
from sentinel_edge.recognition_debug import characters
from sentinel_edge.shared_delivery import SharedDelivery
from sentinel_edge.tracked_ocr import TrackedOCR

TOKENS = ('', *'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
TEXT = 'AP01AB1234'


def posterior(_):
    rows = []
    for char in TEXT:
        row = [0.]*len(TOKENS)
        row[TOKENS.index(char)] = 1.
        rows.append(row)
    return rows


class SharedRecognitionTests(unittest.TestCase):
    def test_fallback_is_only_the_existing_approved_local_pair(self):
        from sentinel_edge.live_detection_viewer import configured_relay_cameras
        config = SimpleNamespace(camera_ids=('cam01','cam02'),media_relay_base_url='rtsp://mediamtx:8554')
        self.assertEqual(set(configured_relay_cameras(config)),{'cam01','cam02'})
        config.camera_ids = ('cam01','cam17')
        self.assertEqual(configured_relay_cameras(config),{})
        config.camera_ids = ('cam01','cam02')
        config.media_relay_base_url = 'rtsp://external.invalid:8554'
        self.assertEqual(configured_relay_cameras(config),{})

    def test_http_evidence_needs_both_token_and_operator_key(self):
        import http.client
        import re
        import socket
        import threading
        import time
        from sentinel_edge.live_detection_viewer import serve
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
        thread = threading.Thread(target=serve,args=(None,None,None),
                                  kwargs={'port':port,'lifetime':2,'operator_key':'private-test-key'},daemon=True)
        thread.start()
        for _ in range(50):
            try:
                conn = http.client.HTTPConnection('127.0.0.1',port,timeout=2)
                conn.request('GET','/'); response = conn.getresponse(); page = response.read().decode(); break
            except OSError:time.sleep(.02)
        self.assertEqual(response.status,200)
        self.assertNotIn('private-test-key',page)
        token = re.search(r'const viewerToken="([^"]+)"',page)[1]
        for headers in ({},{'X-Viewer-Token':token},{'X-Operator-Key':'private-test-key'}):
            conn.request('GET','/engineering',headers=headers)
            response = conn.getresponse(); response.read()
            self.assertEqual(response.status,403)
        conn.request('GET','/engineering',headers={'X-Viewer-Token':token,'X-Operator-Key':'private-test-key'})
        response = conn.getresponse()
        self.assertEqual(response.status,200)
        self.assertEqual(json.loads(response.read())['status'],'paused')
        conn.request('GET','/internal/status',headers={'X-Viewer-Token':token,'X-Operator-Key':'private-test-key'})
        response = conn.getresponse(); response.read()
        self.assertEqual(response.status,403)
        conn.close(); thread.join(timeout=5)
        self.assertFalse(thread.is_alive())

    def engine(self, callback=None):
        return TrackedOCR(SimpleNamespace(tokens=TOKENS,probabilities=posterior),
                          SimpleNamespace(box=lambda image:(5,5,150,35)),debug=True,on_accepted=callback)

    def image(self, value):
        image = np.full((60,180,3),100,np.uint8)
        image[10,10] = value
        return image

    def test_engineering_evidence_is_native_distinct_and_operator_view_is_masked(self):
        import cv2
        engine = self.engine()
        track = SimpleNamespace(track_id='1',box=(0,0,180,60),vehicle_class=2)
        for i in range(3):
            engine.process(self.image(i),[track],i*500)
        debug = engine.debug_snapshot()
        row = debug['tracks'][0]
        self.assertEqual(row['candidates'][0]['plate'],TEXT)
        self.assertEqual(row['recognition_state'],'model_accepted')
        self.assertIn('not been independently verified',row['gate_reasons'][0])
        self.assertEqual(len(row['frames']),3)
        crop = cv2.imdecode(np.frombuffer(base64.b64decode(row['frames'][0]['image']),np.uint8),cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(crop,self.image(0)[5:35,5:150])
        snapshot = {'status':'live','camera_id':'cam01','debug':debug,'frame':'private image','plate':TEXT}
        safe = operator_snapshot(snapshot)
        self.assertEqual(safe['tracks'][0]['masked_plate'],'AP******34')
        self.assertNotIn(TEXT,json.dumps(safe))
        self.assertNotIn('image',json.dumps(safe))
        self.assertEqual(safe['recognition_state'],'model_accepted')
        engine.reset()
        self.assertEqual(engine.debug_snapshot()['tracks'],[])
        self.assertEqual(engine.crop_bank.bytes_used,0)

    def test_duplicate_crop_does_not_create_debug_frames(self):
        engine = self.engine()
        track = SimpleNamespace(track_id='1',box=(0,0,180,60),vehicle_class=2)
        for pts in (0,500,1000):engine.process(self.image(1),[track],pts)
        row = engine.debug_snapshot()['tracks'][0]
        self.assertEqual(len(row['frames']),1)
        self.assertEqual(row['frames'][0]['text'],TEXT)
        self.assertIn('three distinct',row['gate_reasons'][0])

    def test_plate_evidence_expires_even_when_vehicle_stays_and_detector_misses(self):
        engine = self.engine()
        track = SimpleNamespace(track_id='1',box=(0,0,180,60),vehicle_class=2)
        engine.process(self.image(1),[track],0)
        engine.detector.box = lambda image:None
        engine.process(self.image(2),[track],1000)
        engine.process(self.image(3),[track],3000)
        row = engine.debug_snapshot()['tracks'][0]
        self.assertEqual(row['observations'],0)
        self.assertEqual(row['frames'],[])
        self.assertIn('Plate detector found no region',row['gate_reasons'][0])

    def test_plaintext_debug_is_opt_in_and_cleared_on_camera_change(self):
        state = LiveState()
        state.select('cam01')
        state.publish({'camera_id':'cam01','status':'live','debug':{'plate':TEXT}})
        self.assertNotIn(TEXT,json.dumps(state.snapshot()))
        self.assertIn(TEXT,json.dumps(state.snapshot(engineering=True)))
        state.select('cam02')
        self.assertNotIn(TEXT,json.dumps(state.snapshot(engineering=True)))

    def test_heartbeat_cannot_switch_or_restart_a_camera(self):
        clock = [0.]
        state = LiveState(clock=lambda:clock[0])
        state.select('cam01')
        generation = state.capture_stop
        with self.assertRaises(ValueError):state.heartbeat('cam02')
        self.assertEqual(state.camera_id,'cam01')
        clock[0] = 50.
        state.heartbeat('cam01')
        clock[0] = 61.
        state.expire()
        self.assertIs(state.capture_stop,generation)
        state.select(None)
        with self.assertRaises(ValueError):state.heartbeat('cam01')
        self.assertIsNone(state.camera_id)

    def test_stale_http_snapshots_omit_all_sensitive_evidence(self):
        clock = [10.]
        state = LiveState(clock=lambda:clock[0]); state.select('cam01')
        state.publish({'camera_id':'cam01','status':'live','published_at':10.,'frame':'pixels',
                       'crop':{'image':'pixels'},'debug':{'plate':TEXT}})
        clock[0] = 16.
        snapshot = state.snapshot(engineering=True)
        self.assertEqual(snapshot['frame_age_seconds'],6.)
        for key in ('frame','crop','debug'):self.assertNotIn(key,snapshot)

    def test_delivery_is_masked_durable_and_deduplicated_per_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(db_path=Path(directory)/'outbox.sqlite',central_api_url='http://example.invalid',
                                     edge_api_key='test-edge-key',hmac_key='test-hmac-key',edge_id='region-test')
            clock = [0.]
            delivery = SharedDelivery(config,'general',clock=lambda:clock[0])
            candidate = SimpleNamespace(plate=TEXT,score=.99)
            self.assertTrue(delivery.accept('cam01','1',candidate,500))
            self.assertFalse(delivery.accept('cam01','2',candidate,1000))
            self.assertTrue(delivery.accept('cam02','1',candidate,1500))
            self.assertEqual(delivery.depth(),2)
            batch = delivery.outbox.next_batch()
            self.assertNotIn(TEXT,json.dumps(batch))
            self.assertEqual(batch[0][2]['masked_plate'],'AP******34')
            self.assertIn('unverified',batch[0][2]['model_version'])
            persisted = SharedDelivery(config,'general')
            self.assertEqual(persisted.depth(),2)

    def test_only_accepted_observations_invoke_delivery(self):
        calls = []
        engine = self.engine(lambda *args:calls.append(args))
        track = SimpleNamespace(track_id='1',box=(0,0,180,60),vehicle_class=2)
        for i in range(2):engine.process(self.image(i),[track],i*500)
        self.assertEqual(calls,[])
        engine.process(self.image(2),[track],1000)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][1].plate,TEXT)

    def test_character_diagnostics_respect_ctc_repeat_and_blank(self):
        tokens = ('','A','B')
        rows = [[0.,1.,0.],[0.,1.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
        self.assertEqual(''.join(row['character'] for row in characters(rows,tokens)),'AAB')
