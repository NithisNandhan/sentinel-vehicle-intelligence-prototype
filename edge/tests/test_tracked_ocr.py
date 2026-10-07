import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import base64

import numpy as np

from sentinel_edge.experimental_detectors import BMD_TO_COCO, PROFILES, VehicleTrack, VehicleTracker, verified
from sentinel_edge.tracked_ocr import TrackedOCR
from sentinel_edge.detector_ocr_shadow import LatestFrames
from sentinel_edge.stream import VideoFrame
from sentinel_edge.live_detection_viewer import LiveState
from sentinel_edge.tracked_live_viewer import observe_tracked


TOKENS = ('', *'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
TEXT = 'AP01AB1234'


def posterior(_):
    rows = []
    for letter in TEXT:
        row = [0.]*len(TOKENS); row[TOKENS.index(letter)] = 1.
        rows.append(row)
    return rows


class TrackedOCRTests(unittest.TestCase):
    def engine(self):
        self.crops = []
        def probabilities(crop):
            self.crops.append(crop.copy())
            return posterior(crop)
        return TrackedOCR(SimpleNamespace(tokens=TOKENS,probabilities=probabilities),
                          SimpleNamespace(box=lambda image:(5,5,150,35)))

    def image(self,value):
        image = np.full((60,180,3),100,np.uint8)
        image[10,10] = value
        return image

    def tracks(self):
        return [VehicleTrack('1',(0,0,180,60),2)]

    def test_native_pixels_and_three_distinct_frame_acceptance(self):
        engine = self.engine()
        for i in range(3):
            image = self.image(i)
            displays = engine.process(image,self.tracks(),i*500)
        np.testing.assert_array_equal(self.crops[0],self.image(0)[5:35,5:150])
        self.assertEqual(displays[0]['observations'],3)
        self.assertEqual(displays[0]['masked_plate'],'AP******34')
        self.assertEqual(engine.summary()['unique_model_accepted_unverified'],1)
        self.assertNotIn(TEXT,json.dumps(engine.summary()))
        self.assertNotIn(TEXT,json.dumps({k:v for k,v in displays[0].items() if k!='crop'}))

    def test_duplicates_do_not_create_extra_evidence_or_inference(self):
        engine = self.engine()
        for stamp in (0,500,1000):engine.process(self.image(1),self.tracks(),stamp)
        self.assertEqual(len(self.crops),1)
        self.assertEqual(engine.histories['1'].reader.observation_count,1)
        self.assertEqual(engine.summary()['unique_model_accepted_unverified'],0)

    def test_old_ocr_evidence_is_cleared_even_if_vehicle_stays_tracked(self):
        engine = self.engine()
        engine.process(self.image(1),self.tracks(),0)
        engine.process(self.image(2),self.tracks(),500)
        engine.process(self.image(3),self.tracks(),3500)
        self.assertEqual(engine.histories['1'].reader.observation_count,1)

    def test_class_switch_and_pts_reversal_clear_history(self):
        engine = self.engine()
        engine.process(self.image(1),self.tracks(),1000)
        engine.process(self.image(2),[VehicleTrack('1',(0,0,180,60),7)],1500)
        self.assertEqual(engine.histories['1'].reader.observation_count,1)
        engine.process(self.image(3),self.tracks(),0)
        self.assertEqual(engine.histories['1'].reader.observation_count,1)
        self.assertEqual(engine.metrics['scene_resets'],1)

    def test_multiple_vehicles_never_share_the_same_reader(self):
        engine = self.engine()
        engine.process(self.image(1),[VehicleTrack(str(i),(0,0,180,60),2) for i in range(3)],0)
        self.assertEqual(len(engine.histories),3)
        self.assertEqual(len({id(s.reader) for s in engine.histories.values()}),3)

    def test_default_search_visits_every_eligible_track_without_three_roi_cap(self):
        engine = self.engine()
        tracks = [VehicleTrack(str(i),(0,0,180,60),2) for i in range(10)]
        with patch('sentinel_edge.tracked_ocr.crop_quality',return_value=.9):
            displays = engine.process(self.image(1),tracks,0)
        self.assertEqual(len(displays),10)
        self.assertEqual(engine.metrics['eligible_vehicle_tracks'],10)
        self.assertEqual(engine.metrics['plate_roi_searches'],10)
        self.assertEqual(engine.metrics['plate_roi_unsearched_tracks'],0)
        self.assertEqual(len(self.crops),10)

    def test_debug_track_list_includes_every_active_track(self):
        from sentinel_edge.tracked_ocr import MAX_TRACK_HISTORIES
        engine = TrackedOCR(SimpleNamespace(tokens=TOKENS,probabilities=posterior),
            SimpleNamespace(box=lambda image:(5,5,150,35)),debug=True)
        tracks = [VehicleTrack(str(i),(0,0,180,60),2) for i in range(20)]
        with patch('sentinel_edge.tracked_ocr.crop_quality',return_value=.9), patch(
                'sentinel_edge.recognition_debug.describe_track',
                side_effect=lambda key,state,samples,threshold:{'track_id':key}):
            engine.process(self.image(1),tracks,0)
            snapshot = engine.debug_snapshot()
        self.assertEqual(len(snapshot['tracks']),20)
        self.assertEqual(snapshot['roi_budget'],'all eligible tracks')
        self.assertEqual(snapshot['track_history_capacity'],MAX_TRACK_HISTORIES)

    def test_frame_cache_saves_computation_not_temporal_evidence(self):
        first = self.engine(); second = self.engine()
        cache = {}
        first.process(self.image(1),self.tracks(),0,posterior_cache=cache)
        second.process(self.image(1),self.tracks(),0,posterior_cache=cache)
        self.assertEqual(first.metrics['ocr_calls'],1)
        self.assertEqual(second.metrics['ocr_calls'],0)
        self.assertEqual(second.histories['1'].reader.observation_count,1)

    def test_pts_and_roi_budget_are_bounded(self):
        engine = self.engine()
        with self.assertRaises(ValueError):engine.process(self.image(1),[],None)
        with self.assertRaises(ValueError):engine.process(self.image(1),[],True)
        with self.assertRaises(ValueError):TrackedOCR(None,None,roi_budget=0)

    def test_viewer_preserves_original_crop_and_masks_tracked_reading(self):
        import cv2
        images = [self.image(i) for i in range(3)]
        pending = iter(enumerate([VideoFrame(image,i*500,False) for i,image in enumerate(images)],1))
        class Tap:
            decoded = 3
            failed = False
            finished = False
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def take(self,sequence):
                item = next(pending,None)
                self.finished = item is None
                if self.finished:state.capture_stop.set()
                return item
        resets = []
        vehicle = SimpleNamespace(reset=lambda:resets.append(True),track=lambda image:self.tracks())
        state = LiveState()
        state.select('cam01')
        with patch('sentinel_edge.tracked_live_viewer.LatestFrames',Tap), patch(
                'sentinel_edge.live_detection_viewer.viewer_frames',return_value=iter(())):
            observe_tracked(SimpleNamespace(ocr_min_confidence=.85),SimpleNamespace(camera_id='cam01'),
                SimpleNamespace(tokens=TOKENS,probabilities=posterior),
                SimpleNamespace(box=lambda image:(5,5,150,35)),vehicle,state,state.capture_stop)
        result = state.snapshot()
        self.assertEqual(len(resets),1)
        self.assertEqual(result['crop']['observations'],3)
        self.assertEqual(result['crop']['masked_plate'],'AP******34')
        self.assertNotIn(TEXT,json.dumps(result))
        crop = cv2.imdecode(np.frombuffer(base64.b64decode(result['crop']['image']),np.uint8),cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(crop,images[-1][5:35,5:150])
        # Rendering overlays must never mutate the inference source.
        np.testing.assert_array_equal(images[-1],self.image(2))


class DetectorContractTests(unittest.TestCase):
    def test_vehicle_boxes_without_bytetrack_ids_are_retained_with_local_association(self):
        def detection(box,identity=None):
            coordinate = SimpleNamespace(cpu=lambda:SimpleNamespace(tolist=lambda:box))
            return SimpleNamespace(cls=SimpleNamespace(item=lambda:2),id=identity,xyxy=[coordinate])
        class Boxes(list):
            id = None
        results = iter([
            SimpleNamespace(boxes=Boxes([detection([0,0,80,80]),detection([100,0,180,80])])),
            SimpleNamespace(boxes=Boxes([detection([5,0,85,80]),detection([105,0,185,80])])),
        ])
        model = VehicleTracker.__new__(VehicleTracker)
        model.name='yolov8n'
        model.model=SimpleNamespace(track=lambda *args,**kwargs:[next(results)])
        first = model.track(np.zeros((100,200,3),np.uint8))
        second = model.track(np.zeros((100,200,3),np.uint8))
        self.assertEqual(len(first),2)
        self.assertEqual({track.track_id for track in first},{track.track_id for track in second})
        self.assertTrue(all(track.track_id.startswith('det-') for track in first))

    def test_tampered_assets_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model'
            path.write_bytes(b'approved')
            digest = hashlib.sha256(b'approved').hexdigest()
            self.assertEqual(verified(path,digest),path)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):verified(path,digest)

    def test_bmd_taxonomy_is_not_filtered_as_coco(self):
        seen = {}
        item = SimpleNamespace(cls=SimpleNamespace(item=lambda:6),id=SimpleNamespace(item=lambda:9),
            xyxy=[SimpleNamespace(cpu=lambda:SimpleNamespace(tolist=lambda:[0,0,100,100]))])
        class Boxes(list):
            id = object()
        def track(image,**kwargs):
            seen.update(kwargs)
            return [SimpleNamespace(boxes=Boxes([item]))]
        model = VehicleTracker.__new__(VehicleTracker)
        model.name='bmd'; model.model=SimpleNamespace(track=track)
        result = model.track(np.zeros((100,100,3),np.uint8))
        self.assertEqual(set(seen['classes']),set(range(13)))
        self.assertEqual(result[0].vehicle_class,2)
        self.assertEqual(BMD_TO_COCO[7],3)
        self.assertEqual(PROFILES['plate_adapted'],('yolo26s','adapted'))
        self.assertEqual(seen['device'],'cpu')

    def test_latest_frame_retains_cut_when_intermediate_frames_are_dropped(self):
        frames = [('rtsp',VideoFrame(np.zeros((10,10,3),np.uint8),i*500,i==1)) for i in range(3)]
        stop=threading.Event()
        with LatestFrames(iter(frames),stop) as tap:
            tap.thread.join(timeout=2)
            sequence,frame = tap.take(0)
            self.assertEqual(sequence,3)
            self.assertEqual(frame.pts_ms,1000)
            self.assertTrue(frame.discontinuity)
            self.assertEqual(tap.overwritten,2)
            self.assertTrue(tap.finished)
        self.assertIsNone(tap.latest)
        self.assertFalse(stop.is_set())

    def test_decode_tap_shutdown_cancels_only_its_own_source(self):
        parent = threading.Event()
        tokens = []
        def source(token):
            tokens.append(token)
            token.wait(2)
            return
            yield  # cancellation-aware generator without any media
        with LatestFrames(source,parent) as tap:
            pass
        self.assertFalse(parent.is_set())
        self.assertTrue(tokens[0].is_set())
        self.assertFalse(tap.thread.is_alive())


if __name__=='__main__':unittest.main()
