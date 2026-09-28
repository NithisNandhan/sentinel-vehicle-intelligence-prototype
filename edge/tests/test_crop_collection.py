import threading
import unittest
from types import SimpleNamespace

import numpy as np

from sentinel_edge.plate_search import TargetedPlateSearch
from sentinel_edge.sampled_frames import SampledFrames
from sentinel_edge.stream import VideoFrame
from sentinel_edge.reconstruction import CropBank
from sentinel_edge.crop_collector import CropCollector
from sentinel_edge.experimental_detectors import VehicleTrack


class SearchTests(unittest.TestCase):
    def test_primary_success_avoids_tiles(self):
        search = TargetedPlateSearch(SimpleNamespace(box=lambda image:(10,10,110,40)))
        self.assertEqual(search.box(np.zeros((200,400,3),np.uint8)), (10,10,110,40))
        self.assertEqual(search.last_calls,1)
        self.assertFalse(search.last_fallback)

    def test_native_tile_coordinates_and_bounded_work(self):
        image = np.arange(200*400*3,dtype=np.uint8).reshape(200,400,3)
        calls = []
        def detect(region):
            calls.append(region.copy())
            return None if len(calls)==1 else (10,10,110,40)
        search = TargetedPlateSearch(SimpleNamespace(box=detect))
        self.assertEqual(search.box(image),(10,10,110,40))
        self.assertEqual(search.last_calls,5)
        np.testing.assert_array_equal(calls[2],image[0:130,140:400])
        np.testing.assert_array_equal(image[0:130,140:400],calls[2])

    def test_small_roi_and_invalid_proposals_do_not_trigger_ocr(self):
        search = TargetedPlateSearch(SimpleNamespace(box=lambda image:(0,0,2,2)))
        self.assertIsNone(search.box(np.zeros((40,100,3),np.uint8)))
        self.assertEqual(search.last_calls,1)
        self.assertIsNone(search.box(np.zeros((200,400,3),np.uint8)))
        self.assertEqual(search.last_calls,5)


class QueueTests(unittest.TestCase):
    def run_tap(self, points, **kwargs):
        parent = threading.Event()
        frames = [('rtsp',VideoFrame(np.full((10,10,3),i,np.uint8),pts,cut))
                  for i,(pts,cut) in enumerate(points)]
        tap = SampledFrames(iter(frames),parent,**kwargs)
        tap.__enter__()
        tap.thread.join(2)
        self.assertFalse(tap.thread.is_alive())
        return tap,parent

    def test_distinct_sampled_views_survive_slow_consumer(self):
        tap,parent = self.run_tap([(0,False),(100,False),(500,False),(1000,False)])
        self.assertEqual(tap.sampled,3)
        self.assertEqual([tap.take(0)[1].pts_ms for _ in range(3)],[0,500,1000])
        tap.__exit__()
        self.assertEqual(tap.bytes_used,0)
        self.assertFalse(parent.is_set())

    def test_cut_flushes_old_vehicle_pixels_even_if_cut_frame_unsampled(self):
        tap,_ = self.run_tap([(0,False),(500,False),(600,True),(1100,False)])
        _,frame = tap.take(0)
        self.assertEqual(frame.pts_ms,600)
        self.assertTrue(frame.discontinuity)
        self.assertEqual(tap.dropped_samples,2)
        tap.__exit__()

    def test_memory_and_age_bounds(self):
        tap,_ = self.run_tap([(i*500,False) for i in range(12)],max_bytes=600,max_frames=8)
        self.assertLessEqual(tap.peak_bytes,600)
        self.assertEqual(len(tap.queue),2)
        self.assertEqual(tap.dropped_samples,10)
        self.assertEqual(tap.take(0)[1].pts_ms,5000)
        tap.__exit__()

    def test_reversal_and_missing_pts(self):
        tap,_ = self.run_tap([(0,False),(500,False),(None,False),(100,False),(600,False)])
        self.assertEqual(tap.take(0)[1].pts_ms,100)
        tap.__exit__()


class QualityBankTests(unittest.TestCase):
    def test_best_older_view_and_recent_views_retained_without_adding_evidence(self):
        bank = CropBank(max_frames=3,retain_best=True,max_age_ms=2500)
        for i,quality in enumerate((1.,.2,.4,.3)):
            bank.add('a',np.full((10,30,3),i,np.uint8),i*500,quality=quality)
        self.assertEqual([s.pts_ms for s in bank.get('a')],[0,1000,1500])
        self.assertEqual(bank.bytes_used,2700)
        bank.add('a',np.full((10,30,3),5,np.uint8),4000,quality=.2)
        self.assertEqual([s.pts_ms for s in bank.get('a')],[1500,4000])


class CollectorTests(unittest.TestCase):
    def test_collects_crops_before_ocr_consumer_and_releases_all_bytes(self):
        parent = threading.Event()
        frames = [('rtsp',VideoFrame(np.full((60,180,3),i,np.uint8),i*500,False)) for i in range(3)]
        resets = []
        vehicle = SimpleNamespace(reset=lambda:resets.append(True),track=lambda image:[VehicleTrack('1',(0,0,180,60),2)])
        detector = SimpleNamespace(box=lambda image:(10,10,110,40))
        with CropCollector(iter(frames),parent,vehicle,detector) as collector:
            collector.thread.join(2)
            self.assertFalse(collector.failed)
            self.assertEqual(collector.collected_crops,3)
            self.assertEqual(len(collector.queue),3)
            first = collector.take(0)[1]
            self.assertEqual(first.plate_cache['1']['crop'].shape,(30,100,3))
            self.assertGreater(first.captured_at,0)
            for _ in range(2):collector.take(0)
            self.assertEqual(collector.bytes_used,0)
        self.assertEqual(len(resets),1)
        self.assertFalse(parent.is_set())

    def test_prefetched_native_crop_does_not_rerun_detector(self):
        from sentinel_edge.tracked_ocr import TrackedOCR
        tokens = ('','A')
        calls = []
        ocr = SimpleNamespace(tokens=tokens,probabilities=lambda image:[[1.,0.]])
        detector = SimpleNamespace(box=lambda image:calls.append(True))
        pipeline = TrackedOCR(ocr,detector)
        image = np.zeros((60,180,3),np.uint8)
        crop = image[10:40,10:110].copy()
        cache = {'1':dict(box=(10,10,110,40),crop=crop,calls=5,fallback=True,ms=10)}
        result = pipeline.process(image,[VehicleTrack('1',(0,0,180,60),2)],0,plate_cache=cache)
        self.assertEqual(calls,[])
        np.testing.assert_array_equal(result[0]['crop'],crop)
        self.assertEqual(pipeline.metrics['plate_detector_calls'],5)
        self.assertEqual(pipeline.stage_ms['plate'],10)


if __name__=='__main__':
    unittest.main()
