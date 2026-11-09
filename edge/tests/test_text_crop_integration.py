"""Crop ownership, native evidence and derived-view integration regressions."""
import hashlib
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ctc_fixtures import TOKENS, rows_for
from sentinel_edge.crop_collector import CropCollector
from sentinel_edge.experimental_detectors import VehicleTrack
from sentinel_edge.plate_options import PlateOptions
from sentinel_edge.plate_proposals import ProposalPool, checked_quad
from sentinel_edge.stream import VideoFrame
from sentinel_edge.tracked_live_viewer import load_text_detector
from sentinel_edge.tracked_ocr import TrackedOCR


class TextCropIntegrationTests(unittest.TestCase):
    def test_text_outlining_runs_before_optional_tiles(self):
        order=[]
        primary=SimpleNamespace(box=lambda image:None)
        text=SimpleNamespace(polygons=lambda image:order.append('text') or [])
        tiles=SimpleNamespace(boxes=lambda image:order.append('tile') or [])
        pool=ProposalPool(primary,text_detector=text,tile_detector=tiles,min_tile_region=100,
                          max_calls_per_frame=2,clock=lambda:0.)
        pool.begin_frame()
        pool.box(np.zeros((400,1000,3),np.uint8))
        self.assertEqual(order,['text','tile'])

    def test_collector_resets_budget_for_each_frame(self):
        now=[0.]
        def primary(image):
            now[0]+=.1
            return None
        text=SimpleNamespace(polygons=lambda image:[[(20,40),(120,40),(120,65),(20,65)]])
        pool=ProposalPool(SimpleNamespace(box=primary),text_detector=text,frame_budget_ms=200,clock=lambda:now[0])
        vehicle=SimpleNamespace(reset=lambda:None,track=lambda image:[VehicleTrack('1',(0,0,160,80),2)])
        frames=[('recorded',VideoFrame(np.full((80,160,3),i,np.uint8),i*500,False)) for i in range(3)]
        with CropCollector(iter(frames),threading.Event(),vehicle,pool) as collector:
            collector.thread.join(2)
            self.assertFalse(collector.failed)
            self.assertEqual(pool.metrics['frames_started'],3)
            self.assertEqual(pool.metrics['text_detector_calls'],3)
            for _,packet in collector.queue:
                self.assertEqual(packet.plate_cache['1']['sources'],('text',))
                self.assertIsNotNone(packet.plate_cache['1']['text_quad'])

    def test_one_source_has_native_evidence_and_one_rectified_ocr_input(self):
        image=np.arange(80*160*3,dtype=np.uint8).reshape(80,160,3)
        native=image[40:65,20:120].copy()
        inputs=[]; resets=[]
        def probabilities(crop):
            inputs.append(crop.copy())
            return rows_for(list('MH49BE5557'))
        ocr=SimpleNamespace(tokens=TOKENS,probabilities=probabilities)
        detector=SimpleNamespace(begin_frame=lambda:resets.append(1))
        pipeline=TrackedOCR(ocr,detector,debug=True,options=PlateOptions(text_rectify=True))
        cache={'1':dict(box=(20,40,120,65),crop=native,ms=0.,calls=1,fallback=False,
                       sources=('text',),text_quad=((25,43),(115,43),(115,61),(25,61)))}
        track=VehicleTrack('1',(0,0,160,80),2)
        pipeline.process(image,[track],0,plate_cache=cache)
        history=pipeline.histories['1']
        self.assertEqual(resets,[], 'OCR consumer cannot reset the collector-owned detector')
        self.assertEqual(len(inputs),1)
        self.assertNotEqual(inputs[0].shape,native.shape)
        self.assertEqual(history.reader.observation_count,1)
        self.assertEqual(history.reader.samples[0][1],hashlib.sha256(native.tobytes()).hexdigest())
        np.testing.assert_array_equal(pipeline.crop_bank.get('1')[0].image,native)
        frame=pipeline.debug_snapshot()['tracks'][0]['frames'][0]
        self.assertEqual(frame['crop_sources'],['text'])
        self.assertEqual(frame['working_view'],'text_quad')
        self.assertTrue(frame['working_image'])
        self.assertNotEqual(frame['working_image'],frame['image'])
        pipeline.process(image,[track],500,plate_cache=cache)
        self.assertEqual(len(inputs),1, 'Identical native pixels do not gain another vote')
        self.assertEqual(history.reader.observation_count,1)

    def test_corners_must_be_finite_convex_and_inside_source(self):
        shape=(80,160,3)
        for quad in [None,[(0,0)]*4,[(0,0),(160,0),(160,10),(0,10)],
                     [(10,10),(90,40),(90,10),(10,40)],[(0,0),(80,0),(80,float('nan')),(0,20)]]:
            self.assertIsNone(checked_quad(quad,shape))

    def test_explicitly_configured_model_cannot_fail_silently(self):
        with patch.dict('os.environ',{'SENTINEL_TEXT_DETECTOR_ASSETS':'/missing'}):
            with patch('sentinel_edge.text_boundaries.TextBoundaryDetector',side_effect=ValueError('missing')):
                with self.assertRaisesRegex(RuntimeError,'Configured text detector'):
                    load_text_detector()


if __name__=='__main__':
    unittest.main()
