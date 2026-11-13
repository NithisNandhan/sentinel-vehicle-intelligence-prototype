"""Synthetic wiring checks; no models, real recording or real labels are used."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'edge'))
import run_video_review as runner
import sentinel_edge.config
import sentinel_edge.live_detection_viewer
import sentinel_edge.plate_options
import sentinel_edge.tracked_live_viewer


class ObserverIntegrationTests(unittest.TestCase):
    def reference_fixture(self,folder,tracks):
        root=Path(folder);review=root/'manual';review.mkdir();(review/'ocr-model-dev').mkdir()
        source='a'*64
        runner.write_json(review/'manifest.json',dict(source_sha256=source,samples=[dict(id='01',position_ms=500,box=[10,20,110,50])]))
        runner.write_json(review/'labels.json',dict(source_sha256=source,labels=[dict(id='01',status='readable',plate='MH01AB1234')]))
        (review/'ocr-model-dev/rectify.py').write_text('CORNERS={}\nTEXT_REGIONS={}\n')
        view=root/'view.json';runner.write_json(view,dict(source_sha256=source,references=[dict(number=1,reference_id='01',actual='MH01AB1234')],
            bindings=dict(manifest_sha256=runner.file_hash(review/'manifest.json'),labels_sha256=runner.file_hash(review/'labels.json'))))
        return dict(source=dict(sha256=source),tracks=tracks),review,view

    def test_old_reference_links_use_native_geometry_and_exact_time_only(self):
        with tempfile.TemporaryDirectory() as folder:
            tracks=[dict(id='geometry',top_guess='WRONG',plate_observations=[dict(position_ms=500,box=[10,20,110,50])]),
                    dict(id='string_only',top_guess='MH01AB1234',plate_observations=[dict(position_ms=500,box=[300,300,400,330])])]
            manifest,review,view=self.reference_fixture(folder,tracks)
            result=runner.reference_coverage(manifest,review,view)
            self.assertFalse(result['predictions_used_for_linking']);self.assertEqual(result['labels_imported'],0)
            row=result['references'][0];self.assertEqual(row['status'],'unique_geometry_needs_review')
            self.assertEqual([c['track_id'] for c in row['candidates']],['geometry'])
            self.assertNotIn('MH01AB1234',json.dumps(result))

    def test_ambiguous_reference_never_chooses_a_track_or_imports_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            tracks=[dict(id=str(i),plate_observations=[dict(position_ms=500,box=[10,20,110,50])]) for i in range(2)]
            manifest,review,view=self.reference_fixture(folder,tracks)
            result=runner.reference_coverage(manifest,review,view)
            self.assertEqual(result['references'][0]['status'],'ambiguous_geometry')
            self.assertEqual(len(result['references'][0]['candidates']),2)
            self.assertEqual(result['labels_imported'],0)
            manifest['source']['sha256']='b'*64
            with self.assertRaises(ValueError):runner.reference_coverage(manifest,review,view)

    def observer_fixture(self,root):
        recording=SimpleNamespace(start_pts_ms=0)
        observer=runner.ReviewObserver(recording,Path(root))
        track=SimpleNamespace(track_id=7,box=[0,0,120,80],vehicle_class=2)
        pipeline=SimpleNamespace(metrics={'scene_resets':0},histories={})
        def sample(plate,pts,score,minimum,observations=3,status='uncertain',pixel=70):
            candidate=SimpleNamespace(plate=plate,score=score,minimum_character_score=minimum,format_adjusted=False)
            reader=SimpleNamespace(observation_count=observations,samples=[(pts,)],candidates=lambda:[candidate])
            pipeline.histories[7]=SimpleNamespace(reader=reader)
            frame=SimpleNamespace(pts_ms=pts,image=np.full((80,120,3),pixel,dtype=np.uint8))
            display=dict(track_id=7,box=[10,30,110,60],crop=np.full((30,100,3),pixel,dtype=np.uint8),
                         observations=observations,ocr_status=status)
            observer.on_sample(frame,[track],[display],pipeline)
        return observer,pipeline,sample

    def test_strongest_window_matches_panel_instead_of_last_guess(self):
        with tempfile.TemporaryDirectory() as folder:
            observer,_,sample=self.observer_fixture(folder)
            sample('MH01AB1234',500,.95,.9,pixel=40)
            sample('MH01AB1284',1000,.7,.3,observations=8,pixel=120)
            prediction=observer.predictions['track-s0-7']
            self.assertEqual(prediction['top_guess'],'MH01AB1234')
            self.assertEqual(prediction['candidate_ms'],500)
            self.assertEqual(prediction['observations'],3)
            observer.export_assets()
            image=__import__('cv2').imread(str(Path(folder)/observer.tracks['track-s0-7']['samples'][0]['path']))
            self.assertEqual(int(image[0,0,0]),40)
            self.assertEqual(observer.tracks['track-s0-7']['samples'][0]['position_ms'],500)

    def test_model_accepted_window_outranks_later_uncertain_window(self):
        with tempfile.TemporaryDirectory() as folder:
            observer,_,sample=self.observer_fixture(folder)
            sample('MH01AB1234',500,.86,.76,status='accepted by model — NOT verified')
            sample('MH01AB1284',1000,.99,.98,observations=8)
            self.assertEqual(observer.predictions['track-s0-7']['top_guess'],'MH01AB1234')

    def test_three_originals_outrank_isolated_high_score_suggestion(self):
        with tempfile.TemporaryDirectory() as folder:
            observer,_,sample=self.observer_fixture(folder)
            sample('MH01AB1284',500,.99,.99,observations=1)
            sample('MH01AB1234',1000,.8,.7,observations=3)
            self.assertEqual(observer.predictions['track-s0-7']['top_guess'],'MH01AB1234')

    def test_acceptance_tracks_current_scene_segment(self):
        with tempfile.TemporaryDirectory() as folder:
            observer,pipeline,sample=self.observer_fixture(folder)
            sample('MH01AB1234',500,.9,.8)
            pipeline.metrics['scene_resets']=1
            observer.accept('synthetic',7,SimpleNamespace(plate='MH01AB1234'),1000)
            self.assertEqual(observer.predictions['track-s1-7']['accepted'][0]['pts_ms'],1000)

    def run_fixture(self, directory, summary_acceptances=1):
        recording = SimpleNamespace(path=Path('synthetic.mp4'), sha256='b'*64,
                                    recording_id='synthetic', duration_ms=1000, start_pts_ms=0)
        context = MagicMock()
        context.__enter__.return_value.streams.video = [SimpleNamespace(
            width=640, height=480, average_rate=30, frames=30,
            codec_context=SimpleNamespace(name='h264'))]
        args = SimpleNamespace(output=Path(directory)/'fixture', recordings_dir=Path(directory),
                               recording_id='synthetic', cache=Path(directory),
                               detector_profile='general', commit='synthetic-fixture')

        def observe(config, camera, reader, detector, vehicle, state, stop, **kwargs):
            # The callback must reach the collector, without any central client.
            delivery = kwargs.get('delivery')
            self.assertIs(delivery, state)
            delivery.accept('synthetic', 7, SimpleNamespace(plate='MH01AB1234'), 500,
                            recorded=kwargs['recorded'])
            counted = kwargs['recorded']
            counted.decoded = 30
            counted.reached_eof = True
            counted.last_pts = 1000
            state.samples = 2
            return {'unique_model_accepted_unverified': summary_acceptances}

        with patch.object(runner.RecordingLibrary, 'scan', return_value=({'synthetic': recording}, [])), \
             patch.object(runner, 'file_hash', return_value='c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d'), \
             patch.dict(sys.modules, {'av': SimpleNamespace(open=MagicMock(return_value=context))}), \
             patch('sentinel_edge.config.Config.from_env', return_value=SimpleNamespace(ocr_min_confidence=.85)), \
             patch('sentinel_edge.live_detection_viewer.load_models', return_value=(None, None, None)), \
             patch('sentinel_edge.live_detection_viewer.model_identity', return_value={'model': 'synthetic'}), \
             patch('sentinel_edge.tracked_live_viewer.observe_tracked', side_effect=observe), \
             patch.object(runner.ReviewObserver, 'export_assets'):
            runner.main(args)
        return args.output

    def test_model_acceptance_is_captured_by_the_frozen_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.run_fixture(directory)
            predictions = json.loads((root/'predictions.json').read_text())
            self.assertEqual(predictions['tracks']['track-s0-7']['accepted'],
                             [{'plate': 'MH01AB1234', 'pts_ms': 500}])

    def test_incomplete_acceptance_capture_cannot_be_published_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Acceptance observer'):
                self.run_fixture(directory, summary_acceptances=2)
            self.assertFalse((Path(directory)/'fixture'/'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
