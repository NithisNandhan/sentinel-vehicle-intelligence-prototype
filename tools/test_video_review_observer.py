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
    def test_capture_records_effective_identity_guards_and_source_hashes(self):
        from dataclasses import asdict
        from sentinel_edge.plate_continuity import DEFAULT_POLICY
        settings=runner.identity_guard_settings()
        self.assertEqual(settings['plate_continuity'],asdict(DEFAULT_POLICY))
        self.assertEqual(settings['plate_continuity']['max_aspect_change'],2.)
        self.assertEqual(settings['plate_ownership']['minimum_area_fraction'],.85)
        self.assertTrue(settings['plate_ownership']['before_ocr_and_fusion'])
        conflict=settings['independent_ocr_conflict']
        self.assertEqual(conflict['equal_length_differing_characters_min'],3)
        self.assertEqual(conflict['terminal_differing_characters_min'],2)
        self.assertEqual(conflict['character_score_min'],.9)
        self.assertTrue(conflict['veto_only_never_merge'])
        self.assertEqual(set(settings['source_sha256']),{'plate_ownership','plate_continuity','plate_identity'})
        for digest in settings['source_sha256'].values():self.assertRegex(digest,r'^[a-f0-9]{64}$')

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
        def sample(plate,pts,score,minimum,observations=3,status='uncertain',pixel=70,
                   passage_id='passage-seven',ocr_pts=None,track_id=7):
            track.track_id=track_id
            candidate=SimpleNamespace(plate=plate,score=score,minimum_character_score=minimum,format_adjusted=False)
            reader=SimpleNamespace(observation_count=observations,samples=[(pts if ocr_pts is None else ocr_pts,)],candidates=lambda:[candidate])
            pipeline.histories[track_id]=SimpleNamespace(reader=reader,passage_id=passage_id,
                box=track.box,passage_start_reason='initial' if passage_id=='passage-seven' else 'plate_region')
            frame=SimpleNamespace(pts_ms=pts,image=np.full((80,120,3),pixel,dtype=np.uint8))
            display=dict(track_id=track_id,box=[10,30,110,60],crop=np.full((30,100,3),pixel,dtype=np.uint8),
                         observations=observations,ocr_status=status,passage_id=passage_id,vehicle_box=track.box)
            observer.on_sample(frame,[track],[display],pipeline)
        return observer,pipeline,sample

    def test_strongest_window_matches_panel_instead_of_last_guess(self):
        with tempfile.TemporaryDirectory() as folder:
            observer,_,sample=self.observer_fixture(folder)
            sample('MH01AB1234',500,.95,.9,pixel=40)
            sample('MH01AB1284',1000,.7,.3,observations=8,pixel=120)
            prediction=observer.predictions['passage-s0-passage-seven']
            self.assertEqual(prediction['top_guess'],'MH01AB1234')
            self.assertEqual(prediction['candidate_ms'],500)
            self.assertEqual(prediction['observations'],3)
            observer.export_assets()
            image=__import__('cv2').imread(str(Path(folder)/observer.tracks['passage-s0-passage-seven']['samples'][0]['path']))
            self.assertEqual(int(image[0,0,0]),40)
            self.assertEqual(observer.tracks['passage-s0-passage-seven']['samples'][0]['position_ms'],500)

    def test_model_accepted_window_outranks_later_uncertain_window(self):
        with tempfile.TemporaryDirectory() as folder:
            observer,_,sample=self.observer_fixture(folder)
            sample('MH01AB1234',500,.86,.76,status='accepted by model — NOT verified')
            sample('MH01AB1284',1000,.99,.98,observations=8)
            self.assertEqual(observer.predictions['passage-s0-passage-seven']['top_guess'],'MH01AB1234')

    def test_three_originals_outrank_isolated_high_score_suggestion(self):
        with tempfile.TemporaryDirectory() as folder:
            observer,_,sample=self.observer_fixture(folder)
            sample('MH01AB1284',500,.99,.99,observations=1)
            sample('MH01AB1234',1000,.8,.7,observations=3)
            self.assertEqual(observer.predictions['passage-s0-passage-seven']['top_guess'],'MH01AB1234')

    def test_acceptance_tracks_current_scene_segment(self):
        with tempfile.TemporaryDirectory() as folder:
            observer,pipeline,sample=self.observer_fixture(folder)
            sample('MH01AB1234',500,.9,.8)
            pipeline.metrics['scene_resets']=1
            observer.accept('synthetic',7,SimpleNamespace(plate='MH01AB1234'),1000)
            self.assertEqual(observer.predictions['passage-s1-passage-seven']['accepted'][0]['pts_ms'],1000)

    def test_reused_tracker_id_cannot_mix_different_passages(self):
        with tempfile.TemporaryDirectory() as folder:
            observer,pipeline,sample=self.observer_fixture(folder)
            sample('MH01AB1234',500,.95,.9,pixel=40)
            observer.accept('synthetic',7,SimpleNamespace(plate='MH01AB1234'),500)
            sample('MH02CD5678',1000,.8,.7,pixel=120,passage_id='next-passage')
            observer.accept('synthetic',7,SimpleNamespace(plate='MH02CD5678'),1000)
            self.assertEqual(set(observer.tracks),{'passage-s0-passage-seven','passage-s0-next-passage'})
            for uid,text,pixel,pts in [('passage-s0-passage-seven','MH01AB1234',40,500),
                                       ('passage-s0-next-passage','MH02CD5678',120,1000)]:
                row=observer.tracks[uid];prediction=observer.predictions[uid]
                self.assertEqual(row['plate_crop_count'],1)
                self.assertEqual([o['position_ms'] for o in row['plate_observations']],[pts])
                self.assertEqual(prediction['top_guess'],text)
                self.assertEqual([a['plate'] for a in prediction['accepted']],[text])
                self.assertEqual(int(observer.native[uid]['selected'][0][0,0,0]),pixel)
            self.assertEqual(observer.tracks['passage-s0-next-passage']['passage_start_reason'],'plate_region')

    def test_reassociated_tracker_ids_share_only_the_same_passage(self):
        with tempfile.TemporaryDirectory() as folder:
            observer,_,sample=self.observer_fixture(folder)
            sample('MH01AB1234',500,.9,.8)
            sample('MH01AB1234',1000,.95,.9,track_id=12)
            self.assertEqual(len(observer.tracks),1)
            self.assertEqual(observer.tracks['passage-s0-passage-seven']['tracker_ids'],['7','12'])

    def test_duplicate_display_uses_original_admitted_crop(self):
        with tempfile.TemporaryDirectory() as folder:
            observer,_,sample=self.observer_fixture(folder)
            sample('MH01AB1234',500,.9,.8,pixel=40)
            sample('MH01AB1234',1000,.9,.8,status='accepted by model — NOT verified',pixel=120,ocr_pts=500)
            selected=observer.native['passage-s0-passage-seven']['selected']
            self.assertEqual(selected[1]['position_ms'],500)
            self.assertEqual(int(selected[0][0,0,0]),40)

    def test_mismatched_or_missing_passage_rejects_plate_export(self):
        with tempfile.TemporaryDirectory() as folder:
            observer,pipeline,sample=self.observer_fixture(folder)
            sample('MH01AB1234',500,.9,.8)
            frame=SimpleNamespace(pts_ms=1000,image=np.zeros((80,120,3),dtype=np.uint8))
            display=dict(track_id=7,passage_id='wrong',box=[10,30,110,60],
                         crop=np.ones((30,100,3),dtype=np.uint8),observations=3,ocr_status='uncertain')
            with self.assertRaisesRegex(ValueError,'matching recognition passage'):
                observer.on_sample(frame,[],[display],pipeline)
            pipeline.histories.clear()
            with self.assertRaisesRegex(ValueError,'matching recognition passage'):
                observer.on_sample(frame,[],[display],pipeline)

    def test_finalization_records_boundaries_without_upgrading_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            observer,_,sample=self.observer_fixture(folder)
            sample('MH01AB1234',500,.9,.8)
            observer.on_passage(SimpleNamespace(track_ids=(7,12),passage_id='passage-seven',
                first_pts=500,last_pts=800,cause='plate_region'))
            row=observer.tracks['passage-s0-passage-seven']
            self.assertEqual(row['tracker_ids'],['7','12'])
            self.assertEqual(row['passage_end_reason'],'plate_region')
            self.assertEqual(row['passage_last_ms'],800)
            self.assertEqual(observer.predictions[row['id']]['accepted'],[])

    def test_unprocessed_detection_can_never_receive_an_accepted_reading(self):
        with tempfile.TemporaryDirectory() as folder:
            observer,pipeline,_=self.observer_fixture(folder);observer.pipeline=pipeline
            with self.assertRaisesRegex(ValueError,'no recognition passage'):
                observer.accept('synthetic',7,SimpleNamespace(plate='MH01AB1234'),500)

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
            state.pipeline=SimpleNamespace(metrics={'scene_resets':0},histories={
                7:SimpleNamespace(passage_id='fixture-passage')})
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
            manifest=json.loads((root/'manifest.json').read_text())
            self.assertEqual(manifest['settings']['grouping_policy'],'recognition_passage_v1')
            self.assertIn('identity_guards',manifest['settings'])
            self.assertEqual(predictions['tracks']['passage-s0-fixture-passage']['accepted'],
                             [{'plate': 'MH01AB1234', 'pts_ms': 500}])

    def test_incomplete_acceptance_capture_cannot_be_published_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Acceptance observer'):
                self.run_fixture(directory, summary_acceptances=2)
            self.assertFalse((Path(directory)/'fixture'/'manifest.json').exists())

    def test_guard_changes_during_capture_cannot_publish_a_misleading_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner,'identity_guard_settings',side_effect=[{'source':'before'},{'source':'after'}]):
                with self.assertRaisesRegex(ValueError,'source changed during the capture'):
                    self.run_fixture(directory)
            self.assertFalse((Path(directory)/'fixture'/'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
