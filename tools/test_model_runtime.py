"""Preflight contracts without model files, media or CUDA dependencies."""
import json
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_model_runtime import OCR_SHA256,check_models


def fixtures(device='cpu',providers=None):
    reader=SimpleNamespace(probabilities=Mock(),device=device)
    detector=SimpleNamespace(boxes=Mock(),providers=providers or (['CPUExecutionProvider'] if device=='cpu' else ['CUDAExecutionProvider','CPUExecutionProvider']))
    vehicle=SimpleNamespace(track=Mock())
    loader=Mock(return_value=(reader,detector,vehicle))
    identity=dict(ocr_device=device,vehicle_device=device,plate_device=device,ocr_checkpoint_sha256=OCR_SHA256)
    return reader,detector,vehicle,loader,Mock(return_value=identity)


class ModelRuntimeTests(unittest.TestCase):
    def run_check(self,device='cpu',environment=None,providers=None):
        values=fixtures(device,providers)
        result=check_models(device,environment=environment or {},loader=values[3],identity_reader=values[4])
        return result,values

    def test_cpu_executes_three_actual_stages_using_model_only_config(self):
        result,(reader,detector,vehicle,loader,_) = self.run_check()
        self.assertEqual(result['device'],'cpu');self.assertEqual(result['model_preflight'],'passed')
        self.assertEqual(set(vars(loader.call_args.args[0])),{'model_cache_dir'})
        self.assertEqual(loader.call_args.args[0].model_cache_dir,Path('/models'))
        self.assertEqual(loader.call_args.kwargs['device'],'cpu')
        self.assertEqual(vehicle.track.call_args.args[0].shape,(720,1280,3))
        self.assertEqual(detector.boxes.call_args.args[0].shape,(512,512,3))
        self.assertEqual(reader.probabilities.call_args.args[0].shape,(32,120,3))
        for image in [vehicle.track.call_args.args[0],detector.boxes.call_args.args[0],reader.probabilities.call_args.args[0]]:
            self.assertFalse(image.any())

    def test_gpu_requires_cuda_and_all_stage_devices(self):
        result,_=self.run_check('gpu');self.assertEqual(result['device'],'gpu')
        with self.assertRaises(RuntimeError):self.run_check('gpu',providers=['CPUExecutionProvider'])
        values=fixtures('gpu');values[4].return_value['ocr_device']='cpu'
        with self.assertRaises(RuntimeError):check_models('gpu',environment={},loader=values[3],identity_reader=values[4])

    def test_cpu_rejects_gpu_session_even_if_metadata_claims_cpu(self):
        with self.assertRaises(RuntimeError):self.run_check(providers=['CUDAExecutionProvider','CPUExecutionProvider'])

    def test_flags_must_match_environment_before_loading(self):
        loader=Mock()
        for environment in ({'SENTINEL_OCR_DEVICE':'gpu'},{'RECORDED_DETECTOR_PROFILE':'general_quality'}):
            with self.subTest(environment=environment),self.assertRaises(RuntimeError):
                check_models('cpu','general',environment=environment,loader=loader,identity_reader=Mock())
        loader.assert_not_called()

    def test_changed_checkpoint_fails_closed(self):
        values=fixtures();values[4].return_value['ocr_checkpoint_sha256']='changed'
        with self.assertRaises(RuntimeError):check_models('cpu',environment={},loader=values[3],identity_reader=values[4])

    def test_optional_text_runtime_is_explicit_and_exercised(self):
        values=fixtures();text=SimpleNamespace(device='cpu',polygons=Mock(return_value=[]))
        environment={'SENTINEL_TEXT_DETECTOR_ASSETS':'/text-model','SENTINEL_TEXT_DETECTOR_DEVICE':'cpu'}
        result=check_models('cpu',environment=environment,loader=values[3],identity_reader=values[4],text_loader=lambda:text)
        self.assertEqual(result['text_detector'],'cpu');text.polygons.assert_called_once()
        self.assertEqual(values[2].track.call_count,2)
        with self.assertRaises(RuntimeError):
            check_models('cpu',environment={**environment,'SENTINEL_TEXT_DETECTOR_DEVICE':'gpu'},loader=values[3],identity_reader=values[4],text_loader=lambda:text)

    def test_legacy_gpu_wrapper_keeps_success_field(self):
        import check_gpu_runtime
        with patch.object(check_gpu_runtime,'check_models',return_value={'model_preflight':'passed','device':'gpu'}) as check,patch('builtins.print') as output:
            check_gpu_runtime.main()
        self.assertEqual(check.call_args.args[0],'gpu')
        self.assertEqual(json.loads(output.call_args.args[0])['gpu_preflight'],'passed')

    @unittest.skipUnless(shutil.which('pwsh'),'PowerShell unavailable')
    def test_models_only_restore_invokes_only_two_archive_restores(self):
        script=Path(__file__).resolve().parents[1]/'scripts/restore-teammate-handoff.ps1'
        with tempfile.TemporaryDirectory() as folder:
            for name in ('model-cache','alpr-cache'):
                (Path(folder)/(name+'.manifest.json')).write_text('{}')
            def quoted(value):return "'"+str(value).replace("'","''")+"'"
            code=("$global:taskCalls=@(); function docker { $global:taskCalls+=,@($args); $global:LASTEXITCODE=0 }; "
                  f"& {quoted(script)} -AssetsPath {quoted(folder)} -ModelsOnly; "
                  "if($global:taskCalls.Count -ne 4){throw 'Unexpected Docker call count'}; "
                  "$global:taskCalls | ConvertTo-Json -Depth 5 -Compress")
            result=subprocess.run(['pwsh','-NoProfile','-NonInteractive','-Command',code],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            calls=json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(sum(c[0]=='volume' for c in calls),2)
            runs=[c for c in calls if c[0]=='run']
            self.assertEqual(len(runs),2)
            for args in runs:
                self.assertIn('python:3.11-slim',args)
                self.assertFalse(any(x in args for x in ('compose','build','api','dashboard')))
            self.assertEqual({c[c.index('--manifest')+1] for c in runs},
                             {'/assets/model-cache.manifest.json','/assets/alpr-cache.manifest.json'})

    @unittest.skipUnless(shutil.which('pwsh'),'PowerShell unavailable')
    def test_models_only_and_skip_models_fail_before_docker(self):
        script=Path(__file__).resolve().parents[1]/'scripts/restore-teammate-handoff.ps1'
        with tempfile.TemporaryDirectory() as folder:
            def quoted(value):return "'"+str(value).replace("'","''")+"'"
            code=("$ErrorActionPreference='Stop'; function docker { throw 'Docker must not run' }; "
                  f"& {quoted(script)} -AssetsPath {quoted(folder)} -ModelsOnly -SkipModels")
            result=subprocess.run(['pwsh','-NoProfile','-NonInteractive','-Command',code],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('-ModelsOnly cannot be combined with -SkipModels',result.stderr)


if __name__=='__main__':unittest.main()
