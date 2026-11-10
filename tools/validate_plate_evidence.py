"""Triage a completed vehicle-lifecycle capture without model inference or labels.

Uses exactly the retained native crops and original CTC posteriors. The source
capture stays immutable and all source IDs/hypotheses are preserved in a fresh
folder. Run the separate scorer afterwards to measure recognition regression.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

from run_owned_multiframe import build_vehicle_results,sha,write_json
from sentinel_edge.plate_evidence import annotate_plate_evidence


def validate(capture,output):
    import cv2
    capture,output=Path(capture).resolve(),Path(output).resolve()
    if output.exists() or output.is_relative_to(capture):
        raise ValueError('Use a fresh output outside the immutable source capture')
    began=time.monotonic()
    helper=Path(__file__).resolve().parents[1]/'edge/sentinel_edge/plate_evidence.py'
    source_paths=[Path(__file__).resolve(),helper,helper.with_name('plate_filter.py'),
                  helper.with_name('vehicle_plate.py'),helper.with_name('ctc_fusion.py'),
                  Path(__file__).with_name('run_owned_multiframe.py')]
    source_hashes={str(p):sha(p) for p in source_paths}
    document=json.loads((capture/'predictions.json').read_bytes())
    if not document.get('vehicle_lifecycle') or document.get('plate_validation'):
        raise ValueError('An unvalidated vehicle-lifecycle capture is required')
    if not document.get('reached_eof') or document.get('labels_used_during_inference'):
        raise ValueError('Completed label-free inference capture required')
    posteriors=json.loads(gzip.decompress((capture/'posteriors.json.gz').read_bytes()))
    timeline=json.loads((capture/'vehicle-timeline.json').read_bytes())
    groups={g['id']:g for g in document['groups']}
    if len(groups)!=len(document['groups']):raise ValueError('Duplicate plate identity')
    for group in posteriors['groups']:
        if group['id'] not in groups:raise ValueError('Unknown posterior identity')
        public=groups[group['id']]['samples']
        samples=[{k:v for k,v in o.items() if k!='rows'} for o in group['observations']]
        if samples!=public:raise ValueError('Saved original posterior/sample binding changed')
    print(json.dumps(dict(stage='checking_frozen_parent_replay')),flush=True)
    baseline=build_vehicle_results(document['groups'],posteriors['groups'],posteriors['tokens'])
    if baseline!=document['vehicle_results']:
        raise ValueError('Parent fusion replay differs from frozen baseline; stop before changing evidence')

    def load_image(sample):
        relative=Path(sample['path'])
        path=(capture/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to((capture/'assets').resolve()):
            raise ValueError('Native crop path escapes saved assets')
        image=cv2.imread(str(path))
        if image is None or hashlib.sha256(image.tobytes()).hexdigest()!=sample['source_id']:
            raise ValueError('Saved native crop source identity changed')
        return image

    print(json.dumps(dict(stage='validating_native_crops')),flush=True)
    summary=annotate_plate_evidence(document['groups'],posteriors['groups'],posteriors['tokens'],timeline,load_image)
    document['plate_validation']=summary
    print(json.dumps(dict(stage='rebuilding_parent_readings',counts=summary['history_counts'])),flush=True)
    document['vehicle_results']=build_vehicle_results(document['groups'],posteriors['groups'],posteriors['tokens'])
    if source_hashes!={str(p):sha(p) for p in source_paths}:
        raise ValueError('Validation sources changed while running; preserve the source and retry in a fresh folder')
    document['plate_validation_capture']=dict(input_predictions_sha256=sha(capture/'predictions.json'),
        input_posteriors_sha256=sha(capture/'posteriors.json.gz'),input_vehicle_timeline_sha256=sha(capture/'vehicle-timeline.json'),
        runner_sha256=sha(__file__),validator_sha256=sha(helper),fusion_runner_sha256=sha(Path(__file__).with_name('run_owned_multiframe.py')),
        postprocessing_source_sha256={p.name:source_hashes[str(p)] for p in source_paths},
        baseline_parent_replay_identical=True,models_rerun=False,labels_used=False)
    document['postprocess_seconds']=round(time.monotonic()-began,3)
    document['scope']='Post-processing of all retained crops from a completed source; same model outputs, no training or events; selected reference evaluation follows separately'
    output.mkdir(parents=True)
    shutil.copytree(capture/'assets',output/'assets')
    for name in ('vehicle-timeline.json','posteriors.json.gz'):
        shutil.copyfile(capture/name,output/name)
    (output/'source-plate_evidence.py').write_bytes(helper.read_bytes())
    (output/'validation-runner.py').write_bytes(Path(__file__).read_bytes())
    write_json(output/'predictions.json',document)
    write_json(output/'validation-summary.json',dict(**summary,**document['plate_validation_capture'],
        postprocess_seconds=document['postprocess_seconds']))
    print(json.dumps(dict(stage='plate_evidence_validated',**summary,postprocess_seconds=document['postprocess_seconds'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    validate(args.capture,args.output)
