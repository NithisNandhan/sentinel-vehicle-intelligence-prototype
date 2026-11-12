"""Read frozen uncertain crops and check registration resemblance, without new detection.

No labels, manual plate boxes, model tuning, central events or input edits.
The original strict-verifier capture and all of its images stay unchanged.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'edge'))
from render_text_corner_audit import validate_audit
from sentinel_edge.provisional_plate_reading import POLICY,check_reading,ocr_allowed,unconstrained_top


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    import cv2
    from sentinel_edge.awiros_experiment import AwirosExperiment
    from sentinel_edge.ctc_fusion import CTCPlateReader
    source=args.capture.resolve();output=args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'data/benchmarks').resolve()) or output.is_relative_to(source):
        raise ValueError('Fresh, separate local benchmark output required')
    original,_=validate_audit(source)
    if original.get('provisional_ocr_policy'):
        raise ValueError('Replay requires an original strict-verifier capture')
    checkpoint=args.cache/'adaptations/indian-corpus-encoder.safetensors'
    model_hash=sha(checkpoint)
    if model_hash!='c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d' or original['ocr_model_sha256']!=model_hash:
        raise ValueError('Frozen OCR model identity changed')
    output.mkdir(parents=True)
    (output/'progress.json').write_text(json.dumps({'stage':'loading_ocr'}))
    ocr=AwirosExperiment(args.cache,device=args.device,checkpoint=checkpoint)
    document=copy.deepcopy(original)
    shutil.copytree(source/'assets',output/'assets')
    # Old outputs were format-filtered. Reuse the images, not those hypotheses,
    # so every resemblance check starts with an unfiltered OCR top.
    cache={}
    counts=Counter();began=time.monotonic()
    for frame in document['frames']:
        for region in frame['regions']:
            for proposal in region['proposals']:
                verdict=proposal['verdict']
                counts[verdict['status']]+=1
                proposal['previous_ranked_top']=proposal.get('top')
                if ocr_allowed(verdict,allow_uncertain=True):
                    key=proposal['rectified_sha256']
                    if key not in cache:
                        image=cv2.imread(str(source/proposal['rectified_path']))
                        if image is None:raise ValueError('Validated OCR working copy could not decode')
                        cache[key]=unconstrained_top(ocr.probabilities(image),ocr.tokens)
                        counts['new_ocr_calls']+=1
                    else:counts['reused_identical_image_outputs']+=1
                    proposal.update(ocr_ran=True,top=cache[key])
                    if verdict['status']=='uncertain':counts['uncertain_regions_read']+=1
                    proposal['ocr_decoder']='unconstrained_ctc_prefix_top'
                proposal['reading_check']=check_reading(verdict,proposal.get('top'))
                if proposal['reading_check']['display_plate']:
                    counts['provisional_results']+=1
                    if verdict['status']=='uncertain':counts['uncertain_provisional_results']+=1
                counts['ocr_outputs']+=int(proposal['ocr_ran'])
                if counts['new_ocr_calls'] and counts['new_ocr_calls']%20==0:
                    (output/'progress.json').write_text(json.dumps(dict(stage='reading_uncertain',summary=dict(counts))))
        print(json.dumps(dict(stage='frame_replayed',position_ms=frame['position_ms'],counts=dict(counts))),flush=True)
    document.update(provisional_ocr_policy=POLICY,source_audit_sha256=sha(source/'audit.json'),
        source_summary=original['summary'],summary=dict(counts),elapsed_seconds=round(time.monotonic()-began,2),
        scope='OCR replay on frozen timestamp-selected crops; uncertain regions read provisionally; no new detection or reconstruction benchmark',
        inference_device=args.device,labels_used_during_inference=False,central_events_sent=0,
        replay_source_sha256=sha(Path(__file__)),reading_policy_source_sha256=sha(ROOT/'edge/sentinel_edge/provisional_plate_reading.py'),
        model_weights_changed=False)
    (output/'audit.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    validate_audit(output)
    (output/'progress.json').write_text(json.dumps(dict(stage='complete',summary=dict(counts))))
    print(json.dumps(dict(stage='complete',elapsed_seconds=document['elapsed_seconds'],counts=dict(counts))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    parser.add_argument('--device',choices=('cpu','gpu'),default='cpu')
    run(parser.parse_args())
