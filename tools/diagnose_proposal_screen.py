"""Read-only diagnostic of saved plate/non-plate proposals and human references.

Uses deployed OCR and the existing small visual verifier. It does not train,
alter operational gates, infer truth from predictions, or modify frozen runs.
Known negative examples below are visual diagnostic findings, not independently
human-labelled evaluation truth. All plate confidence values are uncalibrated.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    import cv2
    import numpy as np
    from sentinel_edge.awiros_experiment import AwirosExperiment
    from sentinel_edge.ctc_fusion import greedy_reading
    from sentinel_edge.plate_filter import NonPlateFilter,confident_characters
    from sentinel_edge.plate_region_verifier import PlateRegionVerifier
    root=args.benchmarks
    manual=root/'recorded-footage-20261104/manual-review'
    legacy=root/'provisional-reading-review-20261112'
    capture=root/'full-video-passage-review-v2-20261113'
    manifest=json.loads((manual/'manifest.json').read_bytes())
    labels=json.loads((manual/'labels.json').read_bytes())
    refs=json.loads((legacy/'audit-view.json').read_bytes())
    live=json.loads((capture/'manifest.json').read_bytes())
    predictions=json.loads((capture/'predictions.json').read_bytes())
    if any(s!=live['source']['sha256'] for s in (manifest['source_sha256'],labels['source_sha256'],refs['source_sha256'])):
        raise ValueError('Diagnostic sources differ')
    if sha(capture/'manifest.json')!=predictions['manifest_sha256']:
        raise ValueError('Frozen predictions changed')
    truth={r['id']:r for r in labels['labels']}
    inputs=[]
    for r in manifest['samples']:
        path=manual/r['crop']
        if sha(path)!=r['crop_sha256']:raise ValueError('Manual crop changed')
        inputs.append(dict(id='manual-native-'+r['id'],path=str(path),role='human_plate_reference',
            view='original_manual_crop',human_status=truth[r['id']]['status'],
            human_plate=truth[r['id']].get('plate'),reference_id=r['id']))
    for r in refs['references']:
        inputs.append(dict(id='manual-rectified-'+r['reference_id'],path=str(legacy/r['reference_crop']),
            role='human_plate_reference',view='existing_reference_view',
            human_status='readable',human_plate=r['actual'],reference_id=r['reference_id']))
    negative_passages={'passage-s0-932bff85b41d0d99':'truck_windscreen',
        'passage-s0-e11527d6d47a1c2a':'truck_windscreen',
        'passage-s0-aefd631accb2b55e':'truck_windscreen'}
    for r in live['tracks']:
        for i,s in enumerate(r['samples']):
            negative=negative_passages.get(r['id'])
            if r['id']=='passage-s0-98c9c0f6c8e0ad96' and s['position_ms']==15000:
                negative='motorcycle_headlight_trim'
            inputs.append(dict(id=r['id']+'-'+str(i),path=str(capture/s['path']),
                role='visual_qa_negative' if negative else 'unlabelled_pipeline_proposal',
                visual_negative_kind=negative,passage_id=r['id'],tracker_ids=r['tracker_ids'],
                position_ms=s['position_ms'],model_accepted_control=bool(predictions['tracks'][r['id']]['accepted']),
                selected_crop=i==0 and bool(predictions['tracks'][r['id']]['top_guess'])))
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    checkpoint=args.cache/'adaptations/indian-corpus-encoder.safetensors'
    ocr=AwirosExperiment(args.cache,device=args.device,checkpoint=checkpoint)
    verifier=PlateRegionVerifier(args.verifier)
    screen=NonPlateFilter()
    rows=[]
    for i,item in enumerate(inputs):
        image=cv2.imread(item['path'])
        if image is None:raise ValueError('Diagnostic crop cannot decode')
        posterior=ocr.probabilities(image)
        reading=greedy_reading(posterior,ocr.tokens)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        height,width=image.shape[:2]
        visual=verifier.verify(image)
        row=dict(item,width=width,height=height,aspect=width/height,pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            file_sha256=sha(item['path']),reading=''.join(c for c,p in reading),characters=reading,
            character_count=len(reading),confident_count_06=confident_characters(posterior,.6),
            confident_count_09=sum(p>=.9 for _,p in reading),
            mean_score=sum(p for _,p in reading)/len(reading) if reading else 0.,
            min_score=min((p for _,p in reading),default=0.),
            current_filter_reason_without_detector_score=screen.reason(image,posterior,None),
            visual_verifier=visual,gray_std=float(gray.std()),
            edge_fraction=float(np.mean(cv2.Canny(gray,50,150)>0)),
            detector_score_available=False)
        rows.append(row)
        if (i+1)%20==0:print(json.dumps(dict(processed=i+1,total=len(inputs))),flush=True)
    groups={name:[r for r in rows if r['role']==name] for name in {r['role'] for r in rows}}
    summary={name:dict(count=len(values),current_filter_rejects=sum(bool(r['current_filter_reason_without_detector_score']) for r in values),
        visual_verifier_states=dict(Counter(r['visual_verifier']['status'] for r in values))) for name,values in groups.items()}
    report=dict(scope='Saved-crop diagnostic; development controls only, no new physical inventory or full-video accuracy',
        models_changed=False,operational_settings_changed=False,central_events_sent=0,device=args.device,
        source_sha256=live['source']['sha256'],capture_manifest_sha256=sha(capture/'manifest.json'),
        manual_manifest_sha256=sha(manual/'manifest.json'),manual_labels_sha256=sha(manual/'labels.json'),
        ocr_model_sha256=sha(checkpoint),visual_verifier_sha256=verifier.model_sha256,
        diagnostic_source_sha256=sha(__file__),elapsed_seconds=round(time.monotonic()-started,2),
        notice='Visual-QA negatives are not independently human-labelled. Detector confidence was not saved and is treated as unknown, never guessed. Manual original/reference views overlap in physical identity.',
        summary=summary,rows=rows)
    (args.output/'proposal-screen-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,elapsed_seconds=report['elapsed_seconds'],output=str(args.output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmarks',type=Path,default=Path('/benchmarks'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    p.add_argument('--verifier',type=Path,default=Path('/workspace/edge/assets/plate-region-verifier/model.npz'))
    p.add_argument('--device',choices=['cpu','gpu'],default='gpu')
    main(p.parse_args())
