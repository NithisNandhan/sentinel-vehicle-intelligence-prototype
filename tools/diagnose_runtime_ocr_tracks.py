"""Associate saved native crops to human labels by exact-PTS pixel matching.

Never match OCR strings to choose truth. Ambiguous, weak or conflicting pixel
associations are excluded. Detailed local output contains labels and readings.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from sentinel_edge.awiros_experiment import AwirosExperiment, file_hash
from sentinel_edge.ctc_fusion import CTCPlateReader, greedy_text


def run(args):
    import cv2
    import numpy as np
    if args.output.exists():
        raise ValueError('Preserve earlier runtime replay evidence')
    decode = lambda value:cv2.imdecode(np.frombuffer(base64.b64decode(value,validate=True),np.uint8),cv2.IMREAD_COLOR)
    manifest = json.loads((args.review/'manifest.json').read_text())
    labels = json.loads((args.review/'ocr-development-manifest.json').read_text())
    if labels['review_manifest_sha256'] != file_hash(args.review/'manifest.json'):
        raise ValueError('Human labels no longer match the frozen manifest')
    labels = {r['id']:r['transcription'] for r in labels['samples'] if r['id']!='21'}
    references = []
    for r in manifest['samples']:
        if r['id'] not in labels:
            continue
        path = args.review/r['crop']
        if file_hash(path) != r['crop_sha256']:
            raise ValueError('Reference pixels changed')
        references.append(dict(id=r['id'],pts_ms=r['position_ms'],image=cv2.imread(str(path))))
    associations, excluded = [], []
    for path in sorted(args.runtime.glob('track-*.json')):
        snapshot = json.loads(path.read_text())
        track = snapshot['track']
        matches = []
        for frame in track.get('frames',[]):
            native = decode(frame['image']) if frame.get('image') else None
            if native is None or min(native.shape[:2])<16 or native.shape[1]<50:
                continue
            scores = []
            for reference in references:
                if frame['pts_ms'] != reference['pts_ms']:
                    continue
                other = reference['image']
                if all(a<=b for a,b in zip(native.shape[:2],other.shape[:2])):
                    search,template = other,native
                elif all(a>=b for a,b in zip(native.shape[:2],other.shape[:2])):
                    search,template = native,other
                else:
                    continue
                coverage = template.shape[0]*template.shape[1]/max(native.shape[0]*native.shape[1],other.shape[0]*other.shape[1])
                if coverage<.2 or cv2.cvtColor(template,cv2.COLOR_BGR2GRAY).std()<8:
                    continue
                score = float(cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED).max())
                scores.append((score,reference['id'],coverage))
            scores.sort(reverse=True)
            if scores and scores[0][0]>=.995 and (len(scores)==1 or scores[0][0]-scores[1][0]>=.02):
                score,label_id,coverage = scores[0]
                matches.append(dict(manual_id=label_id,pts_ms=frame['pts_ms'],ncc=score,coverage=coverage))
        identities = {m['manual_id'] for m in matches}
        if len(identities)!=1:
            excluded.append(dict(track_id=track['track_id'],reason='No unique strong exact-PTS pixel match'))
            continue
        identity = next(iter(identities))
        associations.append(dict(track=track,manual_id=identity,expected=labels[identity],matches=matches))

    checkpoints = dict(baseline=args.cache/'adaptations/indian-corpus-encoder.safetensors')
    if (args.study/'candidate.safetensors').exists():
        checkpoints['candidate'] = args.study/'candidate.safetensors'
    results = []
    for name,checkpoint in checkpoints.items():
        reader = AwirosExperiment(args.cache,device='gpu',checkpoint=checkpoint)
        for association in associations:
            temporal = CTCPlateReader(reader.tokens)
            frames = []
            for frame in sorted(association['track']['frames'],key=lambda f:f['pts_ms']):
                if not frame.get('image'):
                    continue
                native = decode(frame['image'])
                native_rows = reader.probabilities(native)
                working = decode(frame['working_image']) if frame.get('working_view')=='text_quad' and frame.get('working_image') else None
                rows = reader.probabilities(working) if working is not None else native_rows
                if temporal.samples and frame['pts_ms']-temporal.samples[-1][0]>2500:
                    temporal = CTCPlateReader(reader.tokens)
                source = hashlib.sha256(native.tobytes()).hexdigest()
                temporal.add(rows,timestamp_ms=frame['pts_ms'],source_id=source,quality=frame.get('quality') or 1.)
                frames.append(dict(pts_ms=frame['pts_ms'],native_reading=greedy_text(native_rows,reader.tokens),
                    working_reading=greedy_text(rows,reader.tokens),rectified=working is not None))
            candidates = temporal.candidates()
            top = candidates[0].plate if candidates else None
            accepted = temporal.accepted()
            truth = association['expected']
            any_correct = any(f['working_reading']==truth for f in frames)
            result = dict(model=name,track_id=association['track']['track_id'],manual_id=association['manual_id'],
                expected=truth,pixel_associations=association['matches'],frames=frames,
                any_correct_native_frame=any(f['native_reading']==truth for f in frames),
                any_correct_working_frame=any_correct,fused_top=top,fused_exact=top==truth,
                fusion_lost_correct_frame=bool(any_correct and top!=truth),
                accepted=accepted.plate if accepted else None)
            results.append(result)
        del reader
    summary = {name:dict(associated_tracks=sum(r['model']==name for r in results),
        any_correct_working_frame=sum(r['model']==name and r['any_correct_working_frame'] for r in results),
        fused_exact=sum(r['model']==name and r['fused_exact'] for r in results),
        fusion_lost_correct_frame=sum(r['model']==name and r['fusion_lost_correct_frame'] for r in results)) for name in checkpoints}
    args.output.write_text(json.dumps(dict(summary=summary,results=results,excluded=excluded,
        association_rule='Exact source PTS, NCC >=0.995, coverage >=0.20, distinct-match margin >=0.02; no OCR labels used for association',
        scope='Selected retained runtime tracks, pixel-associated with existing human labels; not exhaustive video recall'),indent=2))
    print(json.dumps(dict(stage='runtime_fusion_replay',summary=summary,excluded_tracks=len(excluded))),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    run(parser.parse_args())
