"""Pre-training near-image leakage guard for the bounded public BMD subset."""
import hashlib
import json
from pathlib import Path
from collections import Counter

import cv2

from sentinel_edge.detector_training_data import exclude_cross_split_similar, perceptual_hash


def main():
    root=Path('/training/vehicle-bmd')
    if Path('/training/runs/vehicle-bmd').exists():
        raise ValueError('Training already started; frozen manifest must remain unchanged')
    manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('source')!='iisc-aim/BMD-45' or manifest.get('license')!='CC-BY-4.0':
        raise ValueError('Audited public BMD manifest required')
    if manifest.get('near_image_guard_completed'):
        print(json.dumps({'status':'guard_already_completed','counts':manifest['counts']}),flush=True)
        return
    manifest['pre_guard_manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for index,row in enumerate(manifest['samples']):
        path=root/row['image_path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['original_sha256']:
            raise ValueError('Original public image changed')
        row['phash']=perceptual_hash(cv2.imread(str(path)))
        row['source']='BMD-45'
        if (index+1)%250==0:
            print(json.dumps({'hashed_images':index+1}),flush=True)
    kept,excluded=exclude_cross_split_similar(manifest['samples'])
    manifest['samples']=kept
    manifest['exclusions'].extend({'id':r['id'],'reason':'cross_split_near_image_group'} for r in excluded)
    manifest['counts']=dict(Counter(r['split'] for r in kept))
    if any(manifest['counts'].get(s,0)<3 for s in ('train','val','test')):
        raise ValueError('Too few disjoint samples; original manifest preserved')
    manifest['near_image_guard_completed']=True
    manifest['near_image_guard_hamming_distance']=6
    manifest['caveat']='Bounded publisher train/val subset; val/test chosen before training; exact pixels and transitive pHash similarity groups guarded across splits; camera and vehicle identities and pretraining overlap not independently verified; source consensus labels not new human truth'
    for split in ('train','val','test'):
        (root/(split+'.txt')).write_text('\n'.join(str(root/r['image_path']) for r in kept if r['split']==split)+'\n')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'counts':manifest['counts'],'near_image_exclusions':len(excluded),
                      'total_exclusions':len(manifest['exclusions'])}),flush=True)


if __name__=='__main__':main()
