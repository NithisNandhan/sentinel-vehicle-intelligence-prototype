"""Bounded primary-source Indian CCTV subset, pinned to a public revision.

The complete BMD-45 repository exceeds 160 GB. Never silently download it all.
Publisher train/val separation is retained; validation/test partitioning is
deterministic but not a verified camera-identity split.
"""
import hashlib
import json
import math
import shutil
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

REF = 'iisc-aim/BMD-45'
ROOT = Path('/training/vehicle-bmd')
LIMIT = 6*1024**3
lock = threading.Lock()
transferred = 0


def get(url,limit):
    with urllib.request.urlopen(url,timeout=90) as response:
        if 'text/html' in response.headers.get('Content-Type','').lower():
            raise ValueError('HTML is not dataset content')
        data = response.read(limit+1)
    if len(data)>limit:
        raise ValueError('Dataset file exceeds size bound')
    return data


def main():
    global transferred
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'manifest.json').exists():
        raise ValueError('Prepared subset preserved; do not overwrite a frozen training manifest')
    meta = json.loads(get('https://huggingface.co/api/datasets/'+REF,16*1024**2))
    if meta.get('gated') or meta.get('cardData',{}).get('license') != 'cc-by-4.0':
        raise ValueError('Public CC BY dataset required')
    prior_path = ROOT/'source-metadata.json'
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        # Resume the original revision, never silently switch downloaded files.
        if prior['sha'] != meta['sha']:
            meta = json.loads(get('https://huggingface.co/api/datasets/'+REF+'/revision/'+prior['sha'],16*1024**2))
    revision = meta['sha']; base = f'https://huggingface.co/datasets/{REF}/resolve/{revision}/'
    (ROOT/'source-metadata.json').write_text(json.dumps(meta,indent=2))
    paths = [s['rfilename'] for s in meta['siblings']]
    rows,categories = [],None
    for source_partition in ('Train','Val'):
        prefix = 'BMD-45-'+source_partition+'/'
        payload = get(base+prefix+'_annotations.coco.json',160*1024**2)
        (ROOT/('annotations-'+source_partition+'.json')).write_bytes(payload)
        data = json.loads(payload)
        if categories is not None and categories != data['categories']:
            raise ValueError('Category taxonomy differs between partitions')
        categories = data['categories']
        annotations = {}
        for annotation in data['annotations']:
            if annotation.get('iscrowd',0):
                continue
            annotations.setdefault(annotation['image_id'],[]).append(annotation)
        images = sorted((i for i in data['images'] if annotations.get(i['id'])),
                        key=lambda i:hashlib.sha256(str(i['id']).encode()).hexdigest())
        images = images[:1000 if source_partition=='Train' else 300]
        for index,im in enumerate(images):
            filename = im['file_name']
            expected = prefix+filename
            if expected not in paths:
                matches = [p for p in paths if p.startswith(prefix) and Path(p).name == Path(filename).name]
                if len(matches)!=1:
                    raise ValueError('Ambiguous publisher image path')
                expected = matches[0]
            split = 'train' if source_partition=='Train' else 'val' if index<150 else 'test'
            rows.append({'id':'bmd-'+source_partition.lower()+'-'+str(im['id']),'source_partition':source_partition,'split':split,
                'publisher_path':expected,'width':im['width'],'height':im['height'],
                'annotations':annotations[im['id']]})
    names = {c['id']:c['name'] for c in categories}
    expected_names = ['Hatchback','Sedan','SUV','MUV','Bus','Truck','Three-wheeler',
                      'Two-wheeler','LCV','Mini-bus','Tempo-traveller','Bicycle','Van']
    if (sorted(names)!=list(range(len(names))) or
            list(names.values()) not in (expected_names,expected_names+['Other'])):
        raise ValueError('Unexpected category IDs; do not guess mapping')
    def fetch(row):
        global transferred
        path = ROOT/'images'/row['split']/(row['id']+'.png')
        receipt = path.with_suffix('.receipt.json')
        if path.exists() and receipt.exists():
            raw = path.read_bytes(); prior=json.loads(receipt.read_text())
            if prior.get('revision') != revision or hashlib.sha256(raw).hexdigest()!=prior['sha256']:
                raise ValueError('Cached image receipt mismatch')
        else:
            if shutil.disk_usage(ROOT).free < 6*1024**3:
                raise ValueError('Disk headroom reached; preserve downloads')
            raw = get(base+row['publisher_path'],12*1024**2)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(raw)
            receipt.write_text(json.dumps({'revision':revision,'sha256':hashlib.sha256(raw).hexdigest(),
                'source_path':row['publisher_path'],'publisher_checksum_verified':False}))
        with lock:
            transferred += len(raw)
            if transferred > LIMIT:
                raise ValueError('Total subset byte limit reached')
        image = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (row['height'],row['width']):
            # Scaling annotations would assume an unverified image transform.
            # Preserve original bytes, but exclude from every learning split.
            return {'id':row['id'],'excluded':True,'reason':'publisher_dimension_mismatch',
                    'declared_shape':[row['height'],row['width']],
                    'decoded_shape':list(image.shape[:2]) if image is not None else None,
                    'publisher_path':row['publisher_path']}
        boxes=[]
        for annotation in row.pop('annotations'):
            x,y,w,h = annotation['bbox']; cls=annotation['category_id']
            if (not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (x,y,w,h))
                    or w<=0 or h<=0):
                raise ValueError('Invalid publisher geometry')
            x1,y1,x2,y2=max(0,x),max(0,y),min(row['width'],x+w),min(row['height'],y+h)
            if cls not in names or x2<=x1 or y2<=y1:
                raise ValueError('Invalid publisher box')
            boxes.append([cls,(x1+x2)/2/row['width'],(y1+y2)/2/row['height'],
                          (x2-x1)/row['width'],(y2-y1)/row['height']])
        label=ROOT/'labels'/row['split']/(row['id']+'.txt'); label.parent.mkdir(parents=True,exist_ok=True)
        label.write_text('\n'.join(' '.join(map(str,b)) for b in boxes)+'\n')
        row.update(boxes=boxes,image_path=str(path.relative_to(ROOT)),
            original_sha256=hashlib.sha256(raw).hexdigest(),pixels=hashlib.sha256(image.tobytes()).hexdigest())
        return row
    completed=[]; exclusions=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(fetch,rows):
            if row.get('excluded'):
                exclusions.append(row)
                continue
            completed.append(row)
            if len(completed)%25==0:
                print(json.dumps({'acquired_images':len(completed),'mib':round(transferred/1024**2,1)}),flush=True)
    by_pixels={}
    for row in completed:
        by_pixels.setdefault(row['pixels'],set()).add(row['split'])
    leaking={p for p,splits in by_pixels.items() if len(splits)>1}
    if leaking:
        for row in completed:
            if row['pixels'] in leaking:
                exclusions.append({'id':row['id'],'reason':'cross_split_exact_duplicate'})
        completed = [row for row in completed if row['pixels'] not in leaking]
    # Explicit file lists are essential: excluded images remain on disk, so
    # directory-based dataset discovery must not accidentally re-admit them.
    for split in ('train','val','test'):
        selected = [str(ROOT/r['image_path']) for r in completed if r['split']==split]
        if len(selected)<3:
            raise ValueError('Too few admitted images after quality gates')
        (ROOT/(split+'.txt')).write_text('\n'.join(selected)+'\n')
    from collections import Counter
    manifest={'kind':'vehicle','source':REF,'revision':revision,'license':'CC-BY-4.0',
        'samples':completed,'counts':dict(Counter(r['split'] for r in completed)), 'category_names':names,
        'exclusions':exclusions,'attempted_images':len(rows),
        'independent_camera_holdout':False,'bytes':transferred,
        'caveat':'Bounded source train/val subset; validation partition split into val/test before training; exact pixel leakage checked; camera identity and near-duplicate leakage not independently verified; source consensus labels not new human truth'}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (ROOT/'data.yaml').write_text(yaml.safe_dump({'path':str(ROOT),'train':'train.txt',
        'val':'val.txt','test':'test.txt','nc':len(names),'names':names}))
    print(json.dumps({k:v for k,v in manifest.items() if k!='samples'}),flush=True)


if __name__ == '__main__':main()
