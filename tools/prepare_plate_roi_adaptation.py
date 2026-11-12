"""Deterministic context-crop augmentation of an existing grouped plate export.

Publisher boxes are the only supervision. Source partitions/groups remain
unchanged; this module never opens the Mumbai recording or manual references.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_hash(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def absolute_boxes(boxes, shape):
    h, w = shape[:2]
    result=[]
    for value in boxes:
        if len(value) != 5 or any(not math.isfinite(float(v)) for v in value):
            raise ValueError('Finite publisher YOLO boxes required')
        cls,x,y,bw,bh=map(float,value)
        if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError('Single-class normalized plate labels required')
        a,b,c,d=(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h
        # Allow numerical roundoff at the normalized image edge only.
        if min(a,b) < -1e-5 or c > w+1e-5 or d > h+1e-5:
            raise ValueError('Publisher label outside source image')
        result.append([0,max(0.,a),max(0.,b),min(float(w),c),min(float(h),d)])
    if not result:
        raise ValueError('Unlabelled source is not a verified negative')
    return result


def map_boxes(boxes, crop, *, minimum_visible=.95):
    """Retain every intersecting annotation; signal unsafe partial cuts."""
    x1,y1,x2,y2=crop;w,h=x2-x1,y2-y1
    if w <= 0 or h <= 0:
        raise ValueError('Nonempty context crop required')
    result=[]
    for cls,a,b,c,d in boxes:
        left,top,right,bottom=max(a,x1),max(b,y1),min(c,x2),min(d,y2)
        if right <= left or bottom <= top:
            continue
        visible=(right-left)*(bottom-top)/((c-a)*(d-b))
        if visible+1e-9 < minimum_visible:
            return None
        result.append([int(cls),((left+right)/2-x1)/w,((top+bottom)/2-y1)/h,
                       (right-left)/w,(bottom-top)/h])
    return result


def context_crop(identifier, boxes, shape):
    """One seeded, plate-containing context; unsafe partial labels use full frame."""
    h,w=shape[:2]
    labels=absolute_boxes(boxes,shape)
    anchor=max(labels,key=lambda b:(b[3]-b[1])*(b[4]-b[2]))
    rng=random.Random(int(hashlib.sha256(('plate-roi-v1:'+identifier).encode()).hexdigest(),16))
    _,a,b,c,d=anchor
    fraction=rng.uniform(.2,.5);aspect=rng.uniform(1.,2.)
    cw=min(w,max(math.ceil((c-a)/fraction),math.ceil(c-a)+2))
    ch=min(h,max(math.ceil(cw/aspect),math.ceil(d-b)+2))
    # Plate tends to sit toward the lower part of a vehicle-like context.
    x=round((a+c)/2-cw*rng.uniform(.35,.65))
    y=round((b+d)/2-ch*rng.uniform(.55,.85))
    x=max(0,min(w-cw,x));y=max(0,min(h-ch,y))
    x=min(x,math.floor(a));x=max(x,math.ceil(c)-cw)
    y=min(y,math.floor(b));y=max(y,math.ceil(d)-ch)
    crop=(int(x),int(y),int(x+cw),int(y+ch))
    if not (0 <= crop[0] < crop[2] <= w and 0 <= crop[1] < crop[3] <= h):
        raise ValueError('Context bounds escaped source image')
    mapped=map_boxes(labels,crop)
    fallback=mapped is None or not mapped
    if fallback:
        crop=(0,0,w,h);mapped=map_boxes(labels,crop)
    return crop,mapped,dict(full_image_fallback=fallback,requested_plate_width_fraction=fraction,
                            requested_context_aspect=aspect)


def validate_lineage(rows):
    """Every source group and exact generated image belongs to one partition."""
    groups,pixels={},{}
    for row in rows:
        if row['split'] not in ('train','val','test') or not row.get('group'):
            raise ValueError('Frozen split and group required')
        for mapping,key in ((groups,row['group']),(pixels,row['pixels'])):
            previous=mapping.setdefault(key,row['split'])
            if previous != row['split']:
                raise ValueError('Generated group or pixel leakage across fixed partitions')


def prepare(source, output):
    import cv2
    import yaml
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Fresh, separate export directory required')
    source_path=source/'manifest.json'
    manifest=json.loads(source_path.read_bytes())
    if manifest.get('kind') != 'plate' or not manifest.get('samples'):
        raise ValueError('Existing publisher-labelled plate export required')
    source_rows=manifest['samples']
    validate_lineage(source_rows)
    output.mkdir(parents=True)
    rows=[];skips=Counter();seen={}
    for row in source_rows:
        image_path=(source/row['image_path']).resolve()
        if not image_path.is_relative_to(source) or sha(image_path) != row['original_sha256']:
            raise ValueError('Frozen publisher image changed')
        image=cv2.imread(str(image_path))
        if image is None or image.shape[:2] != (row['height'],row['width']):
            raise ValueError('Source image dimensions/decode mismatch')
        absolute_boxes(row['boxes'],image.shape)
        crop,mapped,details=context_crop(row['id'],row['boxes'],image.shape)
        x1,y1,x2,y2=crop
        variants=[('original',image,row['boxes'])]
        # Keep reserved test exactly paired with the existing test export.
        if row['split'] != 'test':
            if crop == (0,0,image.shape[1],image.shape[0]):
                skips['context_full_frame_duplicate']+=1
            else:
                variants.append(('context',image[y1:y2,x1:x2],mapped))
        for variant,working,boxes in variants:
            digest=pixel_hash(working)
            prior=seen.get(digest)
            if prior is not None:
                if prior['split'] != row['split']:
                    raise ValueError('Generated exact pixel duplicate crosses fixed partitions')
                if (len(prior['boxes']) != len(boxes) or any(
                        any(abs(float(x)-float(y))>1e-8 for x,y in zip(a,b))
                        for a,b in zip(prior['boxes'],boxes))):
                    raise ValueError('Generated duplicate has conflicting annotations')
                skips[variant+'_identical_pixels']+=1
                continue
            identifier=row['id'] if variant=='original' else row['id']+'-context'
            folder=output/'images'/row['split'];folder.mkdir(parents=True,exist_ok=True)
            suffix=image_path.suffix if variant=='original' else '.png'
            destination=folder/(identifier+suffix)
            if variant=='original':destination.write_bytes(image_path.read_bytes())
            elif not cv2.imwrite(str(destination),working):raise ValueError('Context image write failed')
            labels=output/'labels'/row['split'];labels.mkdir(parents=True,exist_ok=True)
            label_path=labels/(identifier+'.txt')
            label_path.write_text('\n'.join(' '.join(map(str,b)) for b in boxes)+'\n',encoding='utf-8')
            record=copy.deepcopy(row)
            record.update(id=identifier,parent_id=row['id'],variant=variant,
                          image_path=destination.relative_to(output).as_posix(),
                          original_sha256=sha(destination),pixels=digest,
                          label_sha256=sha(label_path),boxes=boxes,
                          width=working.shape[1],height=working.shape[0],
                          source_original_sha256=row['original_sha256'])
            if variant=='context':record.update(context_box=list(crop),context_policy=details)
            seen[digest]=record;rows.append(record)
    validate_lineage(rows)
    counts=dict(Counter(row['split'] for row in rows))
    if any(counts.get(s,0)<3 for s in ('train','val','test')):
        raise ValueError('Fixed partitions have insufficient samples')
    result={k:copy.deepcopy(v) for k,v in manifest.items() if k not in ('samples','counts')}
    result.update(samples=rows,counts=counts,source_manifest_sha256=sha(source_path),
                  source_sample_counts=manifest['counts'],policy='publisher-context-v1',
                  skipped=dict(skips),group_count=len({row['group'] for row in rows}),
                  labels_used='Publisher plate boxes only; no Mumbai images, boundaries or plate text',
                  context_splits=['train','val'],test_originals_only=True,
                  caveat=manifest['caveat']+'; context crops simulate vehicle ROIs, not ground-truth vehicle boxes; exact duplicates removed; source split/group lineage unchanged')
    (output/'data.yaml').write_text(yaml.safe_dump(dict(path=str(output),train='images/train',val='images/val',test='images/test',nc=1,names=['plate'])),encoding='utf-8')
    path=output/'manifest.json';path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    receipt=dict(policy='publisher-context-v1',source_manifest_sha256=sha(source_path),manifest_sha256=sha(path),
                 counts=counts,source_sample_counts=manifest['counts'],skipped=dict(skips),
                 seed='SHA256(plate-roi-v1:+source-id)',partition_groups_preserved=True,
                 test_originals_only=True,test_used_for_selection=False,script_sha256=sha(__file__))
    (output/'preparation-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output)),flush=True)
