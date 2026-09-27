"""Prepare licence-audited detector data with grouped, leakage-checked splits.

Only publisher bounding boxes are training labels. OCR crops or the current
detector's predictions must never be substituted for detection ground truth.
"""
import hashlib
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np
import yaml


def checked_zip(path):
    receipt = json.loads(path.with_suffix('.receipt.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
        raise ValueError('Archive receipt mismatch')
    archive = zipfile.ZipFile(path)
    if sum(i.file_size for i in archive.infolist()) > 3*1024**3:
        raise ValueError('Archive expansion exceeds bound')
    for item in archive.infolist():
        if '..' in Path(item.filename).parts or Path(item.filename).is_absolute() or item.file_size > 64*1024**2:
            raise ValueError('Unsafe archive member')
    return archive


def parse_yolo(text, classes):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError('Only axis-aligned YOLO boxes allowed')
        cls, x, y, w, h = map(float, parts)
        if (not all(math.isfinite(v) for v in (cls,x,y,w,h)) or cls != int(cls)
                or not 0 <= cls < classes or not 0 <= x <= 1 or not 0 <= y <= 1
                or not 0 < w <= 1 or not 0 < h <= 1):
            raise ValueError('Invalid publisher annotation')
        x1,y1,x2,y2 = max(0,x-w/2),max(0,y-h/2),min(1,x+w/2),min(1,y+h/2)
        rows.append([int(cls),(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1])
    if not rows:
        raise ValueError('Unlabelled image is not a verified negative')
    return rows


def perceptual_hash(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    values = cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8].flatten()[1:]
    return sum(int(v > np.median(values)) << i for i,v in enumerate(values))


def exclude_cross_split_similar(rows, max_distance=6):
    """Quarantine entire connected similarity groups across fixed splits.

    Does not move a reserved test sample into training or choose by model output.
    pHash is a conservative heuristic, not proof of camera or vehicle identity.
    """
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    for i,a in enumerate(rows):
        for j,b in enumerate(rows[:i]):
            if (a['pixels']==b['pixels'] or
                    a.get('source')==b.get('source') and
                    (a['phash']^b['phash']).bit_count()<=max_distance):
                x,y=find(i),find(j); parent[max(x,y)]=min(x,y)
    groups={}
    for i,row in enumerate(rows):
        groups.setdefault(find(i),[]).append(row)
    kept=[]; excluded=[]
    for group in groups.values():
        if len({r['split'] for r in group})>1:
            excluded.extend(group)
        else:
            kept.extend(group)
    return kept,excluded


def grouping_key(source, member):
    stem = Path(member).stem
    if source == 'madrid':
        return source+':'+stem.split('_jpg.rf.')[0]
    if source == 'gujarat':
        match = re.search(r'IMG_(\d{8})_(\d{2})(\d{2})(\d{2})',stem)
        if match:
            day,h,m,s = match.groups()
            # Group nearby shots conservatively; image similarity unions below
            # can also connect groups across these five-minute boundaries.
            return f'{source}:{day}:{(int(h)*3600+int(m)*60+int(s))//300}'
    return source+':'+re.sub(r'\s*\(\d+\)$','',stem)


def partition_groups(rows, preserve=False):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a,b):
        a,b = find(a),find(b)
        parent[max(a,b)] = min(a,b)
    for i,a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            if (a['pixels'] == b['pixels'] or a['group_key'] == b['group_key']
                    or a['source'] == b['source'] and (a['phash'] ^ b['phash']).bit_count() <= 6):
                union(i,j)
    groups = {}
    for i,row in enumerate(rows):
        groups.setdefault(find(i),[]).append(row)
    for group in groups.values():
        signature = hashlib.sha256('|'.join(sorted(r['group_key'] for r in group)).encode()).hexdigest()
        if preserve:
            rank = {'train':0,'val':1,'test':2}
            split = max((r['publisher_split'] for r in group),key=rank.get)
        else:
            value = int(signature[:8],16)%100
            split = 'test' if value < 15 else 'val' if value < 30 else 'train'
        for row in group:
            row['split'] = split; row['group'] = signature
    return rows


def read_sources(root,kind):
    rows,rejected = [],[]
    archives = ['gujarat.zip','make-ml.zip'] if kind == 'plate' else [
        'madrid-training.zip','madrid-valid-fine-tuning.zip','madrid-valid-self-learning.zip']
    for filename in archives:
        source = filename.split('-')[0] if filename.startswith('madrid') else filename[:-4]
        with checked_zip(root/'archives'/filename) as archive:
            names = set(archive.namelist())
            for member in sorted(names):
                if Path(member).suffix.lower() not in ('.jpg','.jpeg','.png'):
                    continue
                try:
                    raw = archive.read(member)
                    image = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                    if image is None or max(image.shape[:2]) > 12000:
                        raise ValueError('Image decode or dimensions')
                    if source == 'make-ml':
                        xml = ElementTree.fromstring(archive.read('annotations/'+Path(member).stem+'.xml'))
                        width,height = image.shape[1],image.shape[0]
                        if int(xml.findtext('size/width')) != width or int(xml.findtext('size/height')) != height:
                            raise ValueError('VOC dimension mismatch')
                        lines = []
                        for obj in xml.findall('object'):
                            if obj.findtext('name') != 'licence':
                                raise ValueError('Unexpected VOC class')
                            x1,y1,x2,y2 = [float(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
                            lines.append(f'0 {(x1+x2)/2/width} {(y1+y2)/2/height} {(x2-x1)/width} {(y2-y1)/height}')
                        boxes = parse_yolo('\n'.join(lines),1)
                    else:
                        label = str(Path(member.replace('/images/','/labels/')).with_suffix('.txt'))
                        boxes = parse_yolo(archive.read(label).decode(),1 if kind == 'plate' else 5)
                    publisher_split = ('test' if 'valid-self-learning' in filename else
                                       'val' if 'valid-fine-tuning' in filename else 'train')
                    rows.append({'id':hashlib.sha256((source+':'+member).encode()).hexdigest(),
                        'source':source,'member':member,'original_sha256':hashlib.sha256(raw).hexdigest(),
                        'pixels':hashlib.sha256(image.tobytes()).hexdigest(),'phash':perceptual_hash(image),
                        'group_key':grouping_key(source,member),'publisher_split':publisher_split,
                        'width':image.shape[1],'height':image.shape[0],'boxes':boxes,'raw':raw})
                except (ValueError,KeyError,ElementTree.ParseError) as error:
                    rejected.append({'source':source,'member':member,'reason':str(error)})
    return rows,rejected


def prepare(root,kind,output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Existing export preserved; use a new directory for changed data')
    rows,rejected = read_sources(root,kind)
    # Exact pixel duplicates with conflicting boxes are all quarantined.
    original_count = len(rows)
    unique,conflicts = {},set()
    for row in rows:
        previous = unique.get(row['pixels'])
        if previous and previous['boxes'] != row['boxes']:
            conflicts.add(row['pixels'])
        if previous and kind == 'vehicle':
            rank = {'train':0,'val':1,'test':2}
            previous['publisher_split'] = max(previous['publisher_split'],row['publisher_split'],key=rank.get)
        unique.setdefault(row['pixels'],row)
    rows = [r for r in unique.values() if r['pixels'] not in conflicts]
    rows = partition_groups(rows,preserve=kind=='vehicle')
    counts = Counter(r['split'] for r in rows)
    if any(counts[s]<3 for s in ('train','val','test')):
        raise ValueError('Insufficient disjoint partitions; do not weaken splits')
    output.mkdir(parents=True,exist_ok=True)
    # Plate label = 0. Vehicle publisher indices are retained unchanged.
    for row in rows:
        images = output/'images'/row['split']; labels = output/'labels'/row['split']
        images.mkdir(parents=True,exist_ok=True); labels.mkdir(parents=True,exist_ok=True)
        suffix = Path(row['member']).suffix.lower()
        (images/(row['id']+suffix)).write_bytes(row.pop('raw'))
        (labels/(row['id']+'.txt')).write_text('\n'.join(' '.join(map(str,b)) for b in row['boxes'])+'\n')
        row['image_path'] = str(Path('images')/row['split']/(row['id']+suffix))
    # The manual vehicle ZIPs lack a names file. Retain numeric categories;
    # class-agnostic evaluation avoids claiming an unverified semantic map.
    names = ['plate'] if kind=='plate' else [f'publisher_vehicle_{i}' for i in range(5)]
    (output/'data.yaml').write_text(yaml.safe_dump({'path':str(output),'train':'images/train',
        'val':'images/val','test':'images/test','nc':len(names),'names':names}))
    manifest = {'kind':kind,'samples':rows,'counts':dict(counts),'rejected':rejected,
        'exact_duplicate_rows_removed':original_count-len(unique),'conflicting_pixel_groups':len(conflicts),
        'group_count':len({r['group'] for r in rows}),'independent_camera_holdout':False,
        'label_review':'publisher labels; not independently human audited',
        'split_method':'exact pixel/filename/near-image unions; five-minute Gujarat groups; Madrid publisher partitions',
        'caveat':'Similar frames grouped; vehicle identity and distant temporal leakage not exhaustively verified; source pretraining overlap unknown'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('samples','rejected')}),flush=True)
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['plate','vehicle'],required=True)
    parser.add_argument('--root',type=Path,default=Path('/workspace/data/benchmarks/detector-training'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    prepare(args.root,args.kind,args.output)
