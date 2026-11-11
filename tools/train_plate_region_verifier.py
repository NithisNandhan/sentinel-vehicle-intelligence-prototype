"""Reproducible small visual verifier from frozen, publisher-labelled scenes.

Mumbai evaluation footage and transcriptions are never opened by this trainer.
Background negatives are annotation-derived assumptions, not human verification.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'edge'))
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

from sentinel_edge.plate_region_verifier import FEATURE_DIM, FEATURE_VERSION, PlateRegionVerifier, decision, extract_features, sha256

MUMBAI_HASH = '6b7064605b3d4d12c6c1da4566cbcec1cd2e3e3bd7b77814865fe9df22045f65'


def validate_splits(rows):
    owners = {'group': {}, 'pixels': {}}
    counts = Counter()
    for row in rows:
        split = row['split']
        if split not in ('train', 'val', 'test'):
            raise ValueError('Only frozen train/val/test rows allowed')
        if row.get('original_sha256') == MUMBAI_HASH:
            raise ValueError('Mumbai evaluation source must remain excluded')
        for key, seen in owners.items():
            value = row[key]
            if value in seen and seen[value] != split:
                raise ValueError('Cross-split '+key+' leakage')
            seen[value] = split
        counts[split] += 1
    return dict(counts)


def positive_boxes(row):
    w, h = row['width'], row['height']
    result = []
    for _, cx, cy, bw, bh in row['boxes']:
        box = (max(0, int((cx-bw/2)*w)), max(0, int((cy-bh/2)*h)),
               min(w, int(np.ceil((cx+bw/2)*w))), min(h, int(np.ceil((cy+bh/2)*h))))
        if min(box[2]-box[0], box[3]-box[1]) >= 4:
            result.append(box)
    return result


def overlaps_any(box, positives, margin=3):
    return any(min(box[2], b[2]+margin) > max(box[0], b[0]-margin)
               and min(box[3], b[3]+margin) > max(box[1], b[1]-margin) for b in positives)


def negatives(row, positives):
    rng = np.random.default_rng(int(row['id'][:12], 16))
    w, h = row['width'], row['height']
    candidates = []
    # Adjacent same-size vehicle context provides harder negatives than random sky.
    for x, y, x2, y2 in positives:
        bw, bh = x2-x, y2-y
        for ox, oy in ((-1.4,0),(1.4,0),(0,-1.6),(0,1.6),(-.7,-2.5),(.7,2.5)):
            left, top = round(x+ox*bw), round(y+oy*bh)
            candidates.append((left,top,left+bw,top+bh))
    for _ in range(80):
        bw = int(rng.uniform(.09,.35)*w)
        bh = max(8, round(bw/rng.uniform(1.4,5.5)))
        x, y = int(rng.integers(0,max(1,w-bw+1))), int(rng.integers(0,max(1,h-bh+1)))
        candidates.append((x,y,x+bw,y+bh))
    result = []
    for box in candidates:
        if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h) or min(box[2]-box[0],box[3]-box[1]) < 8:
            continue
        if overlaps_any(box, positives) or box in result:
            continue
        result.append(box)
        if len(result) >= 6:
            break
    return result


def threshold_selection(scores, truth):
    """Validation only: retain >=99% of positives; acceptance aims >=95% precision."""
    positive = np.sort(scores[truth == 1])
    if len(positive) < 20:
        raise ValueError('At least twenty validation plates needed for threshold selection')
    # At most floor(1%) validation positives are rejected. Strict-less rejection.
    index = int(np.floor(.01*len(positive)))
    reject = max(0., float(np.nextafter(positive[index], -np.inf)))
    accept = 1.
    for cutoff in sorted(set(float(v) for v in scores)):
        if cutoff <= max(reject, .5):
            continue
        accepted = scores >= cutoff
        if accepted.sum() >= 10 and truth[accepted].mean() >= .95:
            accept = cutoff
            break
    if accept <= reject:
        accept = min(1., float(np.nextafter(reject, np.inf)))
    return dict(reject_below=reject,accept_above=accept,selection_split='val',
                target_positive_retention=.99,target_accept_precision=.95)


def metrics(scores, truth, thresholds):
    statuses = [decision(float(score), thresholds['reject_below'], thresholds['accept_above']) for score in scores]
    accepted = np.array([s == 'likely_plate' for s in statuses])
    rejected = np.array([s == 'likely_non_plate' for s in statuses])
    positives, negatives = truth == 1, truth == 0
    return dict(examples=len(truth),positives=int(positives.sum()),assumed_background_negatives=int(negatives.sum()),
        tp=int((accepted&positives).sum()),fp=int((accepted&negatives).sum()),
        tn=int((rejected&negatives).sum()),fn=int((rejected&positives).sum()),
        uncertain_positive=int((~accepted&~rejected&positives).sum()),
        uncertain_negative=int((~accepted&~rejected&negatives).sum()),
        positive_retention=float((~rejected&positives).sum()/positives.sum()),
        accepted_precision=float((accepted&positives).sum()/accepted.sum()) if accepted.sum() else None,
        decision_coverage=float((accepted|rejected).mean()))


def run(dataset, output, assets):
    started = time.monotonic()
    if output.exists() or assets.exists():
        raise ValueError('Fresh output and asset folders required')
    manifest = dataset/'manifest.json'
    rows = json.loads(manifest.read_bytes())['samples']
    images_per_split = validate_splits(rows)
    output.mkdir(parents=True); assets.mkdir(parents=True)
    arrays = {split: [[], []] for split in ('train','val','test')}
    membership = []
    for row in rows:
        relative = Path(row['image_path'])
        path = (dataset/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(dataset.resolve()):
            raise ValueError('Unsafe image path')
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (row['height'],row['width']):
            raise ValueError('Image missing or dimensions changed')
        if sha256(path) != row['original_sha256'] or hashlib.sha256(image.tobytes()).hexdigest() != row['pixels']:
            raise ValueError('Frozen source byte/pixel hash changed')
        positives = positive_boxes(row)
        x, y = arrays[row['split']]
        generated = []
        for label, boxes in ((1, positives),(0, negatives(row, positives))):
            for box in boxes:
                a,b,c,d = box; crop = image[b:d,a:c]
                variants = [crop]
                if label and row['split'] == 'train':
                    variants.append(cv2.convertScaleAbs(crop,alpha=.55,beta=60))
                    variants.append(cv2.copyMakeBorder(crop,2,2,max(2,round(crop.shape[1]*.08)),max(2,round(crop.shape[1]*.08)),cv2.BORDER_REPLICATE))
                for variant in variants:
                    x.append(extract_features(variant));y.append(label)
                generated.append(dict(label=label,box=box,variants=len(variants)))
        membership.append(dict(id=row['id'],group=row['group'],pixels=row['pixels'],split=row['split'],
                               source=row['source'],image_sha256=sha256(path),examples=generated))
    data = {split:(np.asarray(x,np.float32),np.asarray(y,np.int32)) for split,(x,y) in arrays.items()}
    model = LogisticRegression(C=.3,class_weight='balanced',solver='lbfgs',max_iter=2000,random_state=31)
    model.fit(*data['train'])
    weights=model.coef_[0].astype(np.float32)
    bias=model.intercept_.astype(np.float64)
    def scores_for(features):
        return np.asarray([1/(1+np.exp(-np.clip(float(feature@weights+float(bias[0])),-60,60))) for feature in features])
    val_scores = scores_for(data['val'][0])
    thresholds = threshold_selection(val_scores,data['val'][1])
    model_path = assets/'model.npz'
    np.savez_compressed(model_path,weights=weights,bias=bias)
    receipt = dict(schema_version=1,model_type='linear-logistic-hog',feature_version=FEATURE_VERSION,
        feature_dimension=FEATURE_DIM,model_sha256=sha256(model_path),thresholds=thresholds,
        manifest_sha256=sha256(manifest),excluded_evaluation_source_sha256=MUMBAI_HASH,
        label_scope='Publisher positive boxes; assumed background negatives outside all annotated plates; not independently human verified',
        training_sources=[dict(source='Gujarat vehicle plates',licence='Apache-2.0',url='https://www.kaggle.com/datasets/paneraghanshyam/gujarat-vehicle-number-plates-yolo-ready'),
                          dict(source='Make ML / Larxel plate detection',licence='CC0',url='https://www.kaggle.com/datasets/andrewmvd/car-plate-detection')],
        score_scope='Uncalibrated visual classifier score; not real-world plate probability or OCR confidence',
        training=dict(seed=31,C=.3,solver='lbfgs',max_iter=2000,iterations=model.n_iter_.tolist(),images_per_split=images_per_split,
                      no_cross_split_group_or_pixel_overlap=True,threshold_selection_split='val',test_used_for_fitting=False))
    model_path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    # Test is evaluated once, only after fixed model and thresholds were serialized.
    verifier=PlateRegionVerifier(model_path)
    test_scores=scores_for(data['test'][0])
    summaries={split:metrics(scores,data[split][1],thresholds) for split,scores in (
        ('train',scores_for(data['train'][0])),('val',val_scores),('test',test_scores))}
    report=dict(model_sha256=receipt['model_sha256'],manifest_sha256=receipt['manifest_sha256'],thresholds=thresholds,
        metrics=summaries,training=receipt['training'],wall_seconds=round(time.monotonic()-started,2),
        label_scope=receipt['label_scope'],mumbai_used=False,excluded_evaluation_source_sha256=MUMBAI_HASH,
        limitations=['Source images share original detector-dataset grouping, not independently verified camera identity',
                     'Background negatives may include missed publisher annotations or text branding; precision is proxy-labelled',
                     'Painted registration and Mumbai CCTV performance are not established by these dataset scores',
                     'Validation thresholds apply to these domains; uncertain candidates should remain reviewable'])
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (output/'membership.json').write_text(json.dumps(membership,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','output','assets'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.dataset,args.output,args.assets)
