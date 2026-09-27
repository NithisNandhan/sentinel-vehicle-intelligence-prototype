"""Bounded isolated candidate training; never promote a live default."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import yaml
import torch
torch.set_num_threads(2)

parser = argparse.ArgumentParser()
parser.add_argument('--kind',choices=['vehicle','plate'],required=True)
parser.add_argument('--epochs',type=int,default=20)
parser.add_argument('--name',default=None)
parser.add_argument('--dataset',default=None)
args = parser.parse_args()
if not 1 <= args.epochs <= 60:
    raise ValueError('Bounded experiment required')
dataset = args.dataset or args.kind
if not dataset.replace('-','').isalnum():
    raise ValueError('Simple dataset name required')
root = Path('/training')/dataset
manifest = json.loads((root/'manifest.json').read_text())
if manifest['kind'] != args.kind or any(manifest['counts'][s]<3 for s in ('train','val','test')):
    raise ValueError('Prepared partitions required')
if manifest.get('source')=='iisc-aim/BMD-45' and not manifest.get('near_image_guard_completed'):
    raise ValueError('BMD near-image leakage guard must finish before training')
name = args.name or args.kind
if not name.replace('-','').isalnum():
    raise ValueError('Simple experiment name required')
output = Path('/training/runs')/name
if output.exists():
    raise ValueError('Existing experiment preserved; use a new run directory explicitly')
started = time.time()
record = {'kind':args.kind,'requested_epochs':args.epochs,'production_promoted':False,
          'training_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
          'split_counts':manifest['counts'],'evaluation_caveat':manifest['caveat'],
          'seed':42,'fp32_training':True,'batch_size':4,'frozen_backbone_layers':10,
          'checkpoint_selected_by':'validation_fitness_not_reserved_test',
          'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
if args.kind == 'vehicle':
    from ultralytics import YOLO
    parent = Path('/models/experiments/detectors/yolo26s.pt')
    record['parent_sha256'] = hashlib.sha256(parent.read_bytes()).hexdigest()
    model = YOLO(str(parent))
    # Publisher categories, not a guessed COCO mapping. Baseline and
    # candidate comparisons use pooled vehicle boxes; production remains COCO.
    model.train(data=str(root/'data.yaml'),epochs=args.epochs,imgsz=640,batch=4,
        workers=0,device=0,amp=False,optimizer='AdamW',lr0=.0002,lrf=.1,
        freeze=10,patience=6,seed=42,deterministic=True,mosaic=.2,close_mosaic=3,
        fliplr=.5,mixup=0,translate=.05,scale=.15,degrees=3,
        project='/training/runs',name=name,exist_ok=False,plots=False,
        save=True,cache=False,verbose=False)
else:
    source = Path('/models/experiments/yolov9-training')
    os.chdir(source); sys.path.insert(0,str(source))
    import train
    from utils.callbacks import Callbacks
    import numpy as np
    # NumPy 2.4 removed trapz; trapezoid is its equivalent integration API.
    # Keep upstream files unchanged and preserve the metric calculation.
    if not hasattr(np,'trapz'):
        np.trapz = np.trapezoid
    from functools import partial
    from tqdm import tqdm
    train.tqdm = partial(tqdm,disable=True)
    train.validate.tqdm = partial(tqdm,disable=True)
    parent = Path('/models/experiments/detectors/yolo-v9-t-512-license-plates.pt')
    receipt = json.loads((parent.parent/'trainable-plate.json').read_text())
    if hashlib.sha256(parent.read_bytes()).hexdigest() != receipt['sha256']:
        raise ValueError('Publisher training checkpoint changed')
    record['parent_sha256'] = receipt['sha256']
    opt = train.parse_opt(known=True)
    opt.weights = str(parent); opt.data = str(root/'data.yaml'); opt.cfg = ''
    opt.epochs = args.epochs; opt.batch_size = 4; opt.imgsz = 512
    opt.device = '0'; opt.workers = 0; opt.freeze = [10]
    opt.optimizer = 'AdamW'; opt.patience = 6; opt.seed = 42
    opt.project = Path('/training/runs'); opt.name = name
    opt.exist_ok = False; opt.noplots = True; opt.close_mosaic = 3
    # Network-disabled container: skip automatic reference-model AMP download
    # and train FP32. This does not modify the pinned source on disk.
    train.check_amp = lambda model: False
    # Plots are disabled; the original dataset checker otherwise downloads a
    # decorative font even in offline mode. No label checks are bypassed.
    import utils.general
    utils.general.check_font = lambda *a,**kw: None
    hyp = yaml.safe_load((source/'data/hyps/hyp.scratch-high.yaml').read_text())
    hyp.update(lr0=.0002,lrf=.1,warmup_epochs=1,warmup_bias_lr=.001,
               mosaic=.2,mixup=0,copy_paste=0,translate=.05,scale=.15,degrees=3)
    hyp_path = root/'hyp-transfer.yaml'; hyp_path.write_text(yaml.safe_dump(hyp))
    opt.hyp = str(hyp_path)
    callbacks = Callbacks()
    def progress(vals,epoch,best_fitness,fi):
        print(json.dumps({'epoch':epoch+1,'validation_map50':round(float(vals[5]),4),
                          'validation_map50_95':round(float(vals[6]),4)}),flush=True)
    callbacks.register_action('on_fit_epoch_end',callback=progress)
    train.main(opt,callbacks)
record.update(elapsed_seconds=round(time.time()-started,2),status='training_completed')
best = output/'weights/best.pt'
if not best.is_file():
    raise ValueError('Trainer returned without selected checkpoint')
record['best_sha256'] = hashlib.sha256(best.read_bytes()).hexdigest()
with (output/'results.csv').open() as stream:
    record['completed_epochs']=len(list(csv.DictReader(stream)))
(output/'training-receipt.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
