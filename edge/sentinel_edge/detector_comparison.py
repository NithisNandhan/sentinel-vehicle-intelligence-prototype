"""Isolated detector diagnostics: frozen permitted inputs, no production changes.

Target coverage is not exhaustive-scene recall or OCR accuracy. Background
objects are unlabelled, so unmatched predictions are not called false positives.
"""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

CONDITIONS = ('native', 'resolution_half', 'resolution_quarter', 'gaussian5', 'motion9')
VEHICLES = ('yolov8n', 'yolo26n', 'yolo26s')
PLATES = ('yolov9t512', 'yolov9s608', 'yolo11m_plate', 'yolo11l_plate', 'yolo11x_plate')


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/union if union > 0 else 0.


def score_target(boxes, target):
    overlaps = [iou(box, target) for box in boxes]
    best = max(overlaps, default=0.)
    return {'target_iou': round(best, 4), 'hit_iou50': int(best >= .5),
            'hit_iou75': int(best >= .75), 'returned_boxes': len(boxes),
            'unmatched_boxes_not_verified_fp': sum(v < .5 for v in overlaps)}


def load_inputs(manifest, root):
    import cv2
    root = root.resolve()
    document = json.loads(manifest.read_text())
    if document.get('permission') != 'noncommercial_benchmark':
        raise ValueError('Explicit benchmark permission required')
    rows, ids = [], set()
    for sample in document['samples']:
        source = urlsplit(sample['source_url'])
        if source.scheme != 'https' or source.hostname not in {'zenodo.org', 'universe.roboflow.com'}:
            raise ValueError('Only previously permitted non-Sentinel sources are allowed')
        if sample['license'] != 'CC-BY-4.0' or sample['id'] in ids:
            raise ValueError('Invalid licence or duplicate sample')
        ids.add(sample['id'])
        relative = Path(sample['path']); path = (root/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or path.suffix.lower() not in {'.png', '.jpg'}:
            raise ValueError('Media must remain inside permitted benchmark root')
        if hashlib.sha256(path.read_bytes()).hexdigest() != sample['sha256']:
            raise ValueError('Frozen original image changed')
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Image decode failed')
        for key in ('vehicle_box', 'plate_box'):
            box = sample[key]
            if (len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= image.shape[1]
                or not 0 <= box[1] < box[3] <= image.shape[0]):
                raise ValueError('Invalid target annotation bounds')
        rows.append((sample, image))
    if not rows or len(rows) > 100:
        raise ValueError('Bounded nonempty diagnostic required')
    return document, rows


def transformed(image, condition, target):
    import cv2
    result = image.copy(); box = list(target)
    if condition.startswith('resolution_'):
        scale = .5 if condition == 'resolution_half' else .25
        height, width = image.shape[:2]
        result = cv2.resize(image, (max(1, round(width*scale)), max(1, round(height*scale))), interpolation=cv2.INTER_AREA)
        sx, sy = result.shape[1]/width, result.shape[0]/height
        box = [target[0]*sx, target[1]*sy, target[2]*sx, target[3]*sy]
    elif condition == 'gaussian5':
        result = cv2.GaussianBlur(result, (5, 5), 0)
    elif condition == 'motion9':
        kernel = np.zeros((9,9), dtype=np.float32); kernel[4,:] = 1/9
        result = cv2.filter2D(result, -1, kernel)
    elif condition != 'native':
        raise ValueError('Unknown diagnostic condition')
    return result, box


def letterbox(image, height, width):
    import cv2
    scale = min(height/image.shape[0], width/image.shape[1])
    resized = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
    left = (width-resized.shape[1])//2; top = (height-resized.shape[0])//2
    canvas = np.full((height,width,3), 114, dtype=np.uint8)
    canvas[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
    tensor = canvas[:,:,::-1].transpose(2,0,1)[None].astype(np.float32)/255
    return np.ascontiguousarray(tensor), scale, left, top


def decode_yolo11(output, scale, left, top, shape, threshold=.25):
    import cv2
    values = np.asarray(output)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] != 5:
        raise ValueError('Unexpected single-class YOLO11 output; do not guess output semantics')
    rows = values[0].T
    rows = rows[rows[:,4] >= threshold]
    boxes, confidences = [], []
    for cx,cy,w,h,confidence in rows:
        x1,y1 = (cx-w/2-left)/scale, (cy-h/2-top)/scale
        x2,y2 = (cx+w/2-left)/scale, (cy+h/2-top)/scale
        box = [max(0.,float(x1)), max(0.,float(y1)), min(float(shape[1]),float(x2)), min(float(shape[0]),float(y2))]
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box); confidences.append(float(confidence))
    if not boxes:
        return []
    rects = [[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in boxes]
    kept = cv2.dnn.NMSBoxes(rects, confidences, threshold, .45)
    return [boxes[int(index)] for index in np.asarray(kept).reshape(-1)]


class PlateModel:
    def __init__(self, name, cache, device):
        import onnxruntime as ort
        options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.device = device
        if name.startswith('yolov9'):
            from open_image_models import create_detector
            file = 'yolo-v9-t-512-license-plates-end2end.onnx' if name == 'yolov9t512' else 'yolo-v9-s-608-license-plates-end2end.onnx'
            self.model = create_detector(cache/file, backend='yolo_v9', class_labels=['License Plate'],
                conf_thresh=.25, providers=providers, sess_options=options)
            self.height = self.width = 512 if name == 'yolov9t512' else 608
            # Provider audit is performed against the actual runtime session below.
            sessions = [v for v in vars(self.model).values() if isinstance(v, ort.InferenceSession)]
            if not sessions:
                for value in vars(self.model).values():
                    if hasattr(value, '__dict__'):
                        sessions.extend(v for v in vars(value).values() if isinstance(v, ort.InferenceSession))
            if not sessions:
                raise ValueError('Unable to audit actual detector providers')
            self.providers = sessions[0].get_providers()
            self.session = None
        else:
            scale = name[len('yolo11')]
            self.session = ort.InferenceSession(str(cache/f'license-plate-finetune-v1{scale}.onnx'), sess_options=options, providers=providers)
            self.providers = self.session.get_providers()
            spec = self.session.get_inputs()[0]
            if len(spec.shape) != 4 or spec.shape[0] not in (1, 'batch', 'batch_size') or spec.shape[1] != 3:
                raise ValueError('Unexpected model input contract')
            self.height = spec.shape[2] if isinstance(spec.shape[2], int) else 640
            self.width = spec.shape[3] if isinstance(spec.shape[3], int) else 640
            self.input_name = spec.name
        if device == 'cuda' and 'CUDAExecutionProvider' not in self.providers:
            raise RuntimeError('GPU requested but runtime fell back to CPU')

    def predict(self, image):
        if self.session is None:
            return [[d.bounding_box.x1,d.bounding_box.y1,d.bounding_box.x2,d.bounding_box.y2] for d in self.model.predict(image)]
        tensor, scale, left, top = letterbox(image, self.height, self.width)
        output = self.session.run(None, {self.input_name:tensor})[0]
        return decode_yolo11(output, scale, left, top, image.shape)


class VehicleModel:
    def __init__(self, name, cache, device, size, classes=(2,3,5,7)):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(2)
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        path = Path('/models/yolov8n.pt') if name == 'yolov8n' else cache/(name+'.pt')
        self.model = YOLO(str(path)); self.device = device; self.size = size
        self.classes = list(classes)

    def predict(self, image):
        result = self.model.predict(image, imgsz=self.size, classes=self.classes, conf=.25,
            iou=.45, device='cpu' if self.device == 'cpu' else 0, verbose=False, save=False)[0]
        return result.boxes.xyxy.cpu().tolist()


def summarize(rows):
    total = len(rows)
    return {'cases': total, 'target_hits_iou50': sum(r['hit_iou50'] for r in rows),
        'target_hits_iou75': sum(r['hit_iou75'] for r in rows),
        'target_coverage_iou50': round(sum(r['hit_iou50'] for r in rows)/total,4),
        'median_target_iou': round(statistics.median(r['target_iou'] for r in rows),4),
        'median_ms': round(statistics.median(r['elapsed_ms'] for r in rows),2),
        'p95_ms': round(float(np.percentile([r['elapsed_ms'] for r in rows],95)),2),
        'returned_box_instances': sum(r['returned_boxes'] for r in rows),
        'unmatched_box_instances_not_verified_fp': sum(r['unmatched_boxes_not_verified_fp'] for r in rows)}


def run(kind, cache, manifest, root, report, devices=('cpu','cuda')):
    document, inputs = load_inputs(manifest, root)
    if not report.resolve().is_relative_to((root/'reports').resolve()) or report.suffix != '.json':
        raise ValueError('Reports must remain inside ignored benchmark reports')
    models = VEHICLES if kind == 'vehicle' else PLATES
    result = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'kind': kind,
        'runtime_versions': {package: version(package)
                             for package in ('numpy','opencv-python-headless','onnxruntime-gpu','ultralytics')},
        'asset_metadata': json.loads((cache/'assets.json').read_text()),
        'input_images': len(inputs), 'distinct_target_vehicles': len({s['vehicle_id'] for s,_ in inputs}),
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'independent_human_ground_truth': False, 'held_out_accuracy_benchmark': False,
        'scene_precision_recall': None, 'ocr_accuracy': None, 'live_validation': False,
        'caveat': document['annotation_scope']+'; approximate AI-reviewed boxes; stress variants are not independent vehicles; provider GPU use is audited, not all-operator GPU placement',
        'confidence_threshold': .25, 'models': []}
    import cv2
    cv2.setNumThreads(2)
    for name in models:
        for device in devices:
            for size in ((640,960) if kind == 'vehicle' else (None,)):
                entry = {'name':name,'device':device,'input_size':size,'status':'unavailable'}
                try:
                    model = VehicleModel(name, cache, device, size) if kind == 'vehicle' else PlateModel(name, cache, device)
                    if kind == 'plate':
                        entry['input_size'] = [model.height, model.width]; entry['providers'] = model.providers
                    for _ in range(3): model.predict(inputs[0][1])
                    rows = []
                    for condition in CONDITIONS:
                        for sample,image in inputs:
                            transformed_image, box = transformed(image, condition, sample[kind+'_box'])
                            started = time.perf_counter(); predictions = model.predict(transformed_image)
                            elapsed = (time.perf_counter()-started)*1000
                            rows.append({'id':sample['id'],'condition':condition,
                                'elapsed_ms':round(elapsed,2),**score_target(predictions,box)})
                    entry.update(status='completed',summary=summarize(rows),
                        conditions={c:summarize([r for r in rows if r['condition']==c]) for c in CONDITIONS}, cases=rows)
                    del model; gc.collect()
                    if kind == 'vehicle' and device == 'cuda':
                        import torch
                        torch.cuda.empty_cache()
                except Exception as error:
                    # No raw third-party exceptions, URLs, private images or plate text.
                    entry['error_type'] = type(error).__name__
                result['models'].append(entry)
                report.write_text(json.dumps(result,indent=2))
                print(json.dumps({k:v for k,v in entry.items() if k not in ('cases','conditions')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['vehicle','plate'],required=True)
    parser.add_argument('--cache',type=Path,default=Path('/models/experiments/detectors'))
    parser.add_argument('--root',type=Path,default=Path('/workspace/data/benchmarks'))
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--devices',default='cpu,cuda')
    args = parser.parse_args()
    run(args.kind,args.cache,args.root/'detector-comparison/manifest.json',args.root,args.report,tuple(args.devices.split(',')))
