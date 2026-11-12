"""Frozen vehicle-ROI detector comparison; labels are read only by score().

This is a development diagnostic on the Mumbai reference timestamps, not
whole-scene precision/recall, a held-out benchmark, or a multiframe comparison.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'edge'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_capture(path):
    """Return frozen automatic inputs; never open labels or manual geometry."""
    import cv2
    data = json.loads((path / 'audit.json').read_bytes())
    if data['source_sha256'] != '6b7064605b3d4d12c6c1da4566cbcec1cd2e3e3bd7b77814865fe9df22045f65':
        raise ValueError('Unexpected development recording')
    cases = []
    for frame in data['frames']:
        for index, region in enumerate(frame['regions']):
            image_path = (path / region['path']).resolve()
            if not image_path.is_relative_to(path.resolve()) or sha(image_path) != region['sha256']:
                raise ValueError('Frozen native vehicle region changed')
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError('Vehicle region decode failed')
            cases.append((f"{frame['position_ms']:06}-{index:03}", frame['position_ms'], region, image))
    return data, cases


def prepare(args):
    """Download only the three requested ONNX models, pinned to publisher hashes."""
    import urllib.request
    repo = 'morsetechlab/yolov11-license-plate-detection'
    def get(url):
        with urllib.request.urlopen(url, timeout=90) as response:
            return json.load(response)
    revision = get('https://huggingface.co/api/models/' + repo)['sha']
    files = {r['path']: r for r in get(f'https://huggingface.co/api/models/{repo}/tree/{revision}?expand=true')}
    args.cache.mkdir(parents=True, exist_ok=True)
    records = []
    for scale in 'mlx':
        name = f'license-plate-finetune-v1{scale}.onnx'
        expected = files[name]['lfs']['oid']
        path = args.cache / name
        if not path.exists() or sha(path) != expected:
            temporary = path.with_suffix('.partial')
            total = 0
            with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', timeout=180) as response, temporary.open('wb') as stream:
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > 600 * 1024 * 1024:
                        raise ValueError('Model exceeds bounded download size')
                    stream.write(block)
            if sha(temporary) != expected:
                raise ValueError('Publisher model checksum mismatch')
            temporary.replace(path)
        records.append(dict(name=name, sha256=expected, revision=revision, source=repo, license='AGPL-3.0'))
        print(json.dumps(dict(stage='cached', model=name, bytes=path.stat().st_size)), flush=True)
    write(args.cache / 'yolo11-comparison-receipt.json', dict(models=records,
        warning='Publisher reports train/test contamination. Published metrics are not used.'))


def native_crop(image, box):
    x1, y1, x2, y2 = box
    a, b = max(0, math.floor(x1)), max(0, math.floor(y1))
    c, d = min(image.shape[1], math.ceil(x2)), min(image.shape[0], math.ceil(y2))
    if c <= a or d <= b:
        raise ValueError('Empty detector crop')
    return image[b:d, a:c].copy(), (a, b)


def infer(args):
    import cv2
    import numpy as np
    source, cases = load_capture(args.capture)
    output = args.output / args.model
    if output.exists():
        raise ValueError('Fresh per-model output required')
    output.mkdir(parents=True)
    (output / 'assets').mkdir()
    began = time.monotonic()
    if args.model == 'iwpod':
        from iwpod_adapter import IWPODDetector
        detector = IWPODDetector(args.iwpod_source)
        model_path = args.iwpod_source / 'weights/iwpod_net.h5'
        providers = ['TensorFlowCPU']
    elif args.model == 'roi_adapted':
        import torch
        import onnxruntime as ort
        from types import SimpleNamespace
        model_path=args.candidate
        receipt=json.loads(model_path.with_suffix('.receipt.json').read_bytes())
        if (sha(model_path)!=receipt['onnx_sha256'] or receipt['input_shape']!=[1,3,512,512]
                or receipt['output_shape']!=[1,5,5376] or not receipt.get('zero_input_cpu_parity')
                or not receipt.get('real_image_parity_passed') or receipt.get('real_image_parity_checks')!=48):
            raise ValueError('Verified adapted raw YOLOv9 export required')
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        providers=([('CUDAExecutionProvider',{'use_tf32':'0'}),'CPUExecutionProvider']
                   if args.device=='cuda' else ['CPUExecutionProvider'])
        session=ort.InferenceSession(str(model_path),options,providers=providers)
        providers=session.get_providers()
        if args.device=='cuda' and 'CUDAExecutionProvider' not in providers:
            raise RuntimeError('Candidate detector GPU fallback')
        detector=SimpleNamespace(session=session,height=512,width=512,input_name=session.get_inputs()[0].name,
                                 predict=lambda image: None)
        # Warm the actual ONNX session, not the wrapper placeholder.
        from sentinel_edge.detector_comparison import letterbox
        detector.predict=lambda image:session.run(None,{detector.input_name:letterbox(image,512,512)[0]})
    else:
        # Import Torch first to expose its CUDA/cuDNN dependencies to ORT.
        if args.device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('Requested CUDA unavailable')
        from sentinel_edge.detector_comparison import PlateModel
        underlying='yolov9t512' if args.model=='baseline_scale' else args.model
        detector = PlateModel(underlying, args.cache, args.device)
        providers = detector.providers
        names = {'yolov9t512': 'yolo-v9-t-512-license-plates-end2end.onnx',
                 'yolov9s608': 'yolo-v9-s-608-license-plates-end2end.onnx'}
        model_path = args.cache / names.get(underlying, f'license-plate-finetune-v1{underlying[6]}.onnx')
    cold_seconds = time.monotonic() - began
    # Warmup excluded from steady-state timings, same first automatic ROI.
    detector.predict(cases[0][3])
    rows, timings = [], []
    detector_calls=0
    for case_id, pts, region, image in cases:
        start = time.monotonic()
        if args.model == 'baseline_scale':
            from sentinel_edge.plate_scale_search import detect_views
            def scored_predict(view):
                return [([d.bounding_box.x1,d.bounding_box.y1,d.bounding_box.x2,d.bounding_box.y2],
                         float(d.confidence)) for d in detector.model.predict(view)]
            proposals,calls=detect_views(image,scored_predict,max_calls=6)
            detector_calls+=calls
        elif args.model == 'iwpod':
            proposals = detector.predict(image)
        elif detector.session is None:
            proposals = [dict(box=[d.bounding_box.x1, d.bounding_box.y1, d.bounding_box.x2, d.bounding_box.y2],
                              confidence=float(d.confidence)) for d in detector.model.predict(image)]
        else:
            from sentinel_edge.detector_comparison import letterbox, decode_yolo11
            tensor, scale, left, top = letterbox(image, detector.height, detector.width)
            values = detector.session.run(None, {detector.input_name: tensor})[0]
            proposals = [dict(box=b, confidence=c) for b, c in decode_yolo11(
                values, scale, left, top, image.shape, with_confidence=True)]
        elapsed_ms = (time.monotonic() - start) * 1000
        if args.model!='baseline_scale':detector_calls+=1
        timings.append(elapsed_ms)
        predictions = []
        for index, proposal in enumerate(proposals):
            box = [float(v) for v in proposal['box']]
            if len(box) != 4 or not all(math.isfinite(v) for v in box):
                raise ValueError('Invalid detector box')
            crop, origin = native_crop(image, box)
            relative = f'assets/{case_id}-{index:02}-native.png'
            if not cv2.imwrite(str(output / relative), crop):
                raise ValueError('Native crop save failed')
            item = dict(box=box, confidence=float(proposal['confidence']), native_path=relative,
                        native_sha256=sha(output / relative), quad=proposal.get('quad'),
                        detection_view=proposal.get('view','full'))
            if proposal.get('rectified') is not None:
                relative = f'assets/{case_id}-{index:02}-rectified.png'
                if not cv2.imwrite(str(output / relative), proposal['rectified']):
                    raise ValueError('Rectified save failed')
                item.update(rectified_path=relative, rectified_sha256=sha(output / relative))
            predictions.append(item)
        rows.append(dict(id=case_id, position_ms=pts, region_id=region['id'], region_box=region['box'],
                         source_path=region['path'], source_sha256=region['sha256'], elapsed_ms=elapsed_ms,
                         predictions=predictions))
        if len(rows) % 20 == 0:
            print(json.dumps(dict(stage='inference', model=args.model, regions=len(rows))), flush=True)
            write(output / 'progress.json', dict(stage='inference', regions=len(rows)))
    result = dict(model=args.model, model_sha256=sha(model_path), source_sha256=source['source_sha256'],
                  capture_sha256=sha(args.capture / 'audit.json'), labels_used_during_inference=False,
                  confidence_threshold=.25 if args.model != 'iwpod' else detector.threshold,
                  providers=providers, device=args.device if args.model != 'iwpod' else 'cpu',
                  cold_load_seconds=cold_seconds, median_ms=statistics.median(timings),
                  p95_ms=float(np.percentile(timings, 95)), elapsed_seconds=time.monotonic() - began,
                  regions=len(cases), detector_calls=detector_calls,
                  proposals=sum(len(r['predictions']) for r in rows), frames=rows,
                  script_sha256=sha(__file__), scene_precision_recall=None, held_out=False)
    if args.model=='baseline_scale':
        result.update(scale_policy='full_view_protected_v2',
                      scale_policy_sha256=sha(ROOT/'edge/sentinel_edge/plate_scale_search.py'))
    write(output / 'predictions.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'frames'}), flush=True)


def ocr(args):
    import cv2
    from sentinel_edge.awiros_experiment import AwirosExperiment
    from sentinel_edge.provisional_plate_reading import unconstrained_top
    path = args.output / args.model / 'predictions.json'
    data = json.loads(path.read_bytes())
    if 'ocr_model_sha256' in data:
        raise ValueError('OCR output already recorded')
    checkpoint = args.ocr_cache / 'adaptations/indian-corpus-encoder.safetensors'
    if sha(checkpoint) != 'c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d':
        raise ValueError('Frozen OCR checkpoint changed')
    engine = AwirosExperiment(args.ocr_cache, device='gpu' if args.device == 'cuda' else 'cpu', checkpoint=checkpoint)
    cache, timings = {}, []
    for index, row in enumerate(data['frames']):
        for p in row['predictions']:
            for variant in ('native', 'rectified'):
                key = variant + '_path'
                if key not in p:
                    continue
                image_path = args.output / args.model / p[key]
                if sha(image_path) != p[variant + '_sha256']:
                    raise ValueError('OCR input changed')
                digest = p[variant + '_sha256']
                if digest not in cache:
                    image = cv2.imread(str(image_path))
                    began = time.monotonic()
                    top = unconstrained_top(engine.probabilities(image), engine.tokens)
                    timings.append((time.monotonic() - began) * 1000)
                    cache[digest] = top
                p[variant + '_reading'] = cache[digest]
        if (index + 1) % 20 == 0 or index + 1 == len(data['frames']):
            print(json.dumps(dict(stage='ocr', model=args.model, regions=index + 1, calls=len(cache))), flush=True)
    data.update(ocr_model_sha256=sha(checkpoint), ocr_device=args.device, ocr_calls=len(cache),
                ocr_median_ms=statistics.median(timings) if timings else None,
                ocr_decoder='unconstrained_ctc_prefix_top', ocr_labels_used=False)
    write(path, data)


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aa, bb = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return intersection / max(1e-12, aa+bb-intersection), intersection / max(1e-12, bb)


def evaluate_reference(target, rows):
    """Geometry selects one prediction, without looking at OCR or true text."""
    candidates = []
    eligible_roi = False
    for row in rows:
        if row['position_ms'] != target['position_ms']:
            continue
        eligible_roi |= overlap(row['region_box'], target['box'])[1] >= .95
        x, y = row['region_box'][:2]
        for p in row['predictions']:
            a,b,c,d=p['box']; full=[x+a,y+b,x+c,y+d]
            iou, coverage = overlap(full, target['box'])
            if iou >= .3:
                candidates.append((iou, float(p['confidence']), coverage, row, p))
    if not candidates:
        return dict(number=target['number'], roi_available=bool(eligible_roi), found=False, matched_candidates=0,
                    native_exact=False, rectified_exact=False, either_exact=False)
    iou, confidence, coverage, row, p = max(candidates, key=lambda v:(v[0],v[1]))
    native = p.get('native_reading') == target['actual']
    rectified = p.get('rectified_reading') == target['actual']
    return dict(number=target['number'], roi_available=bool(eligible_roi), found=True,
                matched_candidates=len(candidates), iou=iou, reference_box_coverage=coverage,
                confidence=confidence, region_id=row['id'], crop=p, native_exact=native,
                rectified_exact=rectified, either_exact=native or rectified)


def score(args):
    review = json.loads((args.review / 'manifest.json').read_bytes())
    view = json.loads(args.reference_view.read_bytes())
    if review['source_sha256'] != view['source_sha256']:
        raise ValueError('Reference source mismatch')
    corners = {}
    for node in ast.parse((args.review / 'ocr-model-dev/rectify.py').read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('CORNERS', 'TEXT_REGIONS') for t in node.targets):
            corners.update(ast.literal_eval(node.value))
    samples = {r['id']:r for r in review['samples']}
    targets = []
    for r in view['references']:
        sample=samples[r['reference_id']];box=sample['box'];points=corners.get(sample['id'])
        if points:
            box=[box[0]+min(p[0] for p in points), box[1]+min(p[1] for p in points),
                 box[0]+max(p[0] for p in points), box[1]+max(p[1] for p in points)]
        targets.append(dict(number=r['number'], position_ms=sample['position_ms'], box=box, actual=r['actual']))
    models=[]
    capture_hash = None
    ocr_hash = None
    for folder in sorted(args.output.iterdir()):
        path=folder/'predictions.json'
        if not path.exists(): continue
        d=json.loads(path.read_bytes())
        if d['source_sha256'] != view['source_sha256'] or not d.get('ocr_model_sha256'):
            raise ValueError('Only complete same-source inference and OCR can be scored')
        if capture_hash is None:
            capture_hash, ocr_hash = d['capture_sha256'], d['ocr_model_sha256']
        if (d['capture_sha256'], d['ocr_model_sha256']) != (capture_hash, ocr_hash):
            raise ValueError('Compared models must use identical frozen inputs and OCR checkpoint')
        cases=[evaluate_reference(t,d['frames']) for t in targets]
        summary={k:d[k] for k in ('model','model_sha256','providers','device','regions','proposals','median_ms','p95_ms','ocr_calls','ocr_device')}
        summary.update(references=len(cases),roi_available=sum(c['roi_available'] for c in cases),
            found_iou30=sum(c['found'] for c in cases), complete_reference_box_95=sum(c['found'] and c['reference_box_coverage']>=.95 for c in cases),
            native_exact=sum(c['native_exact'] for c in cases),rectified_exact=sum(c['rectified_exact'] for c in cases),
            either_view_oracle_exact=sum(c['either_exact'] for c in cases),cases=cases)
        models.append(summary)
    report=dict(scope='Same 149 frozen automatic vehicle ROIs at 14 reference timestamps; 18 development references. Geometry-only association uses maximum IoU >=0.30. No labels used for inference or model tuning.',
                source_sha256=view['source_sha256'], held_out=False, multiframe=False,
                false_positives=None, warning='All other predictions are unlabelled, not verified false positives. Either-view exact is an oracle ceiling, not an automatic selector.',models=models)
    write(args.output/'comparison.json',report)
    print(json.dumps([{k:v for k,v in m.items() if k!='cases'} for m in models]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','infer','ocr','score'))
    p.add_argument('--model',default='yolov9t512',choices=('yolov9t512','yolov9s608','yolo11m_plate','yolo11l_plate','yolo11x_plate','iwpod','roi_adapted','baseline_scale'))
    p.add_argument('--candidate',type=Path,default=Path('/models/experiments/detectors/yolov9t512-context-20261112.onnx'))
    p.add_argument('--device',default='cuda',choices=('cpu','cuda'))
    p.add_argument('--cache',type=Path,default=Path('/models/experiments/detectors'))
    p.add_argument('--ocr-cache',type=Path,default=Path('/models/experiments/awiros'))
    p.add_argument('--capture',type=Path,default=ROOT/'data/benchmarks/provisional-unfiltered-audit-20261111')
    p.add_argument('--output',type=Path,default=ROOT/'data/benchmarks/detector-corners-20261112/results')
    p.add_argument('--iwpod-source',type=Path,default=ROOT/'data/benchmarks/detector-corners-20261112/iwpod-source')
    p.add_argument('--review',type=Path,default=ROOT/'data/benchmarks/recorded-footage-20261104/manual-review')
    p.add_argument('--reference-view',type=Path,default=ROOT/'data/benchmarks/provisional-reading-review-20261112/audit-view.json')
    args=p.parse_args()
    # The cached Paddle build uses CUDA12; this container's Torch uses CUDA13.
    # Resolve only the research OCR process against its already cached CUDA12
    # libraries; do not change the system loader or the detector processes.
    if args.action == 'ocr' and args.device == 'cuda' and not os.environ.get('SENTINEL_COMPARISON_CUDA12'):
        libraries=sorted((args.ocr_cache/'gpu-runtime/nvidia').glob('*/lib'))
        if not libraries:
            raise RuntimeError('Cached Paddle CUDA12 runtime is unavailable')
        env=dict(os.environ, SENTINEL_COMPARISON_CUDA12='1')
        env['LD_LIBRARY_PATH']=':'.join([*(str(v) for v in libraries),os.environ.get('LD_LIBRARY_PATH','')])
        os.execvpe(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env)
    {'prepare':prepare,'infer':infer,'ocr':ocr,'score':score}[args.action](args)


if __name__=='__main__':main()
