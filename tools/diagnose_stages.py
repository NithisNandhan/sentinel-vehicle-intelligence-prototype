"""Stage-by-stage check on labelled instants of a recording.

For each labelled plate box this answers three separate questions with the
unchanged models:

  1. Vehicle: did the vehicle detector box the vehicle carrying this plate?
  2. Plate:   did the plate detector find the plate inside that vehicle box?
  3. OCR:     does the OCR read the plate, from the detector's own crop and
              from a crop cut exactly at the labelled box?

It also OCRs the top and bottom halves of the labelled crop separately, to
test whether two-row plates are the problem. Output is a JSON report and
annotated images, written under an ignored data folder. Single frames only:
this is a diagnosis, not the multi-frame acceptance path and not an accuracy
measurement.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'edge'))


def iou(a, b):
    w, h = min(a[2], b[2])-max(a[0], b[0]), min(a[3], b[3])-max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.
    return w*h/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-w*h)


def edit_distance(a, b):
    row = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j]+1, row[j-1]+1, previous+(x != y))
    return row[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--split', default='development')
    parser.add_argument('--recordings-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('/models/experiments/awiros'))
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('/models/experiments/awiros/adaptations/indian-corpus-encoder.safetensors'))
    parser.add_argument('--detector-profile', default='general')
    parser.add_argument('--device', choices=('cpu', 'gpu'), default=os.getenv('SENTINEL_OCR_DEVICE', 'gpu'))
    args = parser.parse_args()

    import av
    import cv2
    from sentinel_edge.config import Config
    from sentinel_edge.ctc_fusion import greedy_text
    from sentinel_edge.experimental_detectors import COCO_CLASSES
    from sentinel_edge.live_detection_viewer import clipped_box, crop_box, load_models
    from sentinel_edge.plate_search import TargetedPlateSearch
    from sentinel_edge.recorded_evaluation import load_labels, normalise
    from sentinel_edge.recorded_source import RecordingLibrary

    labelled, _ = load_labels(args.labels, split=args.split)
    recordings, _ = RecordingLibrary(args.recordings_dir).scan()
    model, detector, vehicle = load_models(Config.from_env(live=False), args.cache, args.checkpoint,
                                           detector_profile=args.detector_profile, device=args.device)
    search = TargetedPlateSearch(detector)
    names = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def read(crop):
        if crop is None or crop.shape[0] < 8 or crop.shape[1] < 24:
            return None
        return normalise(greedy_text(model.probabilities(crop), model.tokens))

    cases = []
    for recording_id, entry in labelled.items():
        recording = recordings[recording_id]
        wanted = sorted((box['position_ms'], passage, box) for passage in entry['passages'] if passage.readable
                        for box in passage.boxes if box['plate'])
        with av.open(str(recording.path)) as container:
            stream = next(s for s in container.streams if s.type == 'video')
            for frame in container.decode(stream):
                if not wanted:
                    break
                position = round(float(frame.pts*frame.time_base)*1000)-recording.start_pts_ms
                if position < wanted[0][0]:
                    continue
                _, passage, box = wanted.pop(0)
                image = frame.to_ndarray(format='bgr24')
                truth = passage.plate
                label = tuple(int(v) for v in box['plate'])
                centre = ((label[0]+label[2])/2, (label[1]+label[3])/2)
                # 1. Vehicle detection, same model, classes and threshold as the session.
                result = vehicle.model.predict(image, imgsz=640, classes=list(COCO_CLASSES), conf=.25,
                                               device='cpu', verbose=False)[0]
                vehicles = [(clipped_box(image, b), names.get(int(c), str(int(c))), round(float(s), 3))
                            for b, c, s in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())]
                holding = [v for v in vehicles if v[0][0] <= centre[0] <= v[0][2] and v[0][1] <= centre[1] <= v[0][3]]
                holding.sort(key=lambda v: (v[0][2]-v[0][0])*(v[0][3]-v[0][1]))
                # 2. Plate detection inside each vehicle box that holds the labelled plate.
                best = None
                for vbox, vclass, vscore in holding:
                    region = crop_box(image, vbox)
                    local = search.box(region)
                    if local is None:
                        continue
                    local = clipped_box(region, local)
                    found = (local[0]+vbox[0], local[1]+vbox[1], local[2]+vbox[0], local[3]+vbox[1])
                    overlap = iou(found, label)
                    if best is None or overlap > best['iou']:
                        best = {'box': found, 'iou': round(overlap, 3), 'vehicle_class': vclass,
                                'vehicle_box': vbox, 'crop': crop_box(image, found)}
                whole = detector.box(image)
                whole = clipped_box(image, whole) if whole is not None else None
                # 3. OCR on the detector's crop, on the labelled crop, and on its two halves.
                manual = crop_box(image, label)
                detected_text = read(best['crop']) if best else None
                manual_text = read(manual)
                half = manual.shape[0]//2
                rows = [read(manual[:half+2]), read(manual[half-2:])]
                two_row_text = ''.join(t or '' for t in rows) or None
                case = {
                    'passage_id': passage.passage_id, 'position_ms': position, 'truth': truth,
                    'labelled_plate_size_px': [label[2]-label[0], label[3]-label[1]],
                    'vehicle': {'boxes_in_frame': len(vehicles), 'boxes_holding_the_plate': len(holding),
                                'detected': bool(holding),
                                'classes': [f'{c} {s}' for _, c, s in holding]},
                    'plate': {'found_in_vehicle_box': best is not None, 'iou_with_label': best['iou'] if best else None,
                              'matches_label': bool(best and best['iou'] >= .5),
                              'found_in_whole_frame_iou': round(iou(whole, label), 3) if whole else None},
                    'ocr': {
                        'on_detector_crop': detected_text,
                        'on_labelled_crop': manual_text,
                        'on_labelled_crop_rows_separately': two_row_text,
                        'exact_on_detector_crop': detected_text == truth,
                        'exact_on_labelled_crop': manual_text == truth,
                        'exact_rows_separately': two_row_text == truth,
                        'character_errors_on_labelled_crop': edit_distance(manual_text or '', truth),
                        'character_errors_rows_separately': edit_distance(two_row_text or '', truth),
                    },
                }
                cases.append(case)
                # Annotated full frame (downscaled) and enlarged crops for a person to check.
                view = image.copy()
                for vbox, _, _ in vehicles:
                    cv2.rectangle(view, vbox[:2], vbox[2:], (90, 220, 90), 4)
                for vbox, _, _ in holding:
                    cv2.rectangle(view, vbox[:2], vbox[2:], (0, 140, 255), 8)
                cv2.rectangle(view, label[:2], label[2:], (255, 0, 255), 6)
                if best:
                    cv2.rectangle(view, best['box'][:2], best['box'][2:], (0, 210, 255), 6)
                scale = 1600/view.shape[1]
                cv2.imwrite(str(args.output_dir/f'{passage.passage_id}-frame.jpg'),
                            cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
                cv2.imwrite(str(args.output_dir/f'{passage.passage_id}-labelled-crop.png'),
                            cv2.resize(manual, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
                if best:
                    cv2.imwrite(str(args.output_dir/f'{passage.passage_id}-detector-crop.png'),
                                cv2.resize(best['crop'], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    summary = {
        'cases': len(cases),
        'vehicle_detected': sum(c['vehicle']['detected'] for c in cases),
        'plate_found_in_vehicle': sum(c['plate']['found_in_vehicle_box'] for c in cases),
        'plate_box_matches_label': sum(c['plate']['matches_label'] for c in cases),
        'ocr_exact_on_detector_crop': sum(c['ocr']['exact_on_detector_crop'] for c in cases),
        'ocr_exact_on_labelled_crop': sum(c['ocr']['exact_on_labelled_crop'] for c in cases),
        'ocr_exact_rows_read_separately': sum(c['ocr']['exact_rows_separately'] for c in cases),
    }
    report = {'notice': 'Single-frame diagnosis with the unchanged models; not the multi-frame acceptance path, '
                        'not an accuracy measurement. Colours in the frame images: green = every vehicle box, '
                        'orange = vehicle boxes holding the labelled plate, magenta = labelled plate, '
                        'yellow = plate box the detector found.',
              'summary': summary, 'cases': cases}
    (args.output_dir/'stage-diagnosis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
