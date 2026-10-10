"""Cached-only detector adapters for opt-in diagnostics, never production defaults."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .detector_comparison import PlateModel, decode_yolo11, letterbox
from .vision import associate_vehicle_tracks

PROFILES = {
    'legacy': ('yolov8n', 'parent'),
    'general': ('yolo26s', 'parent'),
    'plate_adapted': ('yolo26s', 'adapted'),
    'indian_adapted': ('bmd', 'adapted'),
}
HASHES = {
    'yolov8n': 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36',
    'yolo26s': '646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b',
    'bmd': 'a6d7cb9b39d77ac5f2b494ac4e264454d3d5bc13e6b78d92700671c658a36a5a',
    'parent': '746fdd358ec110418775d7c9d8d07910d48b1a21471f92bf4421f6510d6daade',
    'adapted': 'd0416773dccb3ec8c852390e03a8b6f177850e0f693c2a529fc49c7961525611',
}
# Coarse tracking taxonomy. It does not claim fine subclass accuracy.
BMD_TO_COCO = {0:2, 1:2, 2:2, 3:2, 4:5, 5:7, 6:2, 7:3,
               8:7, 9:5, 10:5, 11:1, 12:2}
COCO_CLASSES = (1, 2, 3, 5, 7)  # Bicycle included consistently in this experiment.


def verified(path, expected):
    path = Path(path)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Diagnostic model missing or changed; no automatic download')
    return path


@dataclass(frozen=True)
class VehicleTrack:
    track_id: str
    box: tuple[int, int, int, int]
    vehicle_class: int
    # The detector's score for this box. Display only; not a plate-reading confidence.
    confidence: float | None = None


class VehicleTracker:
    def __init__(self, name, cache, training_root):
        from ultralytics import YOLO
        import torch
        torch.set_num_threads(2)
        if name not in ('yolov8n', 'yolo26s', 'bmd'):
            raise ValueError('Unknown diagnostic vehicle model')
        path = (Path(training_root)/'runs/vehicle-bmd/weights/best.pt' if name == 'bmd'
                else Path(cache)/('yolov8n.pt' if name == 'yolov8n' else 'experiments/detectors/yolo26s.pt'))
        self.name = name
        self.model = YOLO(str(verified(path, HASHES[name])))
        expected = set(range(13)) if name == 'bmd' else set(range(80))
        if set(self.model.names) != expected:
            raise ValueError('Vehicle taxonomy changed; refuse incompatible class filtering')
        self.frame_index = 0
        self.next_detection_id = 0
        self.previous_tracks = {}

    def reset(self):
        self.frame_index = 0
        self.previous_tracks.clear()
        predictor = getattr(self.model, 'predictor', None)
        for tracker in getattr(predictor, 'trackers', ()):
            tracker.reset()

    def track(self, image):
        self.frame_index = getattr(self,'frame_index',0)+1
        if not hasattr(self,'previous_tracks'):
            self.previous_tracks = {}
        if not hasattr(self,'next_detection_id'):
            self.next_detection_id = 0
        classes = list(BMD_TO_COCO) if self.name == 'bmd' else list(COCO_CLASSES)
        result = self.model.track(image, persist=True, tracker='bytetrack.yaml',
            imgsz=640, classes=classes, device='cpu', conf=.25, iou=.45,
            verbose=False, save=False)[0]
        detections = []
        scores = {}
        for item in (result.boxes if result.boxes is not None else ()):
            cls = int(item.cls.item())
            coarse = BMD_TO_COCO[cls] if self.name == 'bmd' else cls
            coords = tuple(int(v) for v in item.xyxy[0].cpu().tolist())
            score = getattr(item,'conf',None)
            if score is not None:
                scores[coords] = float(score.item())
            identity = getattr(item,'id',None)
            detections.append((str(int(identity.item())) if identity is not None else None,
                               coords,coarse))
        tracks,self.previous_tracks,self.next_detection_id = associate_vehicle_tracks(
            detections,self.previous_tracks,self.frame_index,self.next_detection_id)
        return [VehicleTrack(track_id,coords,coarse,scores.get(coords)) for track_id,coords,coarse in tracks]


class PlateDetector:
    def __init__(self, name, cache):
        import onnxruntime as ort
        root = Path(cache)/'experiments/detectors'
        self.name = name
        if name == 'parent':
            verified(root/'yolo-v9-t-512-license-plates-end2end.onnx', HASHES[name])
            self.model = PlateModel('yolov9t512', root, 'cpu')
            self.session = None
        elif name == 'adapted':
            path = verified(root/'yolov9t512-indian-adapted.onnx', HASHES[name])
            receipt = json.loads(path.with_suffix('.receipt.json').read_text())
            if receipt['input_shape'] != [1,3,512,512] or receipt['output_shape'] != [1,5,5376]:
                raise ValueError('Adapted plate export contract changed')
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
            spec = self.session.get_inputs()[0]
            if spec.shape != [1,3,512,512] or self.session.get_outputs()[0].shape != [1,5,5376]:
                raise ValueError('Runtime plate export shape mismatch')
        else:
            raise ValueError('Unknown diagnostic plate model')

    def boxes(self, image):
        if self.session is None:
            detections = sorted(self.model.model.predict(image),key=lambda d:-float(d.confidence))
            return [[d.bounding_box.x1,d.bounding_box.y1,d.bounding_box.x2,d.bounding_box.y2]
                    for d in detections]
        tensor, scale, left, top = letterbox(image,512,512)
        output = self.session.run(None, {self.session.get_inputs()[0].name:tensor})[0]
        return decode_yolo11(output,scale,left,top,image.shape)

    def box(self, image):
        boxes = self.boxes(image)
        if not boxes:
            return None
        # Both adapters rank proposals by confidence before selecting a crop.
        x1,y1,x2,y2 = boxes[0]
        padx,pady = max(1,round((x2-x1)*.03)), max(1,round((y2-y1)*.03))
        return (max(0,int(x1)-padx),max(0,int(y1)-pady),
                min(image.shape[1],int(x2)+padx),min(image.shape[0],int(y2)+pady))
