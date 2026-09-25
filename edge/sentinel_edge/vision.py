"""Vehicle tracking and multi-frame plate consensus. Images remain in memory."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .privacy import normalize_plate


@dataclass(frozen=True)
class PlateObservation:
    plate: str
    confidence: float
    track_id: str


@dataclass
class _Track:
    attempts: int = 0
    reads: list[tuple[str, float]] = field(default_factory=list)
    last_seen: float = 0.0
    emitted: bool = False
    best_quality: float = 0.0


class VisionPipeline:
    """One instance per camera, so ByteTrack state never crosses camera feeds."""

    VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO car, motorcycle, bus, truck.

    def __init__(self, model_path: Path, *, ocr_min_confidence: float = 0.85,
                 dedup_seconds: int = 300):
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model at {model_path}; cache yolov8n.pt before running live mode")
        try:
            import cv2
            from fast_alpr import ALPR
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Live AI dependencies are unavailable; install edge[live]") from exc
        self.cv2 = cv2
        self.model = YOLO(str(model_path))
        self.alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end",
                         ocr_model="cct-xs-v2-global-model", ocr_device="auto")
        self.ocr_min_confidence = ocr_min_confidence
        self.dedup_seconds = dedup_seconds
        self.tracks: dict[str, _Track] = {}
        self.last_plate_seen: dict[str, float] = {}

    def reset(self) -> None:
        self.tracks.clear()
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) if predictor is not None else []:
            tracker.reset()

    def process(self, image: object) -> list[PlateObservation]:
        now = time.monotonic()
        self.tracks = {key: state for key, state in self.tracks.items() if now - state.last_seen < 30}
        self.last_plate_seen = {key: seen for key, seen in self.last_plate_seen.items()
                                if now - seen < self.dedup_seconds}
        result = self.model.track(image, persist=True, tracker="bytetrack.yaml",
                                  classes=self.VEHICLE_CLASSES, verbose=False, conf=0.25)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        observations: list[PlateObservation] = []
        for box in boxes:
            if box.id is None:
                continue
            track_id = str(int(box.id.item()))
            state = self.tracks.setdefault(track_id, _Track())
            state.last_seen = now
            if state.emitted or state.attempts >= 5:
                continue
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            height, width = image.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 - x1 < 120 or y2 - y1 < 50:
                continue
            crop = image[y1:y2, x1:x2]
            # Skip heavily blurred crops; the threshold is deliberately conservative.
            blur_score = self.cv2.Laplacian(self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2GRAY),
                                            self.cv2.CV_64F).var()
            if blur_score < 35:
                continue
            quality = float(blur_score) * min(x2 - x1, 500) / 500
            if state.best_quality and quality < 0.6 * state.best_quality:
                continue
            state.best_quality = max(state.best_quality, quality)
            state.attempts += 1
            results = self.alpr.predict(crop)
            for result in results:
                ocr = result.ocr
                if ocr is None or not ocr.text:
                    continue
                if isinstance(ocr.confidence, list) and not ocr.confidence:
                    continue
                confidence = statistics.mean(ocr.confidence) if isinstance(ocr.confidence, list) else float(ocr.confidence)
                plate = normalize_plate(ocr.text)
                if confidence < self.ocr_min_confidence or not 5 <= len(plate) <= 20:
                    continue
                state.reads.append((plate, confidence))
                matches = [score for text, score in state.reads if text == plate]
                if len(matches) >= 2:
                    state.emitted = True
                    if plate not in self.last_plate_seen:
                        self.last_plate_seen[plate] = now
                        observations.append(PlateObservation(plate, statistics.mean(matches), track_id))
                    break
        return observations
