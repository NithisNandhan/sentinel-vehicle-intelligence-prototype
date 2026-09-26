"""Vehicle tracking and multi-frame plate consensus. Images remain in memory."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .privacy import normalize_plate
from .temporal import TemporalPlateReader, indian_plate_format


def preferred_onnx_providers(available: list[str]) -> list[str]:
    """Use CUDA when present without probing an uninstalled TensorRT runtime."""
    providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if name in available]
    if not providers:
        raise RuntimeError("ONNX Runtime has no usable execution provider")
    return providers


@dataclass(frozen=True, repr=False)
class PlateObservation:
    plate: str
    confidence: float
    track_id: str


@dataclass(repr=False)
class _Track:
    attempts: int = 0
    reads: list[tuple[str, float]] = field(default_factory=list)
    last_seen: float = 0.0
    emitted: bool = False
    best_quality: float = 0.0
    reader: TemporalPlateReader | None = None
    last_attempt_ms: int | None = None
    plate_center: tuple[float, float] | None = None
    counted_multi: bool = False


class VisionPipeline:
    """One instance per camera, so ByteTrack state never crosses camera feeds."""

    VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO car, motorcycle, bus, truck.

    def __init__(self, model_path: Path, *, ocr_min_confidence: float = 0.85,
                 dedup_seconds: int = 300, multi_frame: bool = True,
                 ocr_model: str | None = None, detector_model: str | None = None):
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model at {model_path}; cache yolov8n.pt before running live mode")
        try:
            import cv2
            import onnxruntime as ort
            from fast_alpr import ALPR
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Live AI dependencies are unavailable; install edge[live]") from exc
        self.cv2 = cv2
        self.model = YOLO(str(model_path))
        # Importing YOLO loads torch; expose its bundled CUDA/cuDNN libraries to ORT.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        providers = preferred_onnx_providers(ort.get_available_providers())
        self.multi_frame = multi_frame
        self.ocr_model_name = ocr_model or ("cct-s-v2-global-model" if multi_frame else "cct-xs-v2-global-model")
        self.detector_model_name = detector_model or ("yolo-v9-t-512-license-plate-end2end" if multi_frame else "yolo-v9-t-384-license-plate-end2end")
        self.alpr = ALPR(detector_model=self.detector_model_name,
                         ocr_model=self.ocr_model_name, ocr_device="auto",
                         detector_providers=providers, ocr_providers=providers)
        self.ocr_min_confidence = ocr_min_confidence
        self.dedup_seconds = dedup_seconds
        self.tracks: dict[str, _Track] = {}
        self.last_plate_seen: dict[str, float] = {}
        self.metrics: dict[str, int | float] = {}
        self.max_tracks = 128
        self.model_version = "yolov8n+fastalpr-cct-s-v2+temporal-v1" if multi_frame else "yolov8n+fastalpr-legacy"

    def _count(self, key: str, value: int = 1) -> None:
        if not hasattr(self, "metrics"):
            self.metrics = {}
        self.metrics[key] = self.metrics.get(key, 0) + value

    def reset(self) -> None:
        self.tracks.clear()
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) if predictor is not None else []:
            tracker.reset()

    def process(self, image: object, *, pts_ms: int | None = None) -> list[PlateObservation]:
        now = time.monotonic()
        self.tracks = {key: state for key, state in self.tracks.items() if now - state.last_seen < 30}
        self.last_plate_seen = {key: seen for key, seen in self.last_plate_seen.items()
                                if now - seen < self.dedup_seconds}
        result = self.model.track(image, persist=True, tracker="bytetrack.yaml",
                                  classes=self.VEHICLE_CLASSES, verbose=False, conf=0.25)[0]
        boxes = result.boxes
        self._count("inference_frames")
        if boxes is None or boxes.id is None:
            return []
        observations: list[PlateObservation] = []
        for box in boxes:
            if box.id is None:
                continue
            track_id = str(int(box.id.item()))
            self._count("vehicle_boxes")
            if track_id not in self.tracks:
                self._count("vehicle_tracks")
                if len(self.tracks) >= getattr(self, "max_tracks", 128):
                    oldest = min(self.tracks, key=lambda key: self.tracks[key].last_seen)
                    del self.tracks[oldest]
            state = self.tracks.setdefault(track_id, _Track())
            state.last_seen = now
            temporal = getattr(self, "multi_frame", False)
            timestamp_ms = pts_ms if pts_ms is not None else round(now * 1000)
            if state.emitted or state.attempts >= (40 if temporal else 5):
                continue
            if temporal and state.last_attempt_ms is not None and timestamp_ms - state.last_attempt_ms < 300:
                continue
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            height, width = image.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 - x1 < (64 if temporal else 120) or y2 - y1 < (32 if temporal else 50):
                continue
            crop = image[y1:y2, x1:x2]
            # Skip heavily blurred crops; the threshold is deliberately conservative.
            blur_score = self.cv2.Laplacian(self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2GRAY),
                                            self.cv2.CV_64F).var()
            if not temporal and blur_score < 35:
                continue
            quality = float(blur_score) * min(x2 - x1, 500) / 500
            if not temporal and state.best_quality and quality < 0.6 * state.best_quality:
                continue
            state.best_quality = max(state.best_quality, quality)
            state.attempts += 1
            state.last_attempt_ms = timestamp_ms
            self._count("alpr_crop_calls")
            if temporal:
                observation = self._temporal_crop(crop, state, track_id, timestamp_ms)
                if observation is not None:
                    state.emitted = True
                    if observation.plate not in self.last_plate_seen:
                        self.last_plate_seen[observation.plate] = now
                        observations.append(observation)
                        self._count("accepted_sightings")
                continue
            results = self.alpr.predict(crop)
            for result in results:
                self._count("plate_regions")
                ocr = result.ocr
                if ocr is None or not ocr.text:
                    continue
                if isinstance(ocr.confidence, list) and not ocr.confidence:
                    continue
                confidence = statistics.mean(ocr.confidence) if isinstance(ocr.confidence, list) else float(ocr.confidence)
                plate = normalize_plate(ocr.text)
                self._count("nonempty_ocr")
                if confidence < self.ocr_min_confidence or not 5 <= len(plate) <= 20:
                    continue
                self._count("high_score_ocr")
                state.reads.append((plate, confidence))
                matches = [score for text, score in state.reads if text == plate]
                if len(matches) >= 2:
                    state.emitted = True
                    if plate not in self.last_plate_seen:
                        self.last_plate_seen[plate] = now
                        observations.append(PlateObservation(plate, statistics.mean(matches), track_id))
                        self._count("accepted_sightings")
                    break
        return observations

    def _probabilities(self, plate_crop):
        """Use the installed CCT's exact preprocessing and full ONNX plate head.

        The public API returns only each slot's winning probability. This narrow
        adapter preserves the full alternatives; integration tests check parity
        with that public API. A changed library contract fails, never fabricates.
        """
        from fast_plate_ocr.inference.plate_recognizer import _load_image_from_source, preprocess_image
        recognizer = self.alpr.ocr.ocr_model
        cfg = recognizer.config
        if cfg.image_color_mode == "grayscale":
            plate_crop = self.cv2.cvtColor(plate_crop, self.cv2.COLOR_BGR2GRAY)
        elif cfg.image_color_mode == "rgb":
            plate_crop = self.cv2.cvtColor(plate_crop, self.cv2.COLOR_BGR2RGB)
        batch = preprocess_image(_load_image_from_source(plate_crop, cfg))
        outputs = recognizer.model.run([recognizer.plate_output_name], {"input": batch})
        return outputs[0].reshape((-1, cfg.max_plate_slots, len(cfg.alphabet)))[0].tolist()

    def _temporal_crop(self, crop, state: _Track, track_id: str, timestamp_ms: int) -> PlateObservation | None:
        detections = self.alpr.detector.predict(crop)
        self._count("plate_regions", len(detections))
        if not detections:
            return None
        # One spatially coherent plate per vehicle/observation, not every box.
        detection = max(detections, key=lambda item: float(item.confidence))
        bbox = detection.bounding_box
        height, width = crop.shape[:2]
        center = ((bbox.x1 + bbox.x2) / (2 * width), (bbox.y1 + bbox.y2) / (2 * height))
        if state.plate_center is not None and sum((a - b) ** 2 for a, b in zip(center, state.plate_center)) > 0.12:
            state.reader = None
            self._count("plate_region_resets")
        state.plate_center = center
        # Small padding is only a crop adjustment, not a super-resolution claim.
        pad_x, pad_y = max(1, round((bbox.x2 - bbox.x1) * 0.03)), max(1, round((bbox.y2 - bbox.y1) * 0.03))
        x1, y1 = max(0, bbox.x1 - pad_x), max(0, bbox.y1 - pad_y)
        x2, y2 = min(width, bbox.x2 + pad_x), min(height, bbox.y2 + pad_y)
        if x2 - x1 < 20 or y2 - y1 < 6:
            self._count("too_small_plate_regions")
            return None
        plate_crop = crop[y1:y2, x1:x2]
        matrix = self._probabilities(plate_crop)
        cfg = self.alpr.ocr.ocr_model.config
        greedy = "".join(cfg.alphabet[max(range(len(row)), key=row.__getitem__)] for row in matrix).rstrip(cfg.pad_char)
        if greedy:
            self._count("nonempty_ocr")
            mean_score = statistics.mean(max(row) for row in matrix[:len(greedy)])
            self.metrics["max_single_model_score"] = max(self.metrics.get("max_single_model_score", 0.0), mean_score)
            if mean_score >= self.ocr_min_confidence:
                self._count("high_score_ocr")
            if indian_plate_format(greedy):
                self._count("format_valid_ocr")
        if state.reader is None:
            state.reader = TemporalPlateReader(cfg.alphabet, cfg.pad_char)
        if not state.reader.add(matrix, timestamp_ms=timestamp_ms,
                                quality=min(1.0, (x2 - x1) * (y2 - y1) / 2000)):
            return None
        self._count("probability_observations")
        if state.reader.observation_count >= 3:
            if not state.counted_multi:
                state.counted_multi = True
                self._count("multi_frame_tracks")
            candidates = state.reader.candidates()
            if candidates:
                self.metrics["max_fused_model_score"] = max(self.metrics.get("max_fused_model_score", 0.0), candidates[0].score)
        accepted = state.reader.accepted(self.ocr_min_confidence)
        return PlateObservation(accepted.plate, accepted.score, track_id) if accepted else None
