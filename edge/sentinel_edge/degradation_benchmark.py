"""Labelled multi-frame evaluation on explicitly permitted, non-Sentinel media.

Ground truth belongs only to scoring/oracle diagnostic code. It is never used
as a watchlist or supplied to neural inference. Reports contain aggregates,
not images, expected text, predicted text, source URLs or local paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from .temporal import TemporalPlateReader, indian_plate_format

LICENSES = {"CC0-1.0", "CC-BY-4.0", "CC-BY-NC-4.0"}
MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".mp4", ".avi", ".mov", ".mkv", ".webm"}
STAGES = {"end_to_end", "oracle_plate_ocr"}


def _local_media(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Media paths must be relative to the dataset root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
        raise ValueError("Media must exist inside the dataset root")
    return path


def _box(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Invalid annotation box")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in value):
        raise ValueError("Invalid annotation box")
    if value[2] <= value[0] or value[3] <= value[1]:
        raise ValueError("Empty annotation box")
    return list(value)


def validate_manifest(document: dict, root: Path) -> list[dict]:
    dataset = document.get("dataset", {})
    for key in ("name", "source_url", "license", "license_url", "permitted_use"):
        if not isinstance(dataset.get(key), str) or not dataset[key]:
            raise ValueError("Missing explicit dataset provenance or permission")
    if dataset["license"] not in LICENSES or dataset["permitted_use"] != "noncommercial_benchmark":
        raise ValueError("Dataset permission is not verified for this benchmark")
    for key in ("source_url", "license_url"):
        parsed = urlparse(dataset[key])
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not host or parsed.username or parsed.password:
            raise ValueError("Invalid public provenance URL")
        if "sentinel" in host or host.endswith("corp8.cloud"):
            raise ValueError("Sentinel media retention is forbidden")
    passages = document.get("passages")
    if not isinstance(passages, list) or not passages:
        raise ValueError("No labelled passages supplied")
    ids, vehicle_splits, plate_splits = set(), {}, {}
    for item in passages:
        if not isinstance(item, dict):
            raise ValueError("Invalid passage")
        for key in ("id", "vehicle_id", "ground_truth"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ValueError("Missing passage identity or verified label")
        if item["id"] in ids:
            raise ValueError("Duplicate passage ID")
        ids.add(item["id"])
        split = item.get("split")
        if split not in {"development", "test"}:
            raise ValueError("A development/test split is required")
        if item["vehicle_id"] in vehicle_splits and vehicle_splits[item["vehicle_id"]] != split:
            raise ValueError("The same vehicle cannot occur across splits")
        vehicle_splits[item["vehicle_id"]] = split
        if item.get("country") != "IN":
            raise ValueError("This benchmark targets Indian plates only")
        text = item["ground_truth"]
        if not text.isascii() or not text.isalnum() or text != text.upper():
            raise ValueError("Ground truth must be independently normalized")
        if text in plate_splits and plate_splits[text] != split:
            raise ValueError("The same plate cannot occur across splits under different vehicle IDs")
        plate_splits[text] = split
        kind = item.get("kind")
        if kind == "sequence":
            if item.get("timing_mode") not in {"verified_pts", "assigned_replay"}:
                raise ValueError("Sequence timing provenance must be explicit")
            frames = item.get("frames")
            if not isinstance(frames, list) or len(frames) < 3:
                raise ValueError("A temporal passage needs at least three actual frames")
            pts = []
            for frame in frames:
                _local_media(root, frame.get("path"))
                stamp = frame.get("pts_ms")
                if not isinstance(stamp, int) or isinstance(stamp, bool) or stamp < 0:
                    raise ValueError("Invalid presentation timestamp")
                pts.append(stamp)
                for name in ("plate_box", "vehicle_box"):
                    _box(frame.get(name))
            if any(b <= a for a, b in zip(pts, pts[1:])):
                raise ValueError("Sequence timestamps must be strictly increasing")
        elif kind == "video":
            _local_media(root, item.get("path"))
            start, end = item.get("start_ms", 0), item.get("end_ms")
            if (not isinstance(start, int) or isinstance(start, bool) or
                    not isinstance(end, int) or isinstance(end, bool) or not 0 <= start < end):
                raise ValueError("A bounded video passage is required")
            for annotation in item.get("annotations", []):
                if not isinstance(annotation.get("pts_ms"), int) or isinstance(annotation.get("pts_ms"), bool):
                    raise ValueError("Invalid annotation timestamp")
                for name in ("plate_box", "vehicle_box"):
                    _box(annotation.get(name))
        else:
            raise ValueError("Only genuine video or temporal image sequences are supported")
    return passages


def validate_sequence(timestamps: list[int], fingerprints: list[str]) -> None:
    if len(timestamps) != len(fingerprints) or len(timestamps) < 3:
        raise ValueError("Too few temporal observations")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Temporal observations must have distinct ordered timestamps")
    if len(set(fingerprints)) < 3:
        raise ValueError("Repeated still images are not independent video observations")


def score_passage(expected: set[str], predicted: set[str]) -> dict:
    if not expected:
        raise ValueError("A verified label is required for scoring")
    outcome = "abstain" if not predicted else "correct" if predicted == expected else "wrong"
    return {"outcome": outcome, "correct": int(outcome == "correct"),
            "wrong": int(outcome == "wrong"), "abstain": int(outcome == "abstain")}


def summarize_outcomes(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["variant"], row["split"])].append(row)
    output = []
    for (stage, variant, split), group in groups.items():
        counts = {key: sum(row.get(key, 0) for row in group) for key in ("correct", "wrong", "abstain")}
        total = sum(counts.values())
        accepted = counts["correct"] + counts["wrong"]
        output.append({"stage": stage, "variant": variant, "split": split, "passages": len(group),
                       **counts, "exact_accuracy": counts["correct"] / total if total else None,
                       "accepted_precision": counts["correct"] / accepted if accepted else None,
                       "coverage": accepted / total if total else None,
                       "probability_observations": sum(row.get("probability_observations", 0) for row in group),
                       "median_passage_processing_ms": statistics.median(row.get("duration_ms", 0) for row in group),
                       "failure_reasons": dict(_reason_counts(group))})
    return {"groups": output, "confidence_is_accuracy": False,
            "independent_unit": "labelled vehicle passage, not frame or degraded copy"}


def _reason_counts(rows):
    result = defaultdict(int)
    for row in rows:
        if row.get("failure_reason"):
            result[row["failure_reason"]] += 1
    return result


def degradation_levels() -> list[dict]:
    return ([{"name": f"resolution_{scale:g}", "family": "resolution", "scale": scale} for scale in
             (1.0, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125)] +
            [{"name": f"blur_{kernel}", "family": "blur", "blur_kernel": kernel} for kernel in (3, 5, 7, 9)] +
            [{"name": f"motion_{kernel}", "family": "motion", "motion_kernel": kernel} for kernel in (3, 5, 9, 15)] +
            [{"name": f"noise_{sigma}", "family": "noise", "noise_sigma": sigma} for sigma in (5, 15, 30)] +
            [{"name": f"darkness_{brightness:g}", "family": "darkness", "brightness": brightness}
             for brightness in (0.75, 0.5, 0.25)])


def _frames(root, passage):
    """Read distinct source frames; assigned replay timing is explicitly labelled."""
    import cv2
    if passage["kind"] == "sequence":
        for item in passage["frames"]:
            image = cv2.imread(str(_local_media(root, item["path"])))
            if image is None:
                raise ValueError("Dataset frame failed decoding")
            yield image, item["pts_ms"], item
        return
    import av
    annotations = passage.get("annotations", [])
    with av.open(str(_local_media(root, passage["path"]))) as container:
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError("Dataset video has no presentation timestamps")
            pts = round(float(frame.pts * frame.time_base) * 1000)
            if pts < passage.get("start_ms", 0):
                continue
            if pts >= passage["end_ms"]:
                break
            annotation = min(annotations, key=lambda a: abs(a["pts_ms"] - pts)) if annotations else {}
            if annotation and abs(annotation["pts_ms"] - pts) > 100:
                annotation = {}
            yield frame.to_ndarray(format="bgr24"), pts, annotation


def degrade(image, spec: dict, seed: int):
    import cv2
    import numpy as np
    scale = spec.get("scale", 1.0)
    height, width = image.shape[:2]
    result = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                        interpolation=cv2.INTER_AREA) if scale != 1 else image.copy()
    if spec.get("blur_kernel"):
        kernel = spec["blur_kernel"]
        result = cv2.GaussianBlur(result, (kernel, kernel), 0)
    if spec.get("motion_kernel"):
        size = spec["motion_kernel"]
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1 / size
        result = cv2.filter2D(result, -1, kernel)
    values = result.astype(np.float32) * spec.get("brightness", 1.0)
    if spec.get("noise_sigma"):
        values += np.random.default_rng(seed).normal(0, spec["noise_sigma"], values.shape)
    return np.clip(values, 0, 255).astype(np.uint8)


def _scaled_box(box, before, after):
    if box is None:
        return None
    sx, sy = after[1] / before[1], after[0] / before[0]
    return (round(box[0] * sx), round(box[1] * sy), round(box[2] * sx), round(box[3] * sy))


def _target_result(observation, annotation):
    return _target_box(observation.vehicle_bbox, annotation)


def _target_box(vehicle, annotation):
    target = annotation.get("vehicle_box")
    if vehicle is None or target is None:
        return False
    intersection = max(0, min(vehicle[2], target[2]) - max(vehicle[0], target[0])) * max(
        0, min(vehicle[3], target[3]) - max(vehicle[1], target[1]))
    union = ((vehicle[2] - vehicle[0]) * (vehicle[3] - vehicle[1]) +
             (target[2] - target[0]) * (target[3] - target[1]) - intersection)
    return union > 0 and intersection / union >= 0.5


def validate_annotation_bounds(annotation, shape):
    height, width = shape[:2]
    for name in ("plate_box", "vehicle_box"):
        box = _box(annotation.get(name))
        if box is not None and (box[2] > width or box[3] > height):
            raise ValueError("Annotation exceeds decoded image bounds")


def run_degradation_benchmark(config, manifest_path: Path, *, family="resolution", stages=None,
                              spacing_ms=300, output_path: Path | None = None, ocr_backend="cct") -> dict:
    """Process labelled passages with temporal recognition ONLY; no central writes."""
    from .vision import VisionPipeline
    document = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    root = manifest_path.parent.resolve()
    passages = validate_manifest(document, root)
    stages = stages or ["end_to_end", "oracle_plate_ocr"]
    if not set(stages) <= STAGES:
        raise ValueError("Invalid benchmark stage")
    specs = [spec for spec in degradation_levels() if spec["family"] == family]
    if not specs:
        raise ValueError("Unknown degradation family")
    pipeline = VisionPipeline(config.model_cache_dir / "yolov8n.pt", ocr_spacing_ms=spacing_ms)
    if ocr_backend == "parseq":
        from .parseq_experiment import PARSeqExperiment
        experiment = PARSeqExperiment(config.model_cache_dir / "experiments")
        pipeline._probabilities = experiment.probabilities
        pipeline.alpr.ocr.ocr_model.config = experiment.config
    elif ocr_backend != "cct":
        raise ValueError("Unknown experimental OCR backend")
    # Warm up neural runtime, then clear all passage-dependent state.
    import cv2
    import numpy as np
    pipeline.model.track(np.zeros((480, 640, 3), dtype=np.uint8), persist=True,
                         tracker="bytetrack.yaml", classes=pipeline.VEHICLE_CLASSES, verbose=False)
    cfg = pipeline.alpr.ocr.ocr_model.config
    rows = []
    for index, passage in enumerate(passages):
        material, byte_count, previous = [], 0, None
        for image, pts, annotation in _frames(root, passage):
            # Sample once before storing; all variants see the exact same frames.
            if previous is not None and pts - previous < 1000 / config.inference_fps:
                continue
            previous = pts
            validate_annotation_bounds(annotation, image.shape)
            byte_count += image.nbytes
            if len(material) >= 1800 or byte_count > 384 * 1024 * 1024:
                raise ValueError("Passage exceeds bounded benchmark memory; shorten the annotated passage")
            material.append((image, pts, annotation))
        validate_sequence([pts for _, pts, _ in material],
                          [hashlib.sha256(image.tobytes()).hexdigest() for image, _, _ in material])
        # Video-derived exports may have different crop sizes. Pad, never stretch,
        # to one canvas so tracker coordinates and aspect ratio remain comparable.
        canvas_h = max(image.shape[0] for image, _, _ in material)
        canvas_w = max(image.shape[1] for image, _, _ in material)
        material = [(cv2.copyMakeBorder(image, 0, canvas_h - image.shape[0],
                                       0, canvas_w - image.shape[1], cv2.BORDER_CONSTANT), pts, annotation)
                    for image, pts, annotation in material]
        if any(not frame.get("vehicle_box") for _, _, frame in material):
            raise ValueError("Independent vehicle boxes are required; cannot score unlabelled distractors")
        for stage in stages:
            if stage == "oracle_plate_ocr" and any(not frame.get("plate_box") for _, _, frame in material):
                continue
            for spec in specs:
                pipeline.reset()
                pipeline.last_plate_seen.clear()
                pipeline.metrics.clear()
                reader = TemporalPlateReader(cfg.alphabet, cfg.pad_char)
                predicted = set()
                sampled = 0
                last_sample = last_ocr = None
                plate_widths, plate_heights = [], []
                target_observations = 0
                target_tracks = {}
                started = time.perf_counter()
                for frame_index, (original, pts, annotation) in enumerate(material):
                    if last_sample is not None and pts - last_sample < 1000 / config.inference_fps:
                        continue
                    last_sample = pts
                    sampled += 1
                    image = degrade(original, spec, seed=index * 100000 + frame_index)
                    scaled = {key: _scaled_box(annotation.get(key), original.shape, image.shape)
                              for key in ("plate_box", "vehicle_box")}
                    if scaled["plate_box"]:
                        box = scaled["plate_box"]
                        plate_widths.append(box[2] - box[0])
                        plate_heights.append(box[3] - box[1])
                    if stage == "end_to_end":
                        for observation in pipeline.process(image, pts_ms=pts):
                            if _target_result(observation, scaled):
                                predicted.add(observation.plate)
                        for key, track in pipeline.tracks.items():
                            if track.last_bbox_pts == pts and _target_box(track.vehicle_bbox, scaled):
                                target_tracks[key] = track
                        if len(target_tracks) > 128:
                            del target_tracks[next(iter(target_tracks))]
                    elif last_ocr is None or pts - last_ocr >= spacing_ms:
                        last_ocr = pts
                        x1, y1, x2, y2 = scaled["plate_box"]
                        h, w = image.shape[:2]
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                        if x2 - x1 < 20 or y2 - y1 < 6:
                            continue
                        matrix = pipeline._probabilities(image[y1:y2, x1:x2])
                        if reader.add(matrix, timestamp_ms=pts):
                            target_observations += 1
                        accepted = reader.accepted(config.ocr_min_confidence)
                        if accepted:
                            predicted.add(accepted.plate)
                duration = (time.perf_counter() - started) * 1000
                scored = score_passage({passage["ground_truth"]}, predicted)
                matrices = target_observations if stage == "oracle_plate_ocr" else sum(t.probability_observations for t in target_tracks.values())
                readers = [reader] if stage == "oracle_plate_ocr" else [t.reader for t in target_tracks.values() if t.reader]
                in_beam = any(any(candidate.plate == passage["ground_truth"] for candidate in candidate_reader.candidates())
                              for candidate_reader in readers)
                best_candidates = [r.candidates()[0] for r in readers if r.candidates()]
                best = max(best_candidates, key=lambda c: c.score) if best_candidates else None
                true_is_best = any(c.plate == passage["ground_truth"] for c in best_candidates)
                reason = None
                if scored["outcome"] != "correct":
                    if len(passage["ground_truth"]) > cfg.max_plate_slots:
                        reason = "unsupported_model_length"
                    elif not indian_plate_format(passage["ground_truth"]):
                        reason = "unsupported_registration_format"
                    elif predicted:
                        reason = "wrong_accepted_identifier"
                    elif stage == "end_to_end" and not target_tracks:
                        reason = "vehicle_detection_or_tracking"
                    elif stage == "end_to_end" and not any(t.plate_regions for t in target_tracks.values()):
                        reason = "plate_localization"
                    elif not any(r.observation_count >= 3 for r in readers):
                        reason = "insufficient_temporal_evidence"
                    elif not in_beam:
                        reason = "true_identifier_absent_from_candidate_beam"
                    elif not true_is_best:
                        reason = "candidate_ranking_ambiguity"
                    else:
                        reason = "acceptance_evidence_below_threshold"
                row = {"passage_index": index, "stage": stage, "variant": spec["name"], "split": passage["split"],
                       **scored, "failure_reason": reason, "duration_ms": round(duration, 2),
                       "sampled_frames": sampled, "probability_observations": matrices,
                       "timing_mode": passage.get("timing_mode", "video_pts"),
                       "target_tracks": len(target_tracks),
                       "ground_truth_in_final_candidate_beam": in_beam,
                       "ground_truth_is_final_top_candidate": true_is_best,
                       "best_candidate_score": best.score if best else None,
                       "best_candidate_weakest_character": best.minimum_character_score if best else None,
                       "plate_width_median_px": statistics.median(plate_widths) if plate_widths else None,
                       "plate_height_median_px": statistics.median(plate_heights) if plate_heights else None,
                       "pipeline_counters": dict(pipeline.metrics)}
                rows.append(row)
                # Aggregate progress only: no labels, source paths or OCR text.
                print(json.dumps({"passage_index": index, "stage": stage, "variant": spec["name"],
                                  "outcome": scored["outcome"], "failure_reason": reason}), flush=True)
        material.clear()
    report = {"evaluation": "multi-frame only", "dataset_license": document["dataset"]["license"],
              "ocr_backend": ocr_backend,
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "vision_model_version": pipeline.model_version,
              "detector_model": pipeline.detector_model_name,
              "ocr_model": "PARSeq" if ocr_backend == "parseq" else pipeline.ocr_model_name,
              "inference_fps": config.inference_fps,
              "runtime_versions": {name: version(name) for name in ("torch", "onnxruntime-gpu", "ultralytics", "fast-alpr")},
              "acceptance": {"mean_character_score": config.ocr_min_confidence,
                             "weakest_character_score": 0.75, "min_observations": 3,
                             "repeated_character_support": 0.60, "min_log_margin": 0.5},
              "passages": len(passages), "ocr_spacing_ms": spacing_ms, "model_slots": cfg.max_plate_slots,
              "family": family, "oracle_stage_is_not_product_performance": True,
              "universal_quality_cutoff_claimed": False, "rows": rows, **summarize_outcomes(rows)}
    report["timing_provenance"] = sorted({row["timing_mode"] for row in rows})
    report["live_timing_validated"] = False  # Offline replay never proves live timing.
    if ocr_backend == "parseq":
        from .parseq_experiment import REVISION
        report["ocr_revision"] = REVISION
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"evaluation": report["evaluation"], "ocr_backend": ocr_backend,
                      "family": family, "passages": len(passages),
                      "conditions": len(rows), "correct": sum(r["correct"] for r in rows),
                      "wrong": sum(r["wrong"] for r in rows),
                      "abstain": sum(r["abstain"] for r in rows),
                      "timing_provenance": report["timing_provenance"]}))
    return report
