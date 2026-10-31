"""Run one isolated recogniser on frozen licensed diagnostic inputs, without training."""
import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

from sentinel_edge.ctc_fusion import CTCPlateReader, greedy_text
from sentinel_edge.indian_ocr_finetune import edit_distance
from sentinel_edge.reconstruction import native_crop


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalise(text):
    return re.sub("[^A-Z0-9]", "", text.upper())


def validate_samples(document, root, split):
    selected = [row for row in document["samples"] if row["split"] == split]
    if not 1 <= len(selected) <= 100:
        raise ValueError("A bounded existing holdout is required")
    seen, labels = set(), set()
    for row in selected:
        filename = (root / row["crop_path"]).resolve()
        if not filename.is_relative_to(root.resolve()) or digest(filename) != row["crop_sha256"]:
            raise ValueError("Modified diagnostic crop or unsafe path")
        if row["id"] in seen or row["quarantined"] or not row.get("label_tier"):
            raise ValueError("Duplicate, quarantined or untiered diagnostic sample")
        if normalise(row["transcription"]) != row["transcription"]:
            raise ValueError("Expected canonical alphanumeric registration")
        seen.add(row["id"])
        labels.add(row["transcription"])
    other = {row["transcription"] for row in document["samples"] if row["split"] != split}
    if labels & other:
        raise ValueError("Registration leakage across splits")
    return selected


def timed(reader, image):
    sync = getattr(reader, "synchronize", lambda: None)
    sync()
    begin = time.perf_counter()
    probabilities = reader.probabilities(image)
    sync()
    return probabilities, (time.perf_counter() - begin) * 1000


def run(args):
    import cv2
    import numpy as np
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = validate_samples(document, args.manifest.parent, args.split)
    sequence = json.loads(args.sequence.read_text(encoding="utf-8-sig"))
    if sequence["dataset"].get("license") != "CC-BY-4.0" or sequence["dataset"].get("permitted_use") != "noncommercial_benchmark":
        raise ValueError("Explicit licensed sequence permission required")
    begin = time.perf_counter()
    if args.backend == "svtrv2":
        from sentinel_edge.svtrv2_experiment import SVTRv2Experiment, REVISION, WEIGHTS_SHA256
        reader = SVTRv2Experiment(args.svtr_root, device="gpu", policy=args.policy)
        model = {"name": "SVTRv2-B", "source_revision": REVISION, "weights_sha256": WEIGHTS_SHA256,
                 "preprocessing": args.policy, "parameters": reader.parameters,
                 "omitted_auxiliary_training_keys": len(reader.omitted_auxiliary_keys)}
    else:
        from sentinel_edge.awiros_experiment import AwirosExperiment, WEIGHTS_SHA256
        checkpoint = args.awiros_cache / "adaptations/indian-corpus-encoder.safetensors" if args.backend == "awiros-adapted" else None
        reader = AwirosExperiment(args.awiros_cache, device="gpu", checkpoint=checkpoint)
        model = {"name": args.backend, "weights_sha256": digest(checkpoint) if checkpoint else WEIGHTS_SHA256,
                 "preprocessing": "publisher-native-48x320"}
    loading_ms = (time.perf_counter() - begin) * 1000
    for _ in range(3):
        timed(reader, np.zeros((32, 128, 3), dtype=np.uint8))
    if args.backend == "svtrv2":
        reader.torch.cuda.reset_peak_memory_stats()
    elif hasattr(reader.paddle.device.cuda, "reset_max_memory_allocated"):
        reader.paddle.device.cuda.reset_max_memory_allocated()
    results, durations = [], []
    for row in samples:
        image = cv2.imread(str(args.manifest.parent / row["crop_path"]))
        if image is None:
            raise ValueError("Cannot decode a diagnostic image")
        probabilities, elapsed = timed(reader, image)
        raw = greedy_text(probabilities, reader.tokens)
        reading, truth = normalise(raw), row["transcription"]
        results.append({"id": row["id"], "crop_path": row["crop_path"], "crop_sha256": row["crop_sha256"],
                        "label_tier": row["label_tier"], "expected": truth, "raw_reading": raw,
                        "reading": reading, "exact": reading == truth, "character_errors": edit_distance(reading, truth),
                        "native_shape": list(image.shape[:2]), "ocr_ms": round(elapsed, 3)})
        durations.append(elapsed)
    sequence_results = []
    for passage in sequence["passages"]:
        if passage.get("kind") != "sequence" or not 3 <= len(passage["frames"]) <= 8:
            raise ValueError("Real distinct-frame diagnostic sequence required")
        temporal = CTCPlateReader(reader.tokens)
        frames, original_ids = [], set()
        for frame in passage["frames"]:
            filename = (args.sequence.parent / frame["path"]).resolve()
            if not filename.is_relative_to(args.sequence.parent.resolve()):
                raise ValueError("Unsafe sequence frame path")
            image = cv2.imread(str(filename))
            if image is None:
                raise ValueError("Cannot decode a sequence frame")
            source_id = hashlib.sha256(image.tobytes()).hexdigest()
            if source_id in original_ids:
                raise ValueError("Repeated source frame")
            original_ids.add(source_id)
            crop = native_crop(image, frame["plate_box"])
            probabilities, elapsed = timed(reader, crop)
            temporal.add(probabilities, timestamp_ms=frame["pts_ms"], source_id=source_id)
            frames.append({"pts_ms": frame["pts_ms"], "source_sha256": digest(filename),
                           "native_shape": list(crop.shape[:2]), "reading": normalise(greedy_text(probabilities, reader.tokens)),
                           "ocr_ms": round(elapsed, 3)})
        start_fusion = time.perf_counter()
        candidates, accepted = temporal.candidates(), temporal.accepted(.85)
        truth = passage["ground_truth"]
        sequence_results.append({"id": passage["id"], "split": passage["split"], "expected": truth,
                                 "timing_mode": passage["timing_mode"], "frames": frames,
                                 "top1": candidates[0].plate if candidates else None,
                                 "top1_exact": bool(candidates and candidates[0].plate == truth),
                                 "truth_rank": next((i for i, candidate in enumerate(candidates, 1) if candidate.plate == truth), None),
                                 "accepted": accepted.plate if accepted else None,
                                 "correct_accepted": bool(accepted and accepted.plate == truth),
                                 "wrong_accepted": bool(accepted and accepted.plate != truth),
                                 "fusion_ms": round((time.perf_counter() - start_fusion) * 1000, 3),
                                 "candidates": [{"plate": c.plate, "score": c.score, "log_score": c.log_score,
                                                 "minimum_character_score": c.minimum_character_score} for c in candidates[:5]]})
    peak = (reader.torch.cuda.max_memory_allocated() if args.backend == "svtrv2"
            else reader.paddle.device.cuda.max_memory_allocated())
    total_chars = sum(len(row["expected"]) for row in results)
    report = {"schema_version": 1, "model": model, "device": "gpu", "loading_ms": round(loading_ms, 2),
              "manifest_sha256": digest(args.manifest), "sequence_manifest_sha256": digest(args.sequence),
              "split": args.split, "evaluation_caveat": document["evaluation_caveat"],
              "stills_are_not_temporal_observations": True, "production_promoted": False,
              "summary": {"samples": len(results), "exact": sum(row["exact"] for row in results),
                          "character_errors": sum(row["character_errors"] for row in results), "characters": total_chars,
                          "median_ocr_ms": round(statistics.median(durations), 3),
                          "p95_ocr_ms": round(float(np.percentile(durations, 95)), 3),
                          "framework_peak_allocated_mib": round(peak / 2**20, 2),
                          "memory_scope": "Framework allocation peak; excludes driver, other frameworks and whole application co-residency"},
              "stills": results, "sequences": sequence_results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"backend": args.backend, "split": args.split, "summary": report["summary"],
                      "sequences": [{k: row[k] for k in ("id", "top1_exact", "truth_rank", "correct_accepted", "wrong_accepted")} for row in sequence_results]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("svtrv2", "awiros-parent", "awiros-adapted"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--svtr-root", type=Path, default=Path("/research"))
    parser.add_argument("--awiros-cache", type=Path, default=Path("/models/experiments/awiros"))
    parser.add_argument("--policy", choices=("publisher_cli", "dataset_round"), default="publisher_cli")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
