"""Local, labelled recording-crop diagnostics using the serving OCR preprocessing."""
from __future__ import annotations

import argparse
import base64
import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .awiros_experiment import AwirosExperiment, file_hash, input_array
from .ctc_fusion import greedy_text, prefix_candidates, validate_ctc
from .sequence_fusion import token_alignment


def posterior_details(rows, tokens):
    """Greedy CTC emissions and their peak timesteps, without forced plate grammar.

    Alternatives sum upper/lowercase token mass at the same timestep. Scores
    remain model posteriors, never calibrated probabilities of plate correctness.
    """
    validate_ctc(rows, tokens)
    steps = []
    for row in rows:
        grouped = {}
        for token, value in zip(tokens[1:], row[1:]):
            char = token.upper()
            if len(char) == 1 and char.isascii() and char.isalnum():
                grouped[char] = grouped.get(char, 0.) + float(value)
        steps.append({"blank": float(row[0]), "scores": grouped})
    winners = [max(range(len(row)), key=row.__getitem__) for row in rows]
    characters = []
    for token, group in itertools.groupby(enumerate(winners), key=lambda item: item[1]):
        indices = [index for index, _ in group]
        char = tokens[token].upper()
        if not token or len(char) != 1 or not char.isascii() or not char.isalnum():
            continue
        peak = max(indices, key=lambda index: rows[index][token])
        scores = steps[peak]["scores"]
        alternatives = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        characters.append({"character": char, "score": scores[char],
                           "raw_token_score": float(rows[peak][token]), "timestep": peak,
                           "alternatives": [{"character": c, "score": p}
                                            for c, p in alternatives if c != char][:5]})
    reading = "".join(item["character"] for item in characters)
    assert reading == re.sub("[^A-Z0-9]", "", greedy_text(rows, tokens))
    scores = [item["score"] for item in characters]
    candidates = [{"reading": text, "log_score": value}
                  for text, value in prefix_candidates(rows, tokens, beam_width=32)[:5]]
    return {"reading": reading, "characters": characters, "timesteps": steps,
            "plate_score": sum(scores) / len(scores) if scores else None,
            "minimum_score": min(scores) if scores else None, "candidates": candidates}


def compare_label(result, expected):
    """Align only after inference. Missing characters have no invented score."""
    positions, edits = token_alignment(expected, result["reading"])
    aligned = []
    for index, position in enumerate(positions):
        observed = result["characters"][position] if position is not None else None
        support = (result["timesteps"][observed["timestep"]]["scores"].get(expected[index], 0.)
                   if observed else None)
        aligned.append({"position": index + 1, "expected": expected[index],
                        "observed_index": position, "expected_support": support})
    used = {index for index in positions if index is not None}
    return {"aligned": aligned, "extra_indices": [i for i in range(len(result["characters"])) if i not in used],
            "character_errors": edits, "exact": expected == result["reading"],
            "expected_in_candidates": any(c["reading"] == expected for c in result["candidates"])}


def image_url(image):
    import cv2
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode a review image")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def inspect_crop(reader, image):
    import numpy as np
    tensor = input_array(image)
    # Denormalize the *actual* input tensor for a viewable RGB PNG. Inference
    # receives the normalized BGR values, not this display conversion.
    pixels = np.rint((tensor[0].transpose(1, 2, 0) * .5 + .5) * 255).clip(0, 255).astype("uint8")
    rows = reader.probabilities(image)
    details = posterior_details(rows, reader.tokens)
    height, width = image.shape[:2]
    content_width = max(1, min(int(width * 48 / height), 320))
    return {**details, "crop": image_url(image), "model_input": image_url(pixels),
            "source_width": width, "source_height": height, "input_width": 320,
            "input_height": 48, "content_width": content_width,
            "padding_width": 320 - content_width, "tensor_shape": list(tensor.shape),
            "tensor_min": float(tensor.min()), "tensor_max": float(tensor.max())}


def checked_crop(root, relative, expected):
    import cv2
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
        raise ValueError("Crop path or checksum does not match the review manifest")
    image = cv2.imread(str(path))
    if image is None or image.size > 12_000_000:
        raise ValueError("Invalid or oversized review crop")
    return image


def build(args):
    root = args.review_root.resolve()
    load = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
    manifest_path = root / "manifest.json"
    manifest, labels = load(manifest_path), load(root / "labels.json")
    corrected_path = root / "ocr-model-dev/rectified-extended-manifest.json"
    corrected = load(corrected_path)
    manifest_hash = file_hash(manifest_path)
    if (labels["review_manifest_sha256"] != manifest_hash
            or corrected["review_manifest_sha256"] != manifest_hash
            or labels["source_sha256"] != manifest["source_sha256"]):
        raise ValueError("Labels and corrected crops refer to a different review set")
    samples = manifest["samples"]
    ids = [sample["id"] for sample in samples]
    if not 1 <= len(ids) <= 100 or len(set(ids)) != len(ids):
        raise ValueError("Review sample IDs must be unique and bounded")
    truth = {sample["id"]: sample for sample in labels["labels"]}
    revised = {sample["id"]: sample for sample in corrected["samples"]}
    if set(truth) != set(ids) or not set(revised).issubset(ids):
        raise ValueError("Labels do not cover this crop set")
    checkpoint = args.cache / "adaptations/indian-corpus-encoder.safetensors"
    reader = AwirosExperiment(args.cache, device=args.device, checkpoint=checkpoint)
    cases = []
    for sample in samples:
        sid = sample["id"]
        native = checked_crop(root, sample["crop"], sample["crop_sha256"])
        source = inspect_crop(reader, native)
        update = revised.get(sid)
        improved = (inspect_crop(reader, checked_crop(corrected_path.parent, update["crop_path"], update["crop_sha256"]))
                    if update and update["crop_sha256"] != sample["crop_sha256"] else source)
        label = truth[sid]
        expected = re.sub("[^A-Z0-9]", "", (label.get("plate") or "").upper())
        readable = label["status"] == "readable"
        if readable and not expected:
            raise ValueError("Readable label has no transcription")
        if readable:
            source["comparison"] = compare_label(source, expected)
            improved["comparison"] = compare_label(improved, expected)
        cases.append({"id": sid, "position_ms": sample["position_ms"],
                      "human_readable": readable, "expected": expected if readable else None,
                      "changed": improved is not source, "native": source, "corrected": improved})
        print(f"Reviewed crop {sid}", flush=True)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "device": args.device,
              "model": "Adapted Awiros CTC", "checkpoint_sha256": file_hash(checkpoint),
              "review_manifest_sha256": manifest_hash,
              "preprocessing": "BGR input, aspect-ratio resize to height 48 (width capped at 320), right zero padding, values / 127.5 - 1, NCHW float32.",
              "score_definition": "Mean of emitted-character scores. Each score sums upper/lowercase token posteriors at the peak raw-token timestep in its greedy CTC run. Uncalibrated.",
              "cases": cases}
    refinement_path = getattr(args, "refinement_report", None)
    if refinement_path:
        refinement = load(refinement_path)
        if refinement["checkpoint_sha256"] != report["checkpoint_sha256"]:
            raise ValueError("Retry report uses a different OCR checkpoint")
        if refinement["manifest_hashes"]["manual"] != file_hash(corrected_path):
            raise ValueError("Retry report uses different corrected crops")
        selected = refinement["datasets"]["mumbai_manual"]
        by_id = {row["id"]: row for row in selected["results"]}
        for case in cases:
            case["reference_uncertain"] = case["id"] in refinement["excluded_uncertain_manual_ids"]
            if case["id"] in by_id:
                row = by_id[case["id"]]
                if row["baseline"] != case["corrected"]["reading"] or row["expected"] != case["expected"]:
                    raise ValueError("Retry report does not match fresh baseline inference")
                case["refinement"] = row
        report["refinement_summary"] = {k: v for k, v in selected.items() if k != "results"}
        report["refinement_report_sha256"] = file_hash(refinement_path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=True, allow_nan=False)
    template = (Path(__file__).resolve().parents[2] / "tools/ocr_input_review.html").read_text(encoding="utf-8")
    html = template.replace("__REVIEW_DATA__", serialized.replace("<", "\\u003c"))
    (output / "review.json").write_text(serialized + "\n", encoding="utf-8")
    (output / "index.html").write_text(html, encoding="utf-8")
    readable = [case for case in cases if case["human_readable"]]
    print(json.dumps({"reviewed": len(cases), "human_readable": len(readable),
                      "native_exact": sum(c["native"]["comparison"]["exact"] for c in readable),
                      "corrected_exact": sum(c["corrected"]["comparison"]["exact"] for c in readable)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=Path("/models/experiments/awiros"))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--refinement-report", type=Path)
    build(parser.parse_args())
