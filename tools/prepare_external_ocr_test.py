"""Freeze a separate-source, deduplicated private OCR evaluation before inference."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import xml.etree.ElementTree as ET
import zipfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pixel_hash(image):
    import cv2
    height, width = image.shape[:2]
    return hashlib.sha256(f"{width}x{height}:RGB:".encode() + cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()


def phash(image):
    import cv2
    import numpy as np
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coefficients = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].reshape(-1)
    bits = coefficients > np.median(coefficients[1:])
    bits[0] = False
    return sum(int(bit) << index for index, bit in enumerate(bits))


def normalise_reference(value):
    """Normalise publisher text without inventing a plate grammar or correction.

    This is the frozen v1 selection policy. It does not establish legal validity
    or independently verify that a publisher reference is readable/correct.
    """
    text = re.sub(r"[\s-]", "", value.upper())
    return text if re.fullmatch(r"[A-Z0-9]{5,15}", text) else None


def prepare(root, previous, maximum=200):
    import cv2
    import numpy as np
    if not 1 <= maximum <= 250:
        raise ValueError("External diagnostic cap must be 1..250")
    destination = root / "adaptation_manifest.json"
    if destination.exists():
        raise ValueError("Frozen evaluation exists; do not silently redefine it")
    permission = json.loads((root / "permission.json").read_text())
    if not permission.get("no_training") or permission["purpose"] != "private_noncommercial_evaluation_only":
        raise ValueError("This source is admitted for evaluation only")
    if digest(root / "source.zip") != permission["archive_sha256"]:
        raise ValueError("Changed source archive")
    old = [json.loads(line) for line in (previous / "corpus.jsonl").read_text().splitlines()]
    seen_registrations = {row["transcription"] for row in old}
    seen_pixels = {row["decoded_pixel_sha256"] for row in old}
    seen_originals = {p["original_sha256"] for row in old for p in row["provenance"] if p.get("original_sha256")}
    nivo = previous.parent / "nivu/manifest.json"
    if nivo.exists():
        seen_registrations.update(row["ground_truth"] for row in json.loads(nivo.read_text())["passages"])
    training = [r for r in json.loads((previous / "adaptation_manifest.json").read_text())["samples"] if r["split"] == "train"]
    training_hashes = []
    for row in training:
        filename = previous / row["crop_path"]
        if digest(filename) != row["crop_sha256"]:
            raise ValueError("Training audit inputs changed")
        image = cv2.imread(str(filename))
        if image is None:
            raise ValueError("Cannot decode prior training crop")
        training_hashes.append(phash(image))
    counts, exclusions, observations = Counter(), [], []
    cropped = root / "crops"
    cropped.mkdir(exist_ok=True)
    with zipfile.ZipFile(root / "source.zip") as archive:
        entries = archive.infolist()
        if len(entries) > 6000 or sum(e.file_size for e in entries) > 1024**3:
            raise ValueError("Dataset expansion limit exceeded")
        names = {e.filename for e in entries}
        images = {str(PurePosixPath(name).with_suffix("")): name for name in names if PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png"}}
        for annotation in sorted(n for n in names if n.lower().endswith(".xml")):
            counts["annotation_files"] += 1
            raw = archive.read(annotation)
            if len(raw) > 256_000 or b"<!DOCTYPE" in raw.upper():
                raise ValueError("Unexpected annotation format")
            tree = ET.fromstring(raw)
            image_name = images.get(str(PurePosixPath(annotation).with_suffix("")))
            if image_name is None:
                exclusions.append({"annotation": annotation, "reason": "missing_image"}); continue
            image_bytes = archive.read(image_name)
            if len(image_bytes) > 32 * 1024**2:
                raise ValueError("Oversized source image")
            original_hash = hashlib.sha256(image_bytes).hexdigest()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None:
                exclusions.append({"annotation": annotation, "reason": "cannot_decode"}); continue
            height, width = image.shape[:2]
            if (int(tree.findtext("size/width")), int(tree.findtext("size/height"))) != (width, height):
                exclusions.append({"annotation": annotation, "reason": "annotation_dimensions_mismatch"}); continue
            for index, obj in enumerate(tree.findall("object")):
                counts["annotated_objects"] += 1
                identifier = hashlib.sha256(f"external-v1|{annotation}|{index}".encode()).hexdigest()
                label = normalise_reference(obj.findtext("name", ""))
                reason = "unsupported_reference" if label is None else "seen_registration" if label in seen_registrations else "seen_original_bytes" if original_hash in seen_originals else None
                if reason:
                    exclusions.append({"id": identifier, "reason": reason}); continue
                values = [float(obj.findtext("bndbox/" + key)) for key in ("xmin", "ymin", "xmax", "ymax")]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("Nonfinite publisher box")
                x1, y1 = max(0, math.floor(values[0])), max(0, math.floor(values[1]))
                x2, y2 = min(width, math.ceil(values[2])), min(height, math.ceil(values[3]))
                if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
                    exclusions.append({"id": identifier, "reason": "invalid_plate_box"}); continue
                crop = image[y1:y2, x1:x2].copy()
                decoded_hash = pixel_hash(crop)
                perceptual = phash(crop)
                if decoded_hash in seen_pixels:
                    exclusions.append({"id": identifier, "reason": "seen_crop_pixels"}); continue
                if any((perceptual ^ previous_hash).bit_count() <= 6 for previous_hash in training_hashes):
                    exclusions.append({"id": identifier, "reason": "perceptually_similar_to_training"}); continue
                filename = cropped / (identifier + ".png")
                if not cv2.imwrite(str(filename), crop):
                    raise ValueError("Cannot preserve native crop")
                observations.append({"id": identifier, "crop_path": "crops/" + filename.name,
                                     "crop_sha256": digest(filename), "decoded_pixel_sha256": decoded_hash,
                                     "perceptual_hash": f"{perceptual:016x}", "transcription": label,
                                     "label_tier": "publisher_xml_transcription_not_independently_human_verified",
                                     "width": x2-x1, "height": y2-y1, "quarantined": False,
                                     "split": "test", "source_group": PurePosixPath(annotation).parts[0],
                                     "original_sha256": original_hash, "annotation_sha256": hashlib.sha256(raw).hexdigest(),
                                     "annotation_member": annotation, "object_index": index, "plate_box": [x1,y1,x2,y2],
                                     "provenance": [{"source": "saisirishan", "license": permission["license"],
                                                     "source_url": permission["source_url"], "attribution": permission["attribution"]}]})
    # Quarantine exact pixel groups with contradictory publisher reference text.
    by_pixels = defaultdict(set)
    for row in observations:
        by_pixels[row["decoded_pixel_sha256"]].add(row["transcription"])
    conflicts = {key for key, values in by_pixels.items() if len(values) > 1}
    unique, selected, own_hashes = set(), [], []
    queues = defaultdict(list)
    for row in sorted(observations, key=lambda r:r["id"]):
        if row["decoded_pixel_sha256"] in conflicts:
            exclusions.append({"id": row["id"], "reason": "conflicting_reference_on_same_pixels"}); continue
        if row["transcription"] in unique:
            exclusions.append({"id": row["id"], "reason": "repeated_registration"}); continue
        own = int(row["perceptual_hash"], 16)
        if any((own ^ other).bit_count() <= 6 for other in own_hashes):
            exclusions.append({"id": row["id"], "reason": "near_duplicate_within_new_pool"}); continue
        unique.add(row["transcription"]); own_hashes.append(own)
        queues[row["source_group"]].append(row)
    # Round-robin acquisition groups, with hash order inside each: no model or
    # image-quality score selects the test samples.
    counts["eligible_distinct_registrations"] = sum(map(len, queues.values()))
    while len(selected) < maximum and any(queues.values()):
        for group in sorted(queues):
            if queues[group] and len(selected) < maximum:
                selected.append(queues[group].pop(0))
    reserved = [row for values in queues.values() for row in values]
    exclusions.extend({"id": row["id"], "reason": "reserved_unscored"} for row in reserved)
    if len(selected) < 20:
        raise ValueError("Too few independent candidates after leakage checks")
    counts["selected_test"] = len(selected)
    document = {"schema_version": 1, "dataset": permission, "samples": selected,
                "selection": "One registration, source-balanced round robin, SHA256 order, before predictions",
                "evaluation_caveat": "Separate publisher source with author XML references, not new independent human verification; public pretraining overlap unknown. Does not establish CCTV/video accuracy.",
                "no_training": True, "production_promoted": False,
                "previous_corpus_sha256": digest(previous / "corpus.jsonl"),
                "previous_adaptation_manifest_sha256": digest(previous / "adaptation_manifest.json"),
                "exclusion_counts": dict(Counter(row["reason"] for row in exclusions)),
                "source_groups": dict(Counter(row["source_group"] for row in selected)),
                "phash_hamming_threshold": 6, "training_crops_phash_checked": len(training_hashes)}
    destination.write_text(json.dumps(document, indent=2), encoding="utf-8")
    (root / "exclusions.json").write_text(json.dumps(exclusions, indent=2), encoding="utf-8")
    (root / "reserve.json").write_text(json.dumps(reserved, indent=2), encoding="utf-8")
    summary = {"manifest_sha256": digest(destination), "counts": dict(counts),
               "exclusions": document["exclusion_counts"], "source_groups": document["source_groups"],
               "note": document["evaluation_caveat"]}
    (root / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--maximum", type=int, default=200)
    args = parser.parse_args()
    prepare(args.root, args.previous, args.maximum)
