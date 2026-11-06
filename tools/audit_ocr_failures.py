"""Create local contact sheets for an existing private OCR test; no new inference."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_report(manifest, report):
    samples = {row["id"]: row for row in manifest["samples"] if row["split"] == report["split"]}
    rows = {row["id"]: row for row in report["stills"]}
    if len(rows) != len(report["stills"]) or len(samples) != len(manifest["samples"]) or set(rows) != set(samples):
        raise ValueError("Report and frozen sample identities differ")
    for key, row in rows.items():
        source = samples[key]
        if (row["expected"] != source["transcription"] or row["crop_path"] != source["crop_path"]
                or row["crop_sha256"] != source["crop_sha256"]
                or row["exact"] != (row["reading"] == row["expected"])
                or row["native_shape"] != [source["height"], source["width"]]):
            raise ValueError("Prediction report differs from frozen crop/reference")
    return samples


def validate_index(index, report):
    failed = [row for row in report["stills"] if not row["exact"]]
    if len(failed) != len(index["samples"]):
        raise ValueError("Failure index size differs from source report")
    for number, (source, row) in enumerate(zip(failed, index["samples"]), 1):
        if (row["number"] != number or any(row[key] != source[key] for key in
                ("id", "expected", "reading", "crop_sha256"))
                or [row["native_height"], row["native_width"]] != source["native_shape"]):
            raise ValueError("Failure index differs from source prediction")


def audit(root):
    import cv2
    import numpy as np
    manifest_path = root / "adaptation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    report_path = root / "results/awiros-adapted-test.json"
    report = json.loads(report_path.read_text())
    if report["manifest_sha256"] != digest(manifest_path) or not manifest.get("no_training"):
        raise ValueError("Frozen private evaluation receipt mismatch")
    samples = validate_report(manifest, report)
    failed = [row for row in report["stills"] if not row["exact"]]
    output = root / "failure-audit"
    output.mkdir(exist_ok=True)
    sheets, index = [], []
    for start in range(0, len(failed), 9):
        canvas = np.full((660, 1380, 3), 246, dtype=np.uint8)
        for offset, row in enumerate(failed[start:start + 9]):
            sample = samples[row["id"]]
            filename = (root / sample["crop_path"]).resolve()
            if not filename.is_relative_to(root.resolve()) or digest(filename) != sample["crop_sha256"]:
                raise ValueError("Modified or unsafe audit crop")
            image = cv2.imread(str(filename))
            height, width = image.shape[:2]
            scale = min(420 / width, 132 / height, 6)
            shown = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_NEAREST)
            x, y = offset % 3 * 460, offset // 3 * 220
            number = start + offset + 1
            for line, text in enumerate((f"#{number} {width}x{height}", f"XML: {row['expected']}", f"OCR: {row['reading']}")):
                cv2.putText(canvas, text, (x + 12, y + 21 + line * 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (30, 45, 40), 1, cv2.LINE_AA)
            canvas[y + 80:y + 80 + shown.shape[0], x + 12:x + 12 + shown.shape[1]] = shown
            index.append({"number": number, "id": row["id"], "native_width": width, "native_height": height,
                          "expected": row["expected"], "reading": row["reading"], "crop_sha256": sample["crop_sha256"]})
        name = f"failures-{len(sheets) + 1:02d}.png"
        if not cv2.imwrite(str(output / name), canvas):
            raise OSError("Cannot save audit sheet")
        sheets.append(name)
    (output / "index.json").write_text(json.dumps({"manifest_sha256": digest(manifest_path),
        "report_sha256": digest(report_path), "permission": "private_noncommercial_no_redistribution",
        "not_independent_human_verification": True, "samples": index, "sheets": sheets}, indent=2))
    print(json.dumps({"failures": len(failed), "sheets": len(sheets), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    audit(parser.parse_args().root)
