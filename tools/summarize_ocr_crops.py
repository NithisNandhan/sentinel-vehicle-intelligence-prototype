"""Summarise fixed OCR ablations and render a private local failure-review viewer."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import subprocess

from audit_ocr_failures import validate_index, validate_report


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(left, right):
    a, b = {row["id"]: row for row in left}, {row["id"]: row for row in right}
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError("Reader comparison has missing or duplicate samples")
    if any(a[key]["expected"] != b[key]["expected"] for key in a):
        raise ValueError("Reader comparison has different references")
    agree = [key for key in a if a[key]["reading"] == b[key]["reading"]]
    return {"samples": len(a), "agree": len(agree),
            "agree_correct": sum(a[key]["exact"] for key in agree),
            "agree_wrong": sum(not a[key]["exact"] for key in agree),
            "disagree": len(a) - len(agree),
            "awiros_only_correct": sum(a[key]["exact"] and not b[key]["exact"] for key in a),
            "svtr_only_correct": sum(not a[key]["exact"] and b[key]["exact"] for key in a),
            "both_wrong": sum(not a[key]["exact"] and not b[key]["exact"] for key in a)}


def private_output_path(output):
    repository = Path(__file__).resolve().parents[1]
    allowed = (repository / "data/benchmarks/reports").resolve()
    if not output.resolve().is_relative_to(allowed):
        raise ValueError("Private viewer must stay in this checkout's benchmark reports directory")
    ignored = subprocess.run(["git", "check-ignore", "--quiet", "--", str(output.resolve())], cwd=repository, check=False)
    if ignored.returncode != 0:
        raise ValueError("Private viewer destination is not ignored by Git")


def summarize(root, corpus, review_path, output, aggregate, allow_private=False):
    manifest_path = root / "adaptation_manifest.json"
    manifest, review = load(manifest_path), load(review_path)
    if (not allow_private or not manifest.get("no_training")
            or not manifest["dataset"].get("no_redistribution_of_images_or_crops")
            or review["source_manifest_sha256"] != digest(manifest_path)):
        raise ValueError("Explicit private-evaluation permission and matching review required")
    private_output_path(output)
    dev = load(root / "crop-experiments/development-final.json")
    svtr_dev = load(root / "crop-experiments/svtrv2-development.json")
    if dev["manifest_sha256"] != digest(corpus / "adaptation_manifest.json") or svtr_dev["manifest_sha256"] != dev["manifest_sha256"]:
        raise ValueError("Changed development manifests")
    external, second = load(root / "results/awiros-adapted-test.json"), load(root / "results/svtrv2-test.json")
    if any(row["manifest_sha256"] != digest(manifest_path) for row in (external, second)):
        raise ValueError("Changed external manifest")
    samples = validate_report(manifest, external)
    validate_report(manifest, second)
    dev_left = [{"id": row["id"], "expected": row["expected"], **row["policies"]["native"]} for row in dev["samples"]]
    index = load(root / "failure-audit/index.json")
    if index["report_sha256"] != digest(root / "results/awiros-adapted-test.json"):
        raise ValueError("Audit refers to a different prediction report")
    if index["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Audit refers to a different frozen manifest")
    validate_index(index, external)
    categories = {number: category for category in review["categories"] for number in category["numbers"]}
    all_numbers = [number for category in review["categories"] for number in category["numbers"]]
    if len(set(all_numbers)) != len(all_numbers) or set(all_numbers) != {row["number"] for row in index["samples"]}:
        raise ValueError("Review categories must cover every failure exactly once")
    evidence = {"schema_version": 1, "production_promoted": False, "decision": "retain_native_adapted_awiros",
        "development_manifest_sha256": dev["manifest_sha256"], "external_manifest_sha256": digest(manifest_path),
        "development_report_sha256": digest(root / "crop-experiments/development-final.json"),
        "checkpoint_sha256": dev["checkpoint_sha256"], "experiment_module_sha256": dev["experiment_module_sha256"],
        "development": dev["summary"], "paired_development": paired(dev_left, svtr_dev["stills"]),
        "paired_already_scored_external_diagnostic": paired(external["stills"], second["stills"]),
        "failure_review": {"samples": len(index["samples"]), "not_independent_human_truth": True,
            "width_below_64": sum(row["native_width"] < 64 for row in index["samples"]),
            "height_below_16": sum(row["native_height"] < 16 for row in index["samples"]),
            "categories": {row["name"]: len(row["numbers"]) for row in review["categories"]}},
        "unscored_reserve_sha256": digest(root / "reserve.json"),
        "reserved_registrations_scored": 0, "reference_labels_changed": False,
        "caveat": "Development data previously used for adaptation; external failures AI-reviewed, not human-adjudicated. No new blind CCTV or full-video evaluation."}
    esc = lambda value: html.escape(str(value), quote=True)
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OCR error analysis</title><style>body{max-width:1120px;margin:auto;padding:24px;background:#f6f5f1;color:#242925;font:15px/1.6 system-ui}section{padding:20px;margin:20px 0;background:#fffdf9;border:1px solid #dedfd8;border-radius:8px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}code,small{overflow-wrap:anywhere}img{max-width:100%;max-height:220px;object-fit:contain}.scroll{overflow:auto}.note{padding:16px;background:#fff6e6;border-left:4px solid #92400e}a{color:#0f766e}</style><h1>OCR error analysis</h1><p><a href="external-ocr-20261101.html">Previous three-model comparison</a></p><p class="note">Private noncommercial evaluation. Do not redistribute this image-containing viewer. External source CC BY-NC-ND 4.0. Visual annotations are AI review, not independent human verification. Original references and scores remain unchanged.</p><h2>Development ablations</h2><p>37 existing validation crops, previously used during adaptation. No reserved external registrations scored. Native adapted Awiros retained; no production changes.</p><div class="scroll"><table><tr><th>Policy</th><th>Exact /37</th><th>Character errors</th><th>Recovered</th><th>Regressed</th><th>Median ms</th></tr>']
    for name, values in dev["summary"].items():
        parts.append('<tr>' + ''.join(f'<td>{esc(value)}</td>' for value in (name, values["exact"], values["character_errors"], values["recovered"], values["regressed"], values["median_ms"])) + '</tr>')
    parts.append('</table></div><p>Policy timing includes the native baseline plus additional view inference/decoding where used; excludes detection, video tracking and dashboard. It is not full-pipeline throughput.</p><h2>Two-reader agreement</h2><div class="scroll"><table><tr><th>Dataset</th><th>Agreed</th><th>Agreed correct</th><th>Agreed wrong</th><th>Disagreed</th></tr>')
    for label, key in (("Development (37)", "paired_development"), ("Previously scored external diagnostic (200)", "paired_already_scored_external_diagnostic")):
        row = evidence[key]
        parts.append('<tr>' + ''.join(f'<td>{esc(value)}</td>' for value in (label, row["agree"], row["agree_correct"], row["agree_wrong"], row["disagree"])) + '</tr>')
    parts.append('</table></div><p>Agreement is not proof of correctness. Disagreement does not reveal which reader is correct. No calibrated confidence selector or ensemble accuracy is claimed.</p><h2>All 63 external differences</h2><p>22 crops are narrower than 64 pixels; 16 have height below 16 pixels. Categories below are review hypotheses. Potential reference problems need human adjudication, not score correction based on OCR.</p>')
    for row in index["samples"]:
        source = samples[row["id"]]
        filename = (root / source["crop_path"]).resolve()
        if not filename.is_relative_to(root.resolve()) or digest(filename) != source["crop_sha256"]:
            raise ValueError("Modified private crop")
        image = base64.b64encode(filename.read_bytes()).decode()
        category = categories[row["number"]]
        parts.append(f'<section><h3>Failure {row["number"]} · {esc(category["name"].replace("_", " "))}</h3><p>Publisher reference: <code>{esc(row["expected"])}</code><br>Awiros reading: <code>{esc(row["reading"])}</code><br>{row["native_width"]} × {row["native_height"]} native pixels</p><img src="data:image/png;base64,{image}" alt="Native private-evaluation plate crop"><p>{esc(category["note"])}</p></section>')
    parts.append('<p>External dataset: Sai Sirisha Nadiminti; Pranav Kant Gaur; Abhilash Bhardwaj. <a href="https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset">Publisher source</a>. No images in this viewer are redistributed to GitHub.</p></html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(parts), encoding="utf-8")
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    aggregate.write_text(json.dumps(evidence, indent=2))
    print(json.dumps({"viewer": str(output), "aggregate": str(aggregate), "development": evidence["development"],
                      "paired_development": evidence["paired_development"],
                      "paired_external": evidence["paired_already_scored_external_diagnostic"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--allow-private-evaluation", action="store_true")
    args = parser.parse_args()
    summarize(args.root, args.corpus, args.review, args.output, args.aggregate, args.allow_private_evaluation)
