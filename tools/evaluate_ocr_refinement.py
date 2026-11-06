"""Evaluate a frozen crop-retry rule against manual and existing regression sets."""
import argparse
import json
from pathlib import Path

from sentinel_edge.awiros_experiment import AwirosExperiment, file_hash
from sentinel_edge.corpus_ocr_adaptation import validate
from sentinel_edge.indian_ocr_finetune import edit_distance
from sentinel_edge.ocr_refinement import POLICY, refine


def evaluate(reader, samples):
    import cv2
    results = []
    for sample in samples:
        if file_hash(sample["path"]) != sample["sha256"]:
            raise ValueError("Evaluation crop changed")
        image = cv2.imread(str(sample["path"]))
        if image is None:
            raise ValueError("Evaluation crop cannot be decoded")
        result = refine(reader, image)
        expected = sample["expected"]
        result.update({"id": sample["id"], "expected": expected,
                       "baseline_exact": result["baseline"] == expected,
                       "refined_exact": result["reading"] == expected,
                       "baseline_errors": edit_distance(result["baseline"], expected),
                       "refined_errors": edit_distance(result["reading"], expected)})
        results.append(result)
    return {"samples": len(results),
            "baseline_exact": sum(r["baseline_exact"] for r in results),
            "refined_exact": sum(r["refined_exact"] for r in results),
            "baseline_character_errors": sum(r["baseline_errors"] for r in results),
            "refined_character_errors": sum(r["refined_errors"] for r in results),
            "recovered": sum(r["refined_exact"] and not r["baseline_exact"] for r in results),
            "regressed": sum(r["baseline_exact"] and not r["refined_exact"] for r in results),
            "changed": sum(r["changed"] for r in results),
            "extra_ocr_calls": sum(r["ocr_calls"] - 1 for r in results), "results": results}


def main(args):
    if args.output.exists():
        raise ValueError("Evaluation output exists. Preserve the frozen measurements.")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    labels = load(args.review / "labels.json")
    manifest = args.review / "ocr-model-dev/rectified-extended-manifest.json"
    corrected = load(manifest)
    assert corrected["review_manifest_sha256"] == labels["review_manifest_sha256"] == file_hash(args.review / "manifest.json")
    references = {r["id"]: r for r in labels["labels"]}
    human = [{"id": r["id"], "path": manifest.parent / r["crop_path"],
              "sha256": r["crop_sha256"], "expected": references[r["id"]]["plate"]}
             for r in corrected["samples"] if r["id"] != "21"]
    corpus_path = args.corpus / "adaptation_manifest.json"
    corpus = load(corpus_path)
    validate(corpus, args.corpus)
    validation = [{"id": r["id"], "path": args.corpus / r["crop_path"],
                   "sha256": r["crop_sha256"], "expected": r["transcription"]}
                  for r in corpus["samples"] if r["split"] == "validation"]
    external_manifest = args.external / "adaptation_manifest.json"
    external = load(external_manifest)
    testing = [{"id": r["id"], "path": args.external / r["crop_path"],
                "sha256": r["crop_sha256"], "expected": r["transcription"]}
               for r in external["samples"] if r["split"] == "test"]
    checkpoint = args.checkpoint or args.cache / "adaptations/indian-corpus-encoder.safetensors"
    reader = AwirosExperiment(args.cache, device="gpu", checkpoint=checkpoint)
    report = {"checkpoint_sha256": file_hash(checkpoint), "policy": POLICY,
              "policy_source_sha256": file_hash(Path(__file__).resolve().parents[1] / "edge/sentinel_edge/ocr_refinement.py"),
              "manifest_hashes": {"manual": file_hash(manifest), "corpus": file_hash(corpus_path),
                                  "external": file_hash(external_manifest)},
              "scope": "Manual development crops and previously examined source-separated publisher-labelled regression set. No new blind test or production promotion.",
              "excluded_uncertain_manual_ids": ["21"], "datasets": {}}
    for name, samples in [("mumbai_manual", human), ("corpus_validation", validation), ("external_regression", testing)]:
        report["datasets"][name] = evaluate(reader, samples)
        print(json.dumps({"dataset": name, **{k: v for k, v in report["datasets"][name].items() if k != "results"}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/models/experiments/awiros"))
    parser.add_argument("--checkpoint", type=Path)
    main(parser.parse_args())
