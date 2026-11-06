"""Run fixed crop ablations on development only; test replay is deliberately disabled."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from sentinel_edge.ocr_crop_experiment import POLICIES, observe_policies
from sentinel_edge.indian_ocr_finetune import edit_distance


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    import cv2
    import numpy as np
    from sentinel_edge.awiros_experiment import AwirosExperiment
    from compare_svtrv2 import validate_samples
    if args.output.exists():
        raise ValueError("Experiment output exists; preserve the earlier measurements")
    module_path = Path(__file__).resolve().parents[1] / "edge/sentinel_edge/ocr_crop_experiment.py"
    manifest = json.loads(args.manifest.read_text())
    samples = validate_samples(manifest, args.manifest.parent, args.split)
    checkpoint = args.cache / "adaptations/indian-corpus-encoder.safetensors"
    reader = AwirosExperiment(args.cache, device="gpu", checkpoint=checkpoint)
    for _ in range(3):
        reader.probabilities(np.zeros((32, 128, 3), dtype=np.uint8))
    results = []
    for number, sample in enumerate(samples, 1):
        image = cv2.imread(str(args.manifest.parent / sample["crop_path"]))
        if image is None:
            raise ValueError("Cannot decode checked diagnostic crop")
        observations = observe_policies(reader, image, args.policies)
        for observation in observations.values():
            observation["exact"] = observation["reading"] == sample["transcription"]
            observation["character_errors"] = edit_distance(observation["reading"], sample["transcription"])
        results.append({"id": sample["id"], "crop_path": sample["crop_path"],
                        "crop_sha256": sample["crop_sha256"], "native_shape": list(image.shape[:2]),
                        "expected": sample["transcription"], "policies": observations})
        if number % 25 == 0:
            print(json.dumps({"processed": number, "total": len(samples)}), flush=True)
    summary = {}
    for policy in args.policies:
        values = [row["policies"][policy] for row in results]
        durations = [row["elapsed_ms"] for row in values]
        summary[policy] = {"samples": len(values), "exact": sum(row["exact"] for row in values),
            "character_errors": sum(row["character_errors"] for row in values),
            "median_ms": round(statistics.median(durations), 3), "p95_ms": round(float(np.percentile(durations, 95)), 3),
            "extra_ocr_calls": sum(row["ocr_calls"] - 1 for row in values)}
        if "native" in args.policies:
            summary[policy].update({"recovered": sum(not row["policies"]["native"]["exact"] and row["policies"][policy]["exact"] for row in results),
                                   "regressed": sum(row["policies"]["native"]["exact"] and not row["policies"][policy]["exact"] for row in results)})
    output = {"schema_version": 1, "manifest_sha256": digest(args.manifest),
              "checkpoint_sha256": digest(checkpoint), "split": args.split,
              "experiment_module_sha256": digest(module_path),
              "configuration": {"beam_width": 32, "clahe_clip_limit": 2., "clahe_grid": [4, 4],
                                "padding_height_fraction": .06, "padding_width_fraction": .025},
              "purpose": "development_ablation",
              "not_video_or_independent_human_truth": True, "production_promoted": False,
              "policies": args.policies, "summary": summary, "samples": results}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({"output": str(args.output), "summary": summary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--cache", type=Path, default=Path("/models/experiments/awiros"))
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
