"""Compare bounded rescalings of the learned Awiros weight updates."""
import argparse
import json
from pathlib import Path

from compare_svtrv2 import normalise, validate_samples
from sentinel_edge.awiros_experiment import AwirosExperiment, WEIGHTS_SHA256, adaptation_keys, file_hash
from sentinel_edge.ctc_fusion import greedy_text
from sentinel_edge.indian_ocr_finetune import edit_distance


# Factors on the learned change from the publisher checkpoint. The first applies
# to the output head and the second to the adapted layers that read the image.
BASELINE = (1.0, 1.0)
FACTORS = (BASELINE, (0., 0.), (.5, .5), (.75, .75), (1.25, 1.25), (1.5, 1.5),
           (.75, 1.), (1.25, 1.), (1., .75), (1., 1.25))


def run(development, diagnostic, cache, output):
    import cv2
    from safetensors.numpy import load_file, save_file

    output.mkdir(parents=True, exist_ok=False)
    checkpoint = cache / "adaptations/indian-corpus-encoder.safetensors"
    reader = AwirosExperiment(cache, device="gpu", checkpoint=checkpoint)
    parent = load_file(str(cache / "model.safetensors"))
    adapted = load_file(str(checkpoint))
    scope = reader.adaptation["training_scope"]
    keys = adaptation_keys(adapted, scope)
    parameters = dict(reader.model.named_parameters())

    def weights(factors):
        head, encoder = factors
        if factors == BASELINE:
            return {key: adapted[key] for key in keys}
        if factors == (0., 0.):
            return {key: parent[key] for key in keys}
        return {key: (parent[key] + (head if key.startswith("head.ctc_head.") else encoder)
                      * (adapted[key] - parent[key])).astype(adapted[key].dtype) for key in keys}

    def load_crops(path, split):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows = validate_samples(manifest, path.parent, split)
        crops = [cv2.imread(str(path.parent / row["crop_path"])) for row in rows]
        if any(crop is None for crop in crops):
            raise ValueError("Cannot decode a checked crop")
        return crops, [row["transcription"] for row in rows]

    def evaluate(factors, crops, labels):
        for key, value in weights(factors).items():
            parameters[key].set_value(value)
        readings = [normalise(greedy_text(reader.probabilities(crop), reader.tokens)) for crop in crops]
        exact = [reading == label for reading, label in zip(readings, labels)]
        return {"factors": factors, "samples": len(labels), "exact": sum(exact),
                "character_errors": sum(edit_distance(reading, label) for reading, label in zip(readings, labels))}, exact

    def rank(row):
        return -row["exact"], row["character_errors"]

    crops, labels = load_crops(development, "validation")
    baseline, baseline_exact = evaluate(BASELINE, crops, labels)
    candidates = [baseline]
    for factors in FACTORS[1:]:
        candidate, exact = evaluate(factors, crops, labels)
        candidate["recovered"] = sum(b and not a for a, b in zip(baseline_exact, exact))
        candidate["regressed"] = sum(a and not b for a, b in zip(baseline_exact, exact))
        candidates.append(candidate)
        print(json.dumps(candidate), flush=True)
    selected = min(candidates, key=rank)
    result = {"parent_sha256": WEIGHTS_SHA256, "current_sha256": file_hash(checkpoint),
              "development_manifest_sha256": file_hash(development), "development": candidates,
              "selected": selected, "production_promoted": False}
    # Record the development choice before the separate diagnostic is read.
    (output / "development-selection.json").write_text(json.dumps(result, indent=2))
    if selected is not baseline:
        crops, labels = load_crops(diagnostic, "test")
        before, old_exact = evaluate(BASELINE, crops, labels)
        after, new_exact = evaluate(selected["factors"], crops, labels)
        result["previously_scored_external_diagnostic"] = {
            "manifest_sha256": file_hash(diagnostic), "current": before, "candidate": after,
            "recovered": sum(b and not a for a, b in zip(old_exact, new_exact)),
            "regressed": sum(a and not b for a, b in zip(old_exact, new_exact))}
        improves_diagnostic = after["exact"] > before["exact"] and after["character_errors"] <= before["character_errors"]
        if improves_diagnostic and selected["factors"] != (0., 0.):
            path = output / "awiros-weight-candidate.safetensors"
            save_file({**adapted, **weights(selected["factors"])}, str(path))
            metadata = {"parent_sha256": WEIGHTS_SHA256, "dictionary_sha256": file_hash(cache / "en_dict.txt"),
                        "weights_sha256": file_hash(path), "training_scope": scope, "frozen_state_unchanged": True}
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
            reloaded = AwirosExperiment(cache, device="gpu", checkpoint=path)
            readings = [normalise(greedy_text(reloaded.probabilities(crop), reloaded.tokens)) for crop in crops]
            if [reading == label for reading, label in zip(readings, labels)] != new_exact:
                raise ValueError("Reloaded checkpoint changed diagnostic exact matches")
            if sum(edit_distance(reading, label) for reading, label in zip(readings, labels)) != after["character_errors"]:
                raise ValueError("Reloaded checkpoint changed diagnostic character errors")
            result["saved_candidate"] = metadata
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/models/experiments/awiros"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.development, args.diagnostic, args.cache, args.output)
