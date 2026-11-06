"""Bounded warm-start adaptation using existing train-only crops and synthetic styles."""
import argparse
import json
import random
from pathlib import Path

from sentinel_edge.awiros_experiment import AwirosExperiment, WEIGHTS_SHA256, adaptation_keys, file_hash, input_array
from sentinel_edge.corpus_ocr_adaptation import validate
from sentinel_edge.ctc_fusion import greedy_text
from sentinel_edge.indian_ocr_finetune import augment, edit_distance
from sentinel_edge.ocr_style import mixed_font_plate


def run(args):
    if (args.output / "style-refined.safetensors").exists() or (args.output / "style-refined.json").exists():
        raise ValueError("Training output exists. Use a new experiment directory.")
    import cv2
    import numpy as np
    from safetensors.numpy import save_file
    manifest = args.corpus / "adaptation_manifest.json"
    rows = validate(json.loads(manifest.read_text()), args.corpus)
    checkpoint = args.cache / "adaptations/indian-corpus-encoder.safetensors"
    reader = AwirosExperiment(args.cache, device="gpu", checkpoint=checkpoint)
    paddle, model = reader.paddle, reader.model
    seed = 20261106
    random.seed(seed)
    rng = np.random.default_rng(seed)
    paddle.seed(seed)
    original = {name: value.numpy().copy() for name, value in model.state_dict().items()}
    keys = adaptation_keys(original, "bounded_image_encoder_tail_and_ctc")
    for name, parameter in model.named_parameters():
        parameter.stop_gradient = name not in keys
    model.eval()
    model.head.ctc_encoder.encoder.use_guide = False
    images = {row["id"]: cv2.imread(str(args.corpus / row["crop_path"]))
              for row in rows if row["split"] in {"train", "validation"}}
    if any(image is None for image in images.values()):
        raise ValueError("A checked corpus image could not be decoded")
    lookup = {char: i for i, char in enumerate(reader.tokens)}
    validation = [r for r in rows if r["split"] == "validation"]
    photos = [r for r in rows if r["split"] == "train" and r["origin"] == "photograph_publisher_declared"]
    generated = [r for r in rows if r["split"] == "train" and r["origin"] == "synthetic"]
    if len(photos) < 3:
        raise ValueError("At least three photographic training crops are required")
    reserved = {r["transcription"] for r in rows if r["split"] != "train"}
    assert not {r["transcription"] for r in photos + generated} & reserved

    def logits(arrays):
        tensor = paddle.to_tensor(np.concatenate([input_array(image) for image in arrays]))
        return model.head.ctc_head.fc(model.head.ctc_encoder(model.backbone(tensor)))

    def evaluate():
        exact = errors = 0
        with paddle.no_grad():
            for row in validation:
                values = paddle.nn.functional.softmax(logits([images[row["id"]]]), axis=2).numpy()[0]
                text = greedy_text(values.tolist(), reader.tokens)
                exact += text == row["transcription"]
                errors += edit_distance(text, row["transcription"])
        return {"samples": len(validation), "exact": exact, "character_errors": errors}

    baseline = evaluate()
    print(json.dumps({"baseline_validation": baseline, "training_photos": len(photos),
                      "synthetic_fraction_maximum": .25}), flush=True)
    optimizer = paddle.optimizer.Adam(learning_rate=.00001,
        parameters=[p for name, p in model.named_parameters() if name in keys],
        grad_clip=paddle.nn.ClipGradByGlobalNorm(1.))
    criterion = paddle.nn.CTCLoss(blank=0, reduction="none")
    best = baseline
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, 5):
        random.shuffle(photos)
        losses = []
        for offset in range(0, min(len(photos) - 2, 300), 3):
            batch = photos[offset:offset + 3]
            arrays = [augment(images[r["id"]], int(rng.integers(0, 4)), rng) for r in batch]
            if not generated or rng.random() < .5:
                extra = random.choice(photos)
                arrays.append(mixed_font_plate(extra["transcription"], rng))
            else:
                extra = random.choice(generated)
                arrays.append(images[extra["id"]])
            labels = [[lookup[c] for c in row["transcription"]] for row in batch + [extra]]
            padded = np.zeros((4, max(map(len, labels))), dtype=np.int32)
            for i, label in enumerate(labels):
                padded[i, :len(label)] = label
            values = logits(arrays)
            loss = criterion(values.transpose((1, 0, 2)), paddle.to_tensor(padded),
                paddle.to_tensor([values.shape[1]] * 4, dtype="int64", place=paddle.CPUPlace()),
                paddle.to_tensor([len(label) for label in labels], dtype="int64")).mean()
            value = float(loss.numpy())
            if not np.isfinite(value):
                raise ValueError("Nonfinite refinement loss")
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            losses.append(value)
        metrics = evaluate()
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "steps": len(losses), "validation": metrics})
        print(json.dumps(history[-1]), flush=True)
        if (-metrics["exact"], metrics["character_errors"]) < (-best["exact"], best["character_errors"]):
            best, best_epoch = metrics, epoch
            best_state = {name: value.numpy().copy() for name, value in model.state_dict().items() if name in keys}
    final = {name: value.numpy().copy() for name, value in model.state_dict().items()}
    if any(not np.array_equal(original[name], final[name]) for name in original if name not in keys):
        raise ValueError("Frozen model state changed")
    selected = {**original, **(best_state if best_state is not None else {name: final[name] for name in keys})}
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "style-refined.safetensors"
    save_file(selected, str(destination))
    metadata = {"parent_sha256": WEIGHTS_SHA256, "initial_checkpoint_sha256": file_hash(checkpoint),
                "dictionary_sha256": file_hash(args.cache / "en_dict.txt"), "weights_sha256": file_hash(destination),
                "training_scope": "bounded_image_encoder_tail_and_ctc", "frozen_state_unchanged": True,
                "manifest_sha256": file_hash(manifest), "seed": seed, "learning_rate": .00001,
                "training_source_sha256": file_hash(Path(__file__)),
                "augmentation_source_sha256": file_hash(Path(__file__).resolve().parents[1] / "edge/sentinel_edge/ocr_style.py"),
                "best_validation_epoch": best_epoch, "validation_preferred": best_epoch > 0,
                "baseline_validation": baseline, "history": history,
                "trainable_parameters": sum(original[k].size for k in keys),
                "scope": "Warm-start bounded style augmentation. Mumbai and external images excluded from training and checkpoint selection.",
                "production_promoted": False}
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    restored = AwirosExperiment(args.cache, device="gpu", checkpoint=destination)
    model.set_state_dict({name: paddle.to_tensor(value) for name, value in selected.items()})
    with paddle.no_grad():
        expected = paddle.nn.functional.softmax(logits([images[validation[0]["id"]]]), axis=2).numpy()[0]
    observed = np.asarray(restored.probabilities(images[validation[0]["id"]]))
    if not np.allclose(observed, expected, atol=1e-5, rtol=1e-4):
        raise ValueError("Saved refinement checkpoint failed serving parity")
    print(json.dumps({"saved": destination.name, "best_validation_epoch": best_epoch,
                      "trainable_parameters": metadata["trainable_parameters"], "parity_passed": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/models/experiments/awiros"))
    run(parser.parse_args())
