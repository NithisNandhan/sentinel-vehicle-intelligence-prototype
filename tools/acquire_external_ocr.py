"""Acquire a separate academic dataset for local noncommercial evaluation only."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from acquire_plate_corpus import download

REFERENCE = "saisirishan/indian-vehicle-dataset"


def acquire(root):
    root.mkdir(parents=True, exist_ok=True)
    url = "https://www.kaggle.com/api/v1/datasets/view/" + REFERENCE
    metadata = json.load(urllib.request.urlopen(url, timeout=45))
    if metadata.get("currentVersionNumber") != 1:
        raise ValueError("Review the new publisher revision before acquiring")
    if "NonCommercial-NoDerivatives" not in metadata.get("licenseName", ""):
        raise ValueError("Publisher licence differs from the reviewed evaluation source")
    (root / "publisher_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    archive = root / "source.zip"
    download(f"https://www.kaggle.com/api/v1/datasets/download/{REFERENCE}?datasetVersionNumber=1",
             archive, max_bytes=512 * 1024 * 1024)
    notice = {"reference": REFERENCE, "version": 1,
              "source_url": "https://www.kaggle.com/datasets/" + REFERENCE,
              "paper": "https://arxiv.org/abs/2207.06657",
              "attribution": "Sai Sirisha Nadiminti; Pranav Kant Gaur; Abhilash Bhardwaj",
              "license": "CC-BY-NC-ND-4.0", "purpose": "private_noncommercial_evaluation_only",
              "no_training": True, "no_redistribution_of_images_or_crops": True,
              "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "pretraining_overlap": "unknown"}
    (root / "permission.json").write_text(json.dumps(notice, indent=2), encoding="utf-8")
    print(json.dumps({"acquired": REFERENCE, "bytes": archive.stat().st_size, "purpose": notice["purpose"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    acquire(parser.parse_args().root)
