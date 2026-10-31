"""Fetch the official SVTRv2-B research assets; no production model changes."""
import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

REVISION = "1ccfc6ee6161f7133b192e16af3a9265273997ba"
OFFICIAL_FOLDER = "https://drive.google.com/drive/folders/11u11ptDzQ4BF9RRsOYdZnXl6ell2h4jN"
WEIGHTS_ID = "1P56B_PVYBF649zUqQFShuAZC1XuNmUcL"
CONFIG_ID = "1Ds-TilzI_2JIAomKyJS5RFCn-fU-UsOn"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path, maximum):
    if path.is_file():
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Sentinel research benchmark"})
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > maximum:
                    raise ValueError("Asset exceeds the declared download limit")
                output.write(chunk)
        with temporary.open("rb") as check:
            prefix = check.read(512)
        if size == 0 or prefix.lstrip().lower().startswith((b"<!doctype html", b"<html")):
            raise ValueError("Publisher returned a web page instead of the requested asset")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    files = [
        ("source.tar.gz", f"https://codeload.github.com/Topdu/OpenOCR/tar.gz/{REVISION}", 64 * 1024 * 1024),
        ("best.pth", f"https://drive.usercontent.google.com/download?id={WEIGHTS_ID}&export=download&confirm=t", 128 * 1024 * 1024),
        ("config.yml", f"https://drive.usercontent.google.com/download?id={CONFIG_ID}&export=download&confirm=t", 128 * 1024),
    ]
    receipts = []
    for name, url, maximum in files:
        destination = root / name
        download(url, destination, maximum)
        receipts.append({"name": name, "url": url, "sha256": digest(destination), "bytes": destination.stat().st_size})
        print(json.dumps({"asset": name, "bytes": destination.stat().st_size, "sha256": receipts[-1]["sha256"]}), flush=True)
    source = root / "OpenOCR"
    source.mkdir(exist_ok=True)
    extracted = 0
    with tarfile.open(root / "source.tar.gz") as archive:
        for member in archive:
            parts = Path(member.name).parts[1:]
            if not parts or parts[0] not in {"openrec", "tools", "LICENSE", "NOTICE", "requirements.txt"} or not member.isfile():
                continue
            destination = source.joinpath(*parts).resolve()
            if not destination.is_relative_to(source.resolve()):
                raise ValueError("Unsafe publisher source archive path")
            extracted += member.size
            if extracted > 64 * 1024 * 1024:
                raise ValueError("Publisher source exceeds extraction limit")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream:
                destination.write_bytes(stream.read())
    manifest = {"model": "SVTRv2-B Union14M-L-Filter", "source_revision": REVISION,
                "publisher_folder": OFFICIAL_FOLDER, "assets": receipts,
                "checksum_scope": "Locally computed acquisition hashes; publisher did not supply an independently verified weight digest",
                "source_files": [{"path": str(p.relative_to(source)).replace("\\", "/"), "sha256": digest(p)}
                                 for p in sorted(source.rglob("*")) if p.is_file() and "__pycache__" not in p.parts],
                "production_promoted": False}
    (root / "assets.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    prepare(parser.parse_args().root)
