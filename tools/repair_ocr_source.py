"""Restore OCR source files that the 7 October handoff archive left out.

The handoff archiver omits any path component named `token` as a possible
secret. PaddleOCR has a source folder with that name
(`ppocr/data/imaug/vqa/token/`), so five pinned source files are missing after
a restore and the OCR reader refuses to load.

This tool copies ONLY files that the pinned manifest (`assets.json`) lists and
that are absent, taking them from the pinned source archive already in the
model cache, and only when the archive and each file match their recorded
SHA256. It never overwrites a file and downloads nothing. Run it once after
`scripts/restore-teammate-handoff.ps1`:

  docker run --rm --entrypoint python -v sentinel-prototype_model_cache:/models \
      -v "$PWD/tools:/tools:ro" sentinel-prototype-api:latest /tools/repair_ocr_source.py
"""
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def repair(cache=Path('/models/experiments/awiros')):
    manifest = json.loads((cache/'assets.json').read_text(encoding='utf-8'))
    source = cache/'PaddleOCR'
    missing = {name for name in manifest['source_files'] if not (source/name).exists()}
    if not missing:
        return 0, 0
    archive = cache/'paddleocr-source.tar.gz'
    recorded = next((a['sha256'] for a in manifest['assets'] if a['name'] == archive.name), None)
    if recorded is None or hashlib.sha256(archive.read_bytes()).hexdigest() != recorded:
        raise SystemExit('The pinned source archive is absent or does not match the manifest; nothing was changed.')
    restored = 0
    with tarfile.open(archive) as bundle:
        for member in bundle:
            name = '/'.join(Path(member.name).parts[1:])
            if name not in missing or not member.isfile():
                continue
            data = bundle.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != manifest['source_files'][name]:
                raise SystemExit(f'{name} in the archive does not match the manifest; stopped.')
            destination = (source/name).resolve()
            if not destination.is_relative_to(source.resolve()):
                raise SystemExit('Unsafe path in the manifest; stopped.')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(data)
            restored += 1
    return len(missing), restored


if __name__ == '__main__':
    absent, restored = repair(Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/models/experiments/awiros'))
    print(f'{absent} pinned source file(s) were missing; {restored} restored with matching hashes.')
    if restored != absent:
        raise SystemExit('Some files could not be restored.')
