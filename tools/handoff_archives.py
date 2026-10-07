"""Create and restore checksum-verified multipart handoff archives.

This tool never reads .env and refuses extraction into a nonempty directory.
Run export inside a Linux container to preserve the model runtime's symlinks.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path

OMIT = {'.git', '.env', '.git-credentials', '.gitconfig', '__pycache__',
        '.pytest_cache', '.mypy_cache', '.ruff_cache', 'node_modules',
        'token', 'stored_tokens', 'settings.json'}


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class PartWriter(io.RawIOBase):
    def __init__(self, directory, name, limit, resume=False):
        super().__init__()
        self.directory, self.name, self.limit = directory, name, limit
        self.current = None
        self.parts = []
        self.size = 0
        self.digest = None
        self.resume = resume
        self.existing_bytes = 0

    def writable(self):
        return True

    def _finish(self):
        if self.current is not None:
            self.current.close()
            self.parts.append({'name': self.path.name, 'bytes': self.size,
                               'sha256': self.digest.hexdigest()})
            print(f'Completed {self.path.name}: {self.size:,} bytes', flush=True)
            self.current = None

    def write(self, value):
        buffer = memoryview(value)
        count = len(buffer)
        while buffer:
            if self.current is None:
                self.path = self.directory / f'{self.name}.tar.gz.part{len(self.parts) + 1:03d}'
                if self.resume and self.path.exists():
                    self.existing_bytes = self.path.stat().st_size
                    self.current = self.path.open('r+b', buffering=1024 * 1024)
                else:
                    self.existing_bytes = 0
                    self.current = self.path.open('xb', buffering=1024 * 1024)
                self.size, self.digest = 0, hashlib.sha256()
            piece = buffer[:self.limit - self.size]
            compare_bytes = min(len(piece), max(0, self.existing_bytes - self.size))
            if compare_bytes and self.current.read(compare_bytes) != bytes(piece[:compare_bytes]):
                raise ValueError('Resume source differs from the existing archive; existing prefix preserved')
            if compare_bytes < len(piece):
                self.current.write(piece[compare_bytes:])
            self.digest.update(piece)
            self.size += len(piece)
            buffer = buffer[len(piece):]
            if self.size == self.limit:
                self._finish()
        return count

    def close(self):
        if not self.closed:
            self._finish()
        super().close()


class PartReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            raise ValueError('Multipart archive must be read as a stream')
        result = bytearray()
        while len(result) < size:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = path.open('rb')
            block = self.current.read(size - len(result))
            if block:
                result.extend(block)
            else:
                self.current.close()
                self.current = None
        return bytes(result)

    def close(self):
        if self.current:
            self.current.close()
        super().close()


def export(name, sources, output, part_mib=256, exclude=(), resume=False,
           stream_buffer=1024 * 1024):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f'{name}.manifest.json'
    if manifest_path.exists() or (list(output.glob(f'{name}.tar.gz.part*')) and not resume):
        raise ValueError(f'Existing archive preserved: {name}')
    sources = [(Path(root).resolve(), prefix.strip('/')) for root, prefix in sources]
    excluded = [Path(p).resolve() for p in exclude]
    checkpoints, omitted = [], []
    files, bytes_total, last_report = 0, 0, time.monotonic()
    first = output / f'{name}.tar.gz.part001'
    mtime = None
    if resume and first.exists():
        with first.open('rb') as stream:
            header = stream.read(10)
        if header[:2] != b'\x1f\x8b':
            raise ValueError('Resume requires an existing gzip archive prefix')
        mtime = int.from_bytes(header[4:8], 'little')
    writer = PartWriter(output, name, part_mib * 1024 * 1024, resume)
    try:
        with gzip.GzipFile(fileobj=writer, mode='wb', compresslevel=1, mtime=mtime) as compressed, \
                tarfile.open(fileobj=compressed, mode='w|', bufsize=stream_buffer,
                             copybufsize=stream_buffer) as archive:
            for root, prefix in sources:
                if not root.is_dir():
                    raise ValueError(f'Source directory missing: {root}')
                for directory, folders, names in os.walk(root, followlinks=False):
                    directory = Path(directory)
                    folders[:] = sorted(p for p in folders if p not in OMIT and
                                        not any((directory / p).resolve().is_relative_to(e) for e in excluded))
                    for filename in sorted(names):
                        path = directory / filename
                        if filename in OMIT or filename.endswith(('.pyc', '.partial', '-wal', '-shm')):
                            continue
                        if any(path.is_relative_to(e) for e in excluded):
                            continue
                        relative = path.relative_to(root).as_posix()
                        target = f'{prefix}/{relative}' if prefix else relative
                        if path.is_symlink() and not path.resolve().is_relative_to(root):
                            omitted.append({'path': target, 'reason': 'external symlink'})
                            continue
                        archive.add(path, arcname=target, recursive=False)
                        files += 1
                        size = path.stat().st_size
                        bytes_total += size
                        if path.suffix in {'.safetensors', '.pt', '.onnx', '.pdparams'}:
                            checkpoints.append({'path': target, 'bytes': size,
                                                'sha256': digest_file(path)})
                        if time.monotonic() - last_report > 25:
                            print(f'{name}: {files:,} files, {bytes_total:,} input bytes', flush=True)
                            last_report = time.monotonic()
    finally:
        writer.close()
    manifest = {'format': 'sentinel-multipart-tar-gzip-v1', 'name': name,
                'files': files, 'input_bytes': bytes_total, 'parts': writer.parts,
                'checkpoints': checkpoints, 'omitted': omitted,
                'excluded_names': sorted(OMIT),
                'restore': 'Verify all part checksums, then extract into an empty destination.'}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'{name}: archive complete ({len(writer.parts)} parts)', flush=True)
    return manifest


def restore(manifest_path, destination):
    manifest_path, destination = Path(manifest_path), Path(destination)
    record = json.loads(manifest_path.read_text(encoding='utf-8'))
    if record.get('format') != 'sentinel-multipart-tar-gzip-v1':
        raise ValueError('Unsupported archive format')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Restore destination must be empty; existing data preserved')
    parts = []
    for entry in record['parts']:
        if Path(entry['name']).name != entry['name']:
            raise ValueError('Invalid part filename')
        path = manifest_path.parent / entry['name']
        if path.stat().st_size != entry['bytes'] or digest_file(path) != entry['sha256']:
            raise ValueError(f'Archive integrity mismatch: {entry["name"]}')
        parts.append(path)
    destination.mkdir(parents=True, exist_ok=True)
    with PartReader(parts) as stream, tarfile.open(fileobj=stream, mode='r|gz') as archive:
        for member in archive:
            archive.extract(member, destination, filter='data')
    for checkpoint in record.get('checkpoints', []):
        path = destination / checkpoint['path']
        if digest_file(path) != checkpoint['sha256']:
            raise ValueError(f'Restored model integrity mismatch: {checkpoint["path"]}')
    print(f'Restored {record["name"]}: {record["files"]} files', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('export')
    create.add_argument('--name', required=True)
    create.add_argument('--source', action='append', required=True, metavar='DIRECTORY=PREFIX')
    create.add_argument('--exclude', action='append', default=[])
    create.add_argument('--output', required=True)
    create.add_argument('--part-mib', type=int, default=256)
    create.add_argument('--resume', action='store_true')
    extract = commands.add_parser('restore')
    extract.add_argument('--manifest', required=True)
    extract.add_argument('--destination', required=True)
    args = parser.parse_args()
    if args.command == 'export':
        export(args.name, [source.rsplit('=', 1) for source in args.source],
               args.output, args.part_mib, args.exclude, args.resume)
    else:
        restore(args.manifest, args.destination)


if __name__ == '__main__':
    main()
