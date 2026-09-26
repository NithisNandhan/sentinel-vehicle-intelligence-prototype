"""Acquire a bounded, public CC BY 4.0 review subset, not the full archive.

Publisher labels are proposals until a human visual review confirms them.
No live credentials or camera footage are used. Requires openpyxl.
"""
from __future__ import annotations
import argparse
import io
import json
import re
import urllib.request
import zipfile
import zlib
from pathlib import Path

URL = 'https://zenodo.org/records/13954136/files/number_plate.zip?download=1'
SIZE = 1468393969


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = self.transferred = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else SIZE if whence == 2 else 0)
        if self.position < 0:
            raise ValueError('Invalid archive position')
        return self.position

    def read(self, length=-1):
        length = SIZE - self.position if length < 0 else min(length, SIZE - self.position)
        if length <= 0:
            return b''
        if length > 12 * 1024**2 or self.transferred + length > 64 * 1024**2:
            raise ValueError('Bounded pilot transfer exceeded')
        start, end = self.position, self.position + length - 1
        request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=45) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                raise ValueError('Precise byte-range access required')
            data = response.read(length + 1)
        if len(data) != length:
            raise ValueError('Truncated archive range')
        self.position += length
        self.transferred += length
        return data


def normalized(name):
    return re.sub(r'[ ()]', '', Path(name).stem).upper()


def main():
    import openpyxl
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    inventory = json.loads((root / 'inventory.json').read_text())
    members = {normalized(item['name']): item for item in inventory['archive_files']
               if Path(item['name']).suffix.lower() in {'.png', '.jpg', '.jpeg'}}
    workbook = openpyxl.load_workbook(root / 'number_plate.xlsx', read_only=True, data_only=True)
    proposed, counts, seen = [], {}, set()
    for name, label, *_ in list(workbook.active.values)[1:]:
        label = str(label).strip().upper()
        state = label[:2]
        # Regional filename groups only; the renamed foreign/mixed groups are excluded.
        if not re.fullmatch(r'[A-Z]{2}\d+', normalized(str(name))):
            continue
        if not re.fullmatch(r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}', label):
            continue
        if normalized(str(name)) not in members or label in seen or counts.get(state, 0) >= 2:
            continue
        item = members[normalized(str(name))]
        if item['bytes'] > 2 * 1024**2:
            continue
        seen.add(label)
        counts[state] = counts.get(state, 0) + 1
        proposed.append({'path': item['name'], 'transcription': label, 'review': 'pending'})
        if len(proposed) >= 48:
            break
    remote = RemoteZip()
    with zipfile.ZipFile(remote) as archive:
        for row in proposed:
            member = archive.getinfo(row['path'])
            target = (root / row['path']).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Unsafe archive member')
            if not target.is_file() or zlib.crc32(target.read_bytes()) != member.CRC:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))  # zipfile verifies member CRC.
            print(json.dumps({'cached': row['path'], 'bytes': member.file_size}), flush=True)
    (root / 'review_pending.json').write_text(json.dumps(proposed, indent=2))
    print(json.dumps({'samples': len(proposed), 'states': len(counts),
                      'transferred_bytes': remote.transferred}), flush=True)


if __name__ == '__main__':
    main()
