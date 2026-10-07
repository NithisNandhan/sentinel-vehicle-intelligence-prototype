"""Fail publication if selected files contain an actual local credential.

Only paths are reported. Credential values are never printed or written.
"""

import argparse
import json
import subprocess
import zipfile
from pathlib import Path


def local_secrets(root):
    result = []
    for line in (root / '.env').read_text(encoding='utf-8-sig').splitlines():
        if '=' not in line or line.lstrip().startswith('#'):
            continue
        key, value = line.split('=', 1)
        value = value.strip().strip('\"\'')
        if len(value) >= 8 and any(part in key.upper() for part in
                                  ('PASSWORD', 'COOKIE', 'EMAIL', 'API_KEY', 'HMAC_KEY', 'TOKEN')):
            result.append(value.encode())
    return result


def check(root, paths, text_only=False):
    secrets = local_secrets(root)
    failed, total = [], 0
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        if text_only and path.suffix.lower() not in {
                '.json', '.jsonl', '.txt', '.md', '.html', '.csv', '.yaml', '.yml',
                '.toml', '.ini', '.py', '.ps1', '.cjs', '.js', '.ts', '.tsx',
                '.xml', '.log', '.vtt', '.sql', '.docx', '.pptx', '.xlsx'}:
            continue
        blobs = []
        if path.suffix in {'.docx', '.pptx', '.xlsx'}:
            with zipfile.ZipFile(path) as archive:
                blobs = [archive.read(name) for name in archive.namelist() if name.endswith('.xml')]
        elif path.stat().st_size <= 64 * 1024 * 1024:
            blobs = [path.read_bytes()]
        else:
            continue
        total += 1
        if any(secret in blob for blob in blobs for secret in secrets):
            failed.append(str(path.relative_to(root)))
    print(json.dumps({'files_checked': total, 'credential_matches': failed}))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--extra', action='append', default=[])
    parser.add_argument('--text-only', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    tracked = subprocess.check_output(['git', '-c', f'safe.directory={root.as_posix()}',
                                       'ls-files', '-z'], cwd=root).decode().split('\0')
    paths = [root / name for name in tracked if name]
    for extra in args.extra:
        extra = root / extra
        paths.extend(extra.rglob('*') if extra.is_dir() else [extra])
    check(root, paths, args.text_only)
