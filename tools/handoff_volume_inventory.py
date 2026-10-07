"""Read-only inventory for model and event volumes; prints no file contents."""

import json
from pathlib import Path


def inventory(root):
    files = [p for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink()]
    return {
        'root': root,
        'files': len(files),
        'bytes': sum(p.stat().st_size for p in files),
        'checkpoints': [
            {'path': str(p), 'bytes': p.stat().st_size}
            for p in files
            if p.suffix in {'.safetensors', '.pt', '.onnx', '.pdparams', '.sqlite3', '.db'}
        ],
        'largest': [
            {'path': str(p), 'bytes': p.stat().st_size}
            for p in sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:4]
        ],
        'top_level': [p.name for p in Path(root).iterdir()],
    }


if __name__ == '__main__':
    print(json.dumps([inventory(root) for root in
                     ['/models', '/alpr', '/training', '/shared', '/edge', '/demo']], indent=2))
