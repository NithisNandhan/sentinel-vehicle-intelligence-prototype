"""Validate a recorded-footage label file: schema, splits, leaks, duplicates, agreement, manifest hashes.

    python tools/validate_annotations.py --labels data/labels/<file>.json [--manifest out.json]

Exit status 1 when any error is found. Never edits labels. See docs/ANNOTATION_GUIDE.md.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'edge'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, help='Write the pinned manifest (label and recording hashes) here')
    args = parser.parse_args()
    from sentinel_edge.annotations import manifest, validate
    report = validate(args.labels)
    print(json.dumps(report, indent=2))
    if args.manifest:
        with args.manifest.open('x', encoding='utf-8') as stream:
            json.dump(manifest(args.labels), stream, indent=2)
    raise SystemExit(0 if report.get('valid') else 1)
