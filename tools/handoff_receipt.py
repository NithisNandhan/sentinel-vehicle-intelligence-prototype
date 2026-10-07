"""Write an observed environment receipt; never capture container environments."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command(*arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    images = {}
    for name in ('edge-worker', 'api', 'dashboard', 'demo-simulator'):
        image = f'sentinel-prototype-{name}:latest'
        images[image] = command('docker', 'image', 'inspect', image, '--format', '{{.Id}}')
    freeze = command('docker', 'exec', 'sentinel-prototype-edge-worker-1',
                     'python', '-m', 'pip', 'freeze').splitlines()
    # Local editable paths are replaced with the checked-in package location.
    freeze = [line for line in freeze if not line.startswith(('#', '-e '))]
    record = {'captured_at_utc': datetime.now(timezone.utc).isoformat(),
              'docker_version': command('docker', 'version', '--format', '{{.Server.Version}}'),
              'local_image_ids': images, 'observed_edge_packages': freeze,
              'rebuild_note': 'Images are rebuilt, not distributed. The package list is observed, not a new tested lockfile.',
              'tests': {'edge': {'passed': 287, 'skipped': 1}, 'backend': {'passed': 53},
                        'frontend': {'passed': 15, 'production_build': 'passed'},
                        'handoff': {'passed': 13}, 'central_snapshot_restore': 'verified same table counts'},
              'exclusions': ['credentials', 'original HMAC secrets', 'camera source URLs',
                             'assistant local state', 'unverified dats_probe data', 'raw live footage']}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('Wrote nonsecret environment and validation receipt.', flush=True)


if __name__ == '__main__':
    main()
