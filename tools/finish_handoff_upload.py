"""Finish an approved private handoff independently of an interactive terminal.

Run once after starting the benchmark export. Logs/status belong in ignored data/.
No credentials are persisted. Integrity errors stop the job instead of overwriting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

import publish_handoff_release as publisher

REQUIRED_BUNDLES = {'model-cache', 'alpr-cache', 'detector-training',
                    'benchmark-data', 'saved-events', 'presentation-assets'}


def complete_inventory(directories):
    names = {path.name.removesuffix('.manifest.json')
             for directory in directories for path in directory.glob('*.manifest.json')}
    return REQUIRED_BUNDLES <= names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', action='append', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--release-id', required=True, type=int)
    parser.add_argument('--status', required=True)
    parser.add_argument('--export-container', default='sentinel-handoff-data-export')
    args = parser.parse_args()
    directories = [Path(value).resolve() for value in args.assets]
    status = Path(args.status).resolve()

    def report(state, **details):
        value = {'state': state, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 'release_id': args.release_id, 'target': args.target, **details}
        temporary = status.with_suffix('.partial')
        temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        temporary.replace(status)
        print(json.dumps(value), flush=True)

    base = ['publish_handoff_release.py', '--target', args.target,
            '--release-id', str(args.release_id), '--tag', 'handoff-2026-10-07',
            '--notes', 'docs/TEAMMATE_RELEASE_NOTES.md', '--workers', '4',
            '--recover-starter-assets']
    for directory in directories:
        base.extend(['--assets', str(directory)])

    def upload(publish):
        sys.argv = base + (['--publish'] if publish else ['--stage-completed'])
        for attempt in range(1, 6):
            report('publishing' if publish else 'uploading-completed-parts', attempt=attempt)
            try:
                publisher.main()
                return
            except (RuntimeError, urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == 5:
                    raise
                # Only retry transport/API failures; ValueError integrity failures never retry.
                report('network-retry', attempt=attempt, error=type(exc).__name__)
                time.sleep(60)

    try:
        if not complete_inventory(directories):
            upload(False)
        deadline = time.monotonic() + 12 * 60 * 60
        while not complete_inventory(directories):
            probe = subprocess.run(
                ['docker', 'inspect', '-f', '{{.State.Running}}', args.export_container],
                capture_output=True, text=True, timeout=30)
            if probe.returncode or probe.stdout.strip() != 'true':
                # Account for the exporter writing its manifest just before exiting.
                if complete_inventory(directories):
                    break
                raise ValueError('Benchmark exporter stopped before completing its manifest')
            if time.monotonic() > deadline:
                raise TimeoutError('Benchmark export exceeded the 12-hour completion limit')
            report('waiting-for-benchmark-manifest')
            time.sleep(60)
        upload(True)
        report('published', release_url=(
            'https://github.com/NithisNandhan/sentinel-vehicle-intelligence-prototype/'
            'releases/tag/handoff-2026-10-07'))
    except Exception as exc:
        report('failed', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
