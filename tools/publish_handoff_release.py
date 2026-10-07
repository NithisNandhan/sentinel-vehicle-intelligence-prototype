"""Publish private-repository handoff assets using existing Git credentials.

Authentication is obtained in memory from Git Credential Manager, never printed
or written to a file. Existing assets are verified rather than overwritten.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import mimetypes
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def git_token(root):
    result = subprocess.run(
        ['git', '-c', f'safe.directory={Path(root).resolve().as_posix()}',
         '-c', 'credential.interactive=false', 'credential', 'fill'],
        input='protocol=https\nhost=github.com\n\n', text=True,
        capture_output=True, check=True)
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if not fields.get('password'):
        raise RuntimeError('No existing authenticated GitHub credential is available')
    return fields['password']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class GitHub:
    def __init__(self, token, repository):
        self.token, self.repository = token, repository
        self.root = f'https://api.github.com/repos/{repository}'

    def request(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {'Authorization': f'Bearer {self.token}', 'Accept': 'application/vnd.github+json',
                   'User-Agent': 'sentinel-handoff', 'X-GitHub-Api-Version': '2022-11-28'}
        if data is not None:
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request(self.root + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise RuntimeError(f'GitHub API request failed: HTTP {exc.code} ({path})') from None

    def upload(self, release, path):
        path = Path(path)
        url = release['upload_url'].split('{', 1)[0]
        if urllib.parse.urlparse(url).hostname != 'uploads.github.com':
            raise ValueError('Unexpected GitHub upload host')
        url += '?' + urllib.parse.urlencode({'name': path.name})
        size = path.stat().st_size
        print(f'Uploading {path.name} ({size / 1024 / 1024:.1f} MiB)', flush=True)
        with path.open('rb') as stream:
            request = urllib.request.Request(url, data=stream, method='POST', headers={
                'Authorization': f'Bearer {self.token}', 'User-Agent': 'sentinel-handoff',
                'Accept': 'application/vnd.github+json', 'Content-Length': str(size),
                'Content-Type': mimetypes.guess_type(path.name)[0] or 'application/octet-stream'})
            try:
                with urllib.request.urlopen(request, timeout=1800) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                raise RuntimeError(f'Asset upload failed: HTTP {exc.code} ({path.name})') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', default='NithisNandhan/sentinel-vehicle-intelligence-prototype')
    parser.add_argument('--root', default='.')
    parser.add_argument('--assets', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--notes', required=True)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--stage-completed', action='store_true',
                        help='Upload only closed completed parts to a draft while other exports run.')
    args = parser.parse_args()
    root, assets = Path(args.root).resolve(), Path(args.assets).resolve()
    github = GitHub(git_token(root), args.repository)
    repository = github.request('GET', '')
    if not repository or not repository.get('private'):
        raise ValueError('This complete handoff is only authorised for the existing private repository')
    commit = github.request('GET', '/commits/' + args.target)
    if not commit or commit['sha'] != args.target:
        raise ValueError('Target commit must already be pushed and match the requested SHA')
    # All archive manifests must exist before any partial archive can be uploaded.
    expected = set()
    for manifest_path in assets.glob('*.manifest.json'):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for entry in manifest['parts']:
            expected.add(entry['name'])
            path = assets / entry['name']
            if path.stat().st_size != entry['bytes']:
                raise ValueError(f'Incomplete archive part: {path.name}')
    actual = {p.name for p in assets.glob('*.part*')}
    if args.stage_completed and args.publish:
        raise ValueError('Incomplete staging cannot publish a release')
    if actual != expected and not args.stage_completed:
        raise ValueError('Archive inventory is incomplete; unfinished parts cannot be published')
    release = github.request('GET', '/releases/tags/' + urllib.parse.quote(args.tag, safe=''))
    if release is None:
        release = github.request('POST', '/releases', {
            'tag_name': args.tag, 'target_commitish': args.target,
            'name': 'Complete teammate handoff — 7 October 2026', 'draft': True,
            'body': Path(args.notes).read_text(encoding='utf-8')})
    uploaded = {}
    page = 1
    while True:
        batch = github.request('GET', f'/releases/{release["id"]}/assets?per_page=100&page={page}')
        for asset in batch:
            uploaded[asset['name']] = asset
        if len(batch) < 100:
            break
        page += 1
    files = sorted(p for p in assets.iterdir() if p.is_file() and not p.name.endswith('.partial'))
    if args.stage_completed:
        # A successor part exists only after PartWriter closes its predecessor.
        # Never upload the final part of an archive lacking a complete manifest.
        unfinished = {}
        for path in files:
            if '.tar.gz.part' in path.name and path.name not in expected:
                group = path.name.split('.tar.gz.part', 1)[0]
                unfinished.setdefault(group, []).append(path)
        incomplete = {parts[-1].name for parts in unfinished.values()}
        files = [path for path in files if path.name not in incomplete]
    def publish_file(path):
        local_hash = sha256(path)
        existing = uploaded.get(path.name)
        if existing:
            if existing.get('size') != path.stat().st_size or existing.get('digest') != 'sha256:' + local_hash:
                raise ValueError(f'Existing asset differs; refusing overwrite: {path.name}')
            print(f'Already verified: {path.name}', flush=True)
            return
        asset = github.upload(release, path)
        for attempt in range(5):
            if asset.get('digest'):
                break
            time.sleep(2)
            asset = github.request('GET', f'/releases/assets/{asset["id"]}')
        if asset['size'] != path.stat().st_size or asset.get('digest') != 'sha256:' + local_hash:
            raise ValueError(f'GitHub upload integrity mismatch: {path.name}')
        print(f'Verified GitHub SHA256: {path.name}', flush=True)
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as executor:
        list(executor.map(publish_file, files))
    if args.publish:
        github.request('PATCH', f'/releases/{release["id"]}', {'draft': False})
    print(json.dumps({'release': release['html_url'], 'files': len(files),
                      'published': args.publish, 'target_commit': args.target}), flush=True)


if __name__ == '__main__':
    main()
