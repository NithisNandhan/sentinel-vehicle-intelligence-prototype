"""Cache only pinned official/research ONNX assets for an isolated comparison."""
import hashlib
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path('/models/experiments/detectors'); root.mkdir(parents=True, exist_ok=True)
    repo = 'morsetechlab/yolov11-license-plate-detection'
    with urllib.request.urlopen('https://huggingface.co/api/models/'+repo, timeout=30) as response:
        revision = json.load(response)['sha']
    with urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'/tree/'+revision+'?expand=true', timeout=30) as response:
        files = {row['path']: row for row in json.load(response)}
    def fetch(scale):
        name = f'license-plate-finetune-v1{scale}.onnx'
        info = files[name]
        expected = info['lfs']['oid']
        path = root/name
        if not path.is_file() or digest(path) != expected:
            temporary = path.with_suffix('.download')
            with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', timeout=120) as response, temporary.open('wb') as target:
                shutil.copyfileobj(response, target)
            if digest(temporary) != expected:
                raise ValueError('Model checksum mismatch')
            temporary.replace(path)
        print('CACHED', name, path.stat().st_size, flush=True)
        return {'name': name, 'sha256': expected, 'revision': revision, 'source': repo, 'license': 'AGPL-3.0'}
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, 'mlx'))
    from open_image_models.detection.core.hub import download_model
    for name in ('yolo-v9-t-512-license-plate-end2end', 'yolo-v9-s-608-license-plate-end2end'):
        path = download_model(name, save_directory=root)
        records.append({'name': path.name, 'sha256': digest(path), 'source': 'ankandrew/open-image-models', 'license': 'MIT'})
        print('CACHED', name, flush=True)
    from ultralytics.utils.downloads import attempt_download_asset
    for name in ('yolo26n.pt', 'yolo26s.pt'):
        path = Path(attempt_download_asset(root/name, release='v8.4.0'))
        records.append({'name': name, 'sha256': digest(path), 'source': 'ultralytics/assets', 'release': 'v8.4.0', 'license': 'AGPL-3.0'})
        print('CACHED', name, flush=True)
    records.append({'name': 'yolov8n.pt', 'sha256': digest(Path('/models/yolov8n.pt')), 'source': 'existing local baseline'})
    (root/'assets.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__': main()
