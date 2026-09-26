"""Public dataset metadata audit and bounded, resumable archive acquisition.

No credentials, third-party scripts or live Sentinel media. All artifacts stay
under the ignored benchmark tree. Dataset permissions must be checked first.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
import time
import urllib.request
import urllib.error
import urllib.parse
import zipfile
from pathlib import Path

KAGGLE = [
    'umar1103/final-licence', 'abtexp/synthetic-indian-license-plates',
    'santoshvishwakarma99/indian-license-plates', 'santoshvishwakarma99/indian-license-plate-dataset',
    'kedarsai/indian-license-plates-with-labels', 'ankitbarai507/indian-vehicle-license-plates',
    'tkm22092/indian-number-plate-images', 'chiraggodaw/license-plate-dataset-resized',
    'saisirishan/indian-vehicle-dataset', 'thamizhsterio/indian-license-plates',
    'gauravsanwal/indian-licence-plate', 'thekeh/indian-license-plate-letter-segmentation-dataset',
    'siddheshmm/synthetic-license-plates-indian', 'sunrajbishnolia/license-plate-detection',
    'dataclusterlabs/indian-number-plates-dataset', 'dataclusterlabs/electric-vehicle-number-plate-image-dataset',
    'vedantgupta007/indian-vehicle-license-plate-dataset-result', 'dataturks/vehicle-number-plate-detection',
    'paneraghanshyam/gujarat-vehicle-number-plates-yolo-ready', 'deepakat002/indian-vehicle-number-plate-yolo-annotation',
    'ckay16/indian-number-plate-detection', 'bellair/indian-licence-plates',
    'vishanktomar/indian-car-number-plate-detection', 'shaikhshahid/indian-car-number-plate',
]
HF = ['overfitowl/synthetic-indian-licenseplates-dataset', 'thundarstrom/indian-anpr-ocr-corpus',
      'Dataclusterlabspvtltd/indian-number-plates-dataset', 'adityaver/Indian-Military_and_Civilian-Plates',
      'UniqueData/license_plates', 'UniDataPro/license-plate-detection',
      'AnirudhLanka2002/TR_OCR_Training_License_Plate', 'adityamangal98/HR-LR-images-of-the-Indian-License-plates',
      'AY000554/Car_plate_OCR_dataset', 'thirdeyelabs/indian-road-dataset']
APPROVED_DOWNLOADS = {'umar1103/final-licence', 'abtexp/synthetic-indian-license-plates'}
TROCR_REF = 'AnirudhLanka2002/TR_OCR_Training_License_Plate'
TROCR_REVISION = 'b3cda9572099a7023cea30a2ee0537d0647fd62e'


def json_get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise ValueError('Unexpected metadata response')
        data = response.read(16 * 1024**2 + 1)
        if len(data) > 16 * 1024**2:
            raise ValueError('Metadata response exceeds bounds')
        return json.loads(data)


def audit(root):
    root.mkdir(parents=True, exist_ok=True)
    sources = []
    for ref in KAGGLE:
        item = {'platform': 'kaggle', 'ref': ref, 'url': f'https://www.kaggle.com/datasets/{ref}'}
        try:
            data = json_get(f'https://www.kaggle.com/api/v1/datasets/view/{ref}')
            item.update({key: data.get(key) for key in ('title', 'description', 'licenseName', 'totalBytes', 'currentVersionNumber')})
            files = json_get(f'https://www.kaggle.com/api/v1/datasets/list/{ref}')
            item['sample_files'] = [{'name': value['name'], 'bytes': value['totalBytes']}
                                    for value in files.get('datasetFiles', [])]
        except Exception as error:
            item['access_error'] = f'{type(error).__name__}: {error}'
        sources.append(item)
        (root / 'source_audit.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
        print(json.dumps({key: item.get(key) for key in ('ref', 'licenseName', 'totalBytes', 'access_error')}
                         | {'sample_files': item.get('sample_files', [])[:3]}), flush=True)
    for ref in HF:
        item = {'platform': 'huggingface', 'ref': ref, 'url': f'https://huggingface.co/datasets/{ref}'}
        try:
            data = json_get(f'https://huggingface.co/api/datasets/{ref}')
            item.update({key: data.get(key) for key in ('sha', 'cardData', 'siblings', 'usedStorage', 'gated')})
        except Exception as error:
            item['access_error'] = f'{type(error).__name__}: {error}'
        sources.append(item)
        (root / 'source_audit.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
        print(json.dumps({'ref': ref, 'sha': item.get('sha'), 'storage': item.get('usedStorage'),
                          'license': (item.get('cardData') or {}).get('license'),
                          'file_count': len(item.get('siblings', [])),
                          'access_error': item.get('access_error')}), flush=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'source_audit.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')


def download(url, target, *, max_bytes, md5=None):
    """Resume precise partial bytes; fail closed on HTML/error/disk pressure."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if target.stat().st_size > max_bytes:
            raise ValueError('Cached archive exceeds transfer bound')
        receipt_path = target.with_suffix('.receipt.json')
        if not receipt_path.is_file():
            raise ValueError('Cached archive has no acquisition receipt; review required')
        receipt = json.loads(receipt_path.read_text())
        digest = hashlib.sha256()
        publisher = hashlib.md5()
        with target.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024**2), b''):
                digest.update(chunk)
                publisher.update(chunk)
        if (receipt.get('url') != url or receipt.get('sha256') != digest.hexdigest()
                or receipt.get('bytes') != target.stat().st_size or (md5 and publisher.hexdigest() != md5)):
            raise ValueError('Cached archive receipt/checksum mismatch')
        with zipfile.ZipFile(target) as archive:
            if archive.testzip():
                raise ValueError('Cached ZIP CRC failed')
        print(json.dumps({'cached_archive': target.name, 'bytes': target.stat().st_size}), flush=True)
        return
    partial = target.with_suffix(target.suffix + '.partial')
    size = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={'Range': f'bytes={size}-'} if size else {})
    with urllib.request.urlopen(request, timeout=45) as response:
        if size and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {size}-')):
            raise ValueError('Download server cannot resume precisely; partial retained')
        if 'text/html' in response.headers.get('Content-Type', '').lower():
            raise ValueError('Dataset download returned HTML, not an archive')
        if int(response.headers.get('Content-Length', '0')) + size > max_bytes:
            raise ValueError('Archive exceeds transfer bound')
        if shutil.disk_usage(target.parent).free < 4 * 1024**3 + max_bytes - size:
            raise ValueError('Insufficient disk headroom; existing files preserved')
        tick, last = time.monotonic(), size
        with partial.open('ab' if size else 'wb') as stream:
            while chunk := response.read(1024**2):
                size += len(chunk)
                if size > max_bytes or shutil.disk_usage(target.parent).free < 4 * 1024**3:
                    raise ValueError('Transfer or disk bound reached; partial retained')
                stream.write(chunk)
                if size - last >= 16 * 1024**2:
                    print(json.dumps({'archive': target.name, 'downloaded_mib': round(size / 1024**2, 1),
                                      'elapsed_s': round(time.monotonic() - tick, 1)}), flush=True)
                    last = size
    if md5:
        digest = hashlib.md5()
        with partial.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024**2), b''):
                digest.update(chunk)
        if digest.hexdigest() != md5:
            raise ValueError('Publisher archive MD5 mismatch')
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip():
            raise ValueError('ZIP CRC verification failed')
    partial.replace(target)
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    (target.with_suffix('.receipt.json')).write_text(json.dumps({'url': url, 'bytes': size,
            'sha256': digest.hexdigest(), 'zip_crc_verified': True, 'publisher_md5_verified': bool(md5)}, indent=2))
    print(json.dumps({'download_complete': target.name, 'bytes': size, 'sha256': digest.hexdigest()}), flush=True)


def download_bapatla(target):
    """Small precise ranges tolerate this publisher's stalled long transfers."""
    url = 'https://zenodo.org/records/13954136/files/number_plate.zip?download=1'
    total, checksum = 1468393969, 'e0a3d1e121f52806abb65248c8b86149'
    if target.exists():
        return download(url, target, max_bytes=total, md5=checksum)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.zip.partial')
    size = partial.stat().st_size if partial.exists() else 0
    if size > total:
        raise ValueError('Partial archive exceeds publisher size')
    while size < total:
        if shutil.disk_usage(target.parent).free < 4 * 1024**3 + total - size:
            raise ValueError('Insufficient disk headroom; partial preserved')
        end = min(total, size + 4 * 1024**2) - 1
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={'Range': f'bytes={size}-{end}'})
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {size}-{end}/{total}':
                        raise ValueError('Publisher precise range mismatch')
                    data = response.read(end - size + 2)
                if len(data) != end - size + 1:
                    raise ValueError('Truncated range; partial preserved')
                break
            except (TimeoutError, urllib.error.URLError, ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        with partial.open('ab') as stream:
            stream.write(data)
        size = end + 1
        print(json.dumps({'archive': target.name, 'range_downloaded_mib': round(size / 1024**2, 1)}), flush=True)
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with partial.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            sha.update(chunk)
            md5.update(chunk)
    if md5.hexdigest() != checksum:
        raise ValueError('Completed publisher archive MD5 mismatch; not merged')
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip():
            raise ValueError('Completed publisher archive CRC failed')
    partial.replace(target)
    target.with_suffix('.receipt.json').write_text(json.dumps({'url': url, 'bytes': total,
        'sha256': sha.hexdigest(), 'zip_crc_verified': True, 'publisher_md5_verified': True}, indent=2))
    print(json.dumps({'download_complete': target.name, 'bytes': total, 'sha256': sha.hexdigest()}), flush=True)


def download_trocr(target):
    """Pinned public image/CSV snapshot, assembled locally without running code."""
    metadata = json_get(f'https://huggingface.co/api/datasets/{TROCR_REF}')
    if metadata.get('sha') != TROCR_REVISION or metadata.get('cardData', {}).get('license') != 'mit':
        raise ValueError('OCR source revision/licence changed; review required')
    if target.exists():
        receipt = json.loads(target.with_suffix('.receipt.json').read_text())
        if receipt.get('revision') != TROCR_REVISION:
            raise ValueError('Cached OCR source revision mismatch')
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != receipt['sha256']:
            raise ValueError('Cached OCR source checksum mismatch')
        print(json.dumps({'cached_archive': target.name}), flush=True)
        return
    members = [s['rfilename'] for s in metadata['siblings'] if
               s['rfilename'].startswith('Exported_TROCR/cropped_images/') and s['rfilename'].endswith('.jpg')]
    members += ['Exported_TROCR/lpr.csv', 'README.md']
    if not 2 < len(members) <= 500 or any('..' in Path(n).parts for n in members):
        raise ValueError('OCR snapshot file bounds violated')
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < 4 * 1024**3 + 64 * 1024**2:
        raise ValueError('Insufficient OCR snapshot disk headroom')
    def fetch(name):
        url = f'https://huggingface.co/datasets/{TROCR_REF}/resolve/{TROCR_REVISION}/{urllib.parse.quote(name)}'
        with urllib.request.urlopen(url, timeout=30) as response:
            if 'text/html' in response.headers.get('Content-Type', ''):
                raise ValueError('OCR snapshot returned HTML')
            data = response.read(256 * 1024 + 1)
        if len(data) > 256 * 1024:
            raise ValueError('OCR snapshot member exceeds bounds')
        return name, data
    with ThreadPoolExecutor(max_workers=6) as workers:
        values = list(workers.map(fetch, members))
    if sum(len(data) for _, data in values) > 64 * 1024**2:
        raise ValueError('OCR snapshot total exceeds bounds')
    partial = target.with_suffix('.zip.partial')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in values:
            archive.writestr(name, data)
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip():
            raise ValueError('OCR snapshot CRC failed')
    partial.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.receipt.json').write_text(json.dumps({
        'url': f'https://huggingface.co/datasets/{TROCR_REF}/tree/{TROCR_REVISION}',
        'revision': TROCR_REVISION, 'license': 'MIT', 'bytes': target.stat().st_size,
        'sha256': digest, 'zip_crc_verified': True, 'locally_assembled_snapshot': True,
        'members': [{'name': name, 'sha256': hashlib.sha256(data).hexdigest()} for name, data in values],
    }, indent=2))
    print(json.dumps({'download_complete': target.name, 'files': len(values), 'bytes': target.stat().st_size}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--kaggle')
    parser.add_argument('--bapatla', action='store_true')
    parser.add_argument('--mendeley-characters', action='store_true')
    parser.add_argument('--hf-trocr', action='store_true')
    args = parser.parse_args()
    if args.audit:
        audit(args.root)
    elif args.kaggle:
        if args.kaggle not in APPROVED_DOWNLOADS:
            raise ValueError('Source audited but not approved for this OCR corpus')
        metadata = json_get(f'https://www.kaggle.com/api/v1/datasets/view/{args.kaggle}')
        if metadata['licenseName'] not in {'Attribution 4.0 International (CC BY 4.0)', 'CC0: Public Domain'}:
            raise ValueError('Source licence requires a separate permission review')
        if metadata['currentVersionNumber'] != 1:
            raise ValueError('Publisher version changed; review before downloading')
        destination = args.root / 'archives' / (args.kaggle.replace('/', '-') + '.zip')
        download(f'https://www.kaggle.com/api/v1/datasets/download/{args.kaggle}?datasetVersionNumber={metadata["currentVersionNumber"]}',
                 destination, max_bytes=2 * 1024**3)
    elif args.bapatla:
        download_bapatla(args.root / 'archives' / 'bapatla.zip')
    elif args.hf_trocr:
        download_trocr(args.root / 'archives' / 'hf-indian-trocr.zip')
    elif args.mendeley_characters:
        # Exact public file link observed on the publisher's download page;
        # source page explicitly declares CC BY 4.0. No guessed access endpoint.
        download('https://data.mendeley.com/public-files/datasets/znghtgfnnf/files/6d2ae1a3-3de8-45cf-b7c0-2da9ec6317bb/file_downloaded',
                 args.root / 'archives' / 'mendeley-characters.zip', max_bytes=500 * 1024**2)
    else:
        parser.error('Select audit or an approved source download')
