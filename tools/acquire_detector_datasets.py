"""Acquire specifically audited detection datasets, never live CCTV recordings.

Publisher declarations are preserved, not proof of every image's provenance.
No third-party scripts are executed and no authentication barriers bypassed.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from acquire_plate_corpus import download, json_get

ROOT = Path('/workspace/data/benchmarks/detector-training')
KAGGLE = {
    'gujarat': ('paneraghanshyam/gujarat-vehicle-number-plates-yolo-ready', 'Apache 2.0'),
    'make-ml': ('andrewmvd/car-plate-detection', 'CC0: Public Domain'),
}


def kaggle(name, ref, expected):
    metadata = json_get('https://www.kaggle.com/api/v1/datasets/view/'+ref)
    if metadata.get('licenseName') != expected or metadata.get('isPrivate'):
        raise ValueError('Dataset permission changed')
    version = metadata['currentVersionNumber']
    (ROOT/(name+'-metadata.json')).write_text(json.dumps(metadata, indent=2))
    download(f'https://www.kaggle.com/api/v1/datasets/download/{ref}?datasetVersionNumber={version}',
             ROOT/'archives'/(name+'.zip'), max_bytes=512*1024**2)
    return {'name': name, 'source': 'https://www.kaggle.com/datasets/'+ref,
            'version': version, 'license': expected, 'status': 'acquired'}


def madrid():
    metadata = json_get('https://zenodo.org/api/records/10912886')
    if metadata['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('Dataset permission changed')
    (ROOT/'madrid-metadata.json').write_text(json.dumps(metadata, indent=2))
    # Keep auto-labels separate: do not train or score on them as human truth.
    for item in metadata['files']:
        if item['key'] == 'auto-labeled.zip':
            continue
        download(item['links']['self'], ROOT/'archives'/('madrid-'+item['key']),
                 max_bytes=64*1024**2, md5=item['checksum'].removeprefix('md5:'))
    return {'name': 'madrid', 'source': 'https://zenodo.org/records/10912886',
            'license': 'CC-BY-4.0', 'status': 'manual_partitions_acquired'}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = [(name, lambda n=name, r=ref, e=licence: kaggle(n,r,e))
            for name,(ref,licence) in KAGGLE.items()] + [('madrid',madrid)]
    results = []
    def run(job):
        name, callback = job
        try:
            return callback()
        except Exception as error:
            return {'name':name, 'status':'unavailable', 'error_type':type(error).__name__}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(run,jobs):
            results.append(result); print(json.dumps(result),flush=True)
            (ROOT/'acquisition.json').write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
