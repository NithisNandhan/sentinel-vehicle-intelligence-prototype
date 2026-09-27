"""Freeze small visually reviewed diagnostics before candidate inference.

Approximate AI-reviewed target boxes, not exhaustive independent human labels.
Only previously permitted Bapatla/NIVU originals; never Sentinel media.
"""
import hashlib
import json
from pathlib import Path


BOXES = {
    'AN1': ([15,140,265,406], [95,365,185,387]),
    'BR4': ([0,90,248,362], [39,324,136,347]),
    'CH3': ([17,97,244,301], [54,271,145,291]),
    'DN1': ([11,30,236,255], [96,208,172,226]),
    'GA1': ([39,115,211,263], [129,207,178,225]),
    'HP2': ([0,0,272,324], [112,296,233,317]),
    'JH1': ([0,0,272,363], [76,331,224,354]),
    'KL3': ([7,60,240,259], [37,213,88,241]),
    'MP1': ([6,110,262,337], [91,270,170,288]),
    'OD3': ([54,5,216,136], [109,83,165,98]),
}


def main():
    root = Path('/workspace/data/benchmarks')
    rows = []
    for name, (vehicle, plate) in BOXES.items():
        path = root/'bapatla/images'/f'{name}.png'
        rows.append({'id': name, 'vehicle_id': name, 'path': str(path.relative_to(root)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'vehicle_box': vehicle, 'plate_box': plate, 'license': 'CC-BY-4.0',
            'source_url': 'https://zenodo.org/records/13954136',
            'review': 'AI visually checked approximate target boxes before candidate inference'})
    original = json.loads((root/'nivu/manifest.json').read_text())
    for frame in original['passages'][0]['frames']:
        path = root/'nivu'/frame['path']
        rows.append({'id': path.stem, 'vehicle_id': 'nivu-video10',
            'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'vehicle_box': frame['vehicle_box'], 'plate_box': frame['plate_box'],
            'license': 'CC-BY-4.0', 'source_url': original['dataset']['source_url'],
            'review': 'Existing development-visible approximate boxes; not new held-out data'})
    document = {'schema_version': 1, 'permission': 'noncommercial_benchmark',
        'annotation_scope': 'one target vehicle and plate per image; background objects not exhaustively labelled',
        'independent_human_ground_truth': False, 'held_out_accuracy_benchmark': False,
        'sources': {'Bapatla': 'Bhuma Chandra Mohan et al., 2024; CC BY 4.0',
                    'NIVU': 'NIVU Indian License Plate Dataset, 2022; CC BY 4.0'},
        'samples': rows}
    target = root/'detector-comparison'; target.mkdir(exist_ok=True)
    (target/'manifest.json').write_text(json.dumps(document, indent=2))
    print('FROZEN_DIAGNOSTIC_IMAGES', len(rows), 'DISTINCT_TARGET_VEHICLES', len({r['vehicle_id'] for r in rows}))


if __name__ == '__main__': main()
