"""Licensed, provenance-preserving Indian OCR corpus preparation, not training.

Original archives stay unchanged. PNG exports preserve decoded crop pixels.
Publisher labels are not promoted to human ground truth, synthetic examples
are not real vehicles, and legacy benchmark registrations never enter training.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import html
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

REGIONS = set('AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR PB PY RJ SK TN TR TS UK UP WB'.split())
SOURCES = {
    'trocr': {'archive': 'hf-indian-trocr.zip', 'license': 'MIT',
              'url': 'https://huggingface.co/datasets/AnirudhLanka2002/TR_OCR_Training_License_Plate/tree/b3cda9572099a7023cea30a2ee0537d0647fd62e',
              'attribution': 'AnirudhLanka2002; TR OCR Training License Plate; revision b3cda9572099a7023cea30a2ee0537d0647fd62e',
              'origin': 'mixed_origin_unverified', 'label_kind': 'publisher_csv',
              'origin_caution': 'Publisher declares MIT; underlying photo sources and human versus automated label creation unspecified'},
    'abtexp': {'archive': 'abtexp-synthetic-indian-license-plates.zip', 'license': 'CC0-1.0',
               'url': 'https://www.kaggle.com/datasets/abtexp/synthetic-indian-license-plates',
               'attribution': 'abtexp; Synthetic Indian License Plates, version 1',
               'origin': 'synthetic', 'label_kind': 'generated_filename'},
    'umar': {'archive': 'umar1103-final-licence.zip', 'license': 'CC-BY-4.0',
             'url': 'https://www.kaggle.com/datasets/umar1103/final-licence/versions/1',
             'attribution': 'umar1103; Final Licence, version 1',
             'origin': 'mixed_origin_unverified', 'label_kind': 'publisher_filename'},
    'bapatla': {'archive': 'bapatla.zip', 'license': 'CC-BY-4.0',
                'url': 'https://zenodo.org/records/13954136',
                'attribution': 'Bhuma Chandra Mohan; CH V M S N Pavan Kumar; Katakam; Bethala Venkata Varma; Vatada Sai; Kagitha Sambaiah; Bapatla Engineering College, 2024',
                'origin': 'photograph_publisher_declared', 'label_kind': 'publisher_workbook',
                'origin_caution': 'Individual photograph acquisition and upstream permissions not independently established'},
}
MIN_FREE = 4 * 1024**3
MAX_IMAGE = 12 * 1024**2


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_label(value):
    # Never guess O/0, B/8 or a missing character from Indian syntax.
    return re.sub(r'[\s-]', '', str(value)).upper()


def indian_style(label):
    """Eligibility filter, NOT proof of a legal/real registration or completeness."""
    return bool((label[:2] in REGIONS and re.fullmatch(r'[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}', label))
                or re.fullmatch(r'\d{2}BH\d{4}[A-Z]{1,2}', label))


def safe_member(info):
    name = info.filename.replace('\\', '/')
    parts = PurePosixPath(name)
    if (parts.is_absolute() or '..' in parts.parts or ':' in name
            or (info.external_attr >> 16) & 0o170000 == 0o120000):
        raise ValueError('Unsafe archive member')
    if info.file_size > MAX_IMAGE or info.file_size > max(1, info.compress_size) * 500:
        raise ValueError('Archive member exceeds bounds')
    return name


def pixel_hash(image):
    image = image.convert('RGB')
    return hashlib.sha256(f'{image.width}x{image.height}:RGB:'.encode() + image.tobytes()).hexdigest()


def split_for(label, *, synthetic=False, legacy=None, reserved=()):
    if label in reserved:
        return 'regression_only'
    if legacy and label in legacy:
        return {'test': 'legacy_test', 'validation': 'legacy_validation', 'train': 'train'}[legacy[label]]
    # Real and synthetic copies of the same registration cannot straddle splits.
    # Synthetic-only holdouts are not used to report real-camera accuracy.
    bucket = int(hashlib.sha256(('sentinel-corpus-v1:' + label).encode()).hexdigest()[:8], 16) % 10
    return 'holdout_needs_review' if bucket == 0 else 'train'


class Corpus:
    def __init__(self, root, *, legacy=None, reserved=(), min_free=MIN_FREE):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        self.rejected = []
        self.legacy = legacy or {}
        self.reserved = set(reserved)
        self.min_free = min_free
        self.counts = Counter()

    def load_existing(self):
        """Extend a completed merge without silently trusting an altered manifest."""
        summary = json.loads((self.root / 'summary.json').read_text())
        path = self.root / 'corpus.jsonl'
        if summary.get('schema_version') != 1 or file_hash(path) != summary['corpus_sha256']:
            raise ValueError('Existing corpus manifest integrity mismatch')
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                # Canonical content-addressed paths can be validated without
                # ~30k slow Windows/WSL metadata round trips. No crop is read
                # here; verify_corpus resolves paths and hashes every file.
                expected = f"crops/{row['id'][:2]}/{row['id']}.png"
                if (not re.fullmatch(r'[a-f0-9]{64}', row['id']) or row['crop_path'] != expected
                        or len(self.rows) >= 50_000):
                    raise ValueError('Existing corpus crop path/count invalid')
                if row['id'] in self.rows or row['id'] != row['decoded_pixel_sha256']:
                    raise ValueError('Existing corpus contains invalid identities')
                self.rows[row['id']] = row
        if len(self.rows) != summary['unique_crops']:
            raise ValueError('Existing corpus summary count mismatch')
        self.counts.update(summary['observations'])
        with (self.root / 'rejected.jsonl').open(encoding='utf-8') as stream:
            self.rejected = [json.loads(line) for line in stream]
        return {p['source'] for row in self.rows.values() for p in row['provenance']}

    def add(self, image, label, provenance, *, review='publisher_unverified', quarantined=False):
        label = normalized_label(label)
        if not indian_style(label):
            self.reject(provenance, 'unsupported_or_non_indian_label')
            return
        image = image.convert('RGB')
        if image.width * image.height > 20_000_000 or image.width < 16 or image.height < 8:
            self.reject(provenance, 'invalid_or_tiny_crop')
            return
        key = pixel_hash(image)
        self.counts[provenance['source'] + '_eligible_observations'] += 1
        if key in self.rows:
            row = self.rows[key]
            row['provenance'].append(provenance)
            row['label_claims'] = sorted(set(row['label_claims']) | {label})
            row['quarantined'] |= quarantined or len(row['label_claims']) > 1
            if review == 'visually_verified' and not row['quarantined']:
                row['review'] = review
            # An exact match to the declared generator resolves mixed origin.
            if provenance['origin'] == 'synthetic':
                row['origin'] = 'synthetic'
                if row['review'] != 'visually_verified':
                    row['review'] = 'synthetic_generated'
            self.counts['exact_decoded_duplicates'] += 1
            return
        destination = self.root / 'crops' / key[:2] / (key + '.png')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.root).free < self.min_free + MAX_IMAGE:
            raise ValueError('Disk headroom reached; existing artifacts preserved')
        if not destination.exists():
            image.save(destination, format='PNG')
        else:
            from PIL import Image
            with Image.open(destination) as cached:
                if pixel_hash(cached) != key:
                    raise ValueError('Cached corpus crop was modified')
        self.rows[key] = {
            'id': key, 'crop_path': destination.relative_to(self.root).as_posix(),
            'crop_sha256': file_hash(destination), 'decoded_pixel_sha256': key,
            'transcription': label, 'label_claims': [label],
            'width': image.width, 'height': image.height, 'review': review,
            'origin': provenance['origin'], 'quarantined': quarantined,
            'provenance': [provenance],
        }

    def reject(self, provenance, reason):
        self.rejected.append({'provenance': provenance, 'reason': reason})
        self.counts['rejected_' + reason] += 1

    def finish(self):
        # Conflicts poison the registration too: do not train a differently
        # encoded image of a disputed label in another source or split.
        conflicted = {label for row in self.rows.values() if row['quarantined'] for label in row['label_claims']}
        rows = sorted(self.rows.values(), key=lambda row: row['id'])
        for row in rows:
            row['quarantined'] |= bool(set(row['label_claims']) & conflicted)
            row['split'] = ('quarantine' if row['quarantined'] else
                            split_for(row['transcription'], synthetic=row['origin'] == 'synthetic',
                                      legacy=self.legacy, reserved=self.reserved))
            row['training_eligible'] = row['split'] == 'train'
            row['independent_evaluation_eligible'] = False
        def dump(name, values):
            with (self.root / name).open('w', encoding='utf-8') as stream:
                for row in values:
                    stream.write(json.dumps(row, ensure_ascii=True) + '\n')
        dump('corpus.jsonl', rows)
        dump('train_photographs.jsonl', [r for r in rows if r['training_eligible'] and r['origin'] == 'photograph_publisher_declared'])
        dump('train_synthetic.jsonl', [r for r in rows if r['training_eligible'] and r['origin'] == 'synthetic'])
        dump('train_origin_unverified.jsonl', [r for r in rows if r['training_eligible'] and r['origin'] == 'mixed_origin_unverified'])
        dump('review_queue.jsonl', [r for r in rows if r['review'] != 'visually_verified' or r['quarantined']])
        dump('holdout_needs_review.jsonl', [r for r in rows if r['split'] == 'holdout_needs_review'])
        dump('quarantine.jsonl', [r for r in rows if r['quarantined']])
        dump('rejected.jsonl', self.rejected)
        summary = {
            'schema_version': 1, 'unique_crops': len(rows),
            'unique_registration_strings': len({r['transcription'] for r in rows}),
            'origin_counts': dict(Counter(r['origin'] for r in rows)),
            'review_counts': dict(Counter(r['review'] for r in rows)),
            'split_counts': dict(Counter(r['split'] for r in rows)),
            'training_by_origin': dict(Counter(r['origin'] for r in rows if r['training_eligible'])),
            'native_width_bins_by_origin': {
                origin: dict(Counter('under_32' if r['width'] < 32 else '32_to_63' if r['width'] < 64
                                     else '64_to_127' if r['width'] < 128 else '128_and_above'
                                     for r in rows if r['origin'] == origin))
                for origin in sorted({r['origin'] for r in rows})},
            'observations': dict(self.counts),
            'sources': SOURCES,
            'merged_sources': sorted({p['source'] for row in rows for p in row['provenance']}),
            'policies': {
                'label_verification': 'Only explicit prior human review is visually_verified; publisher/generated labels remain distinct',
                'duplicates': 'Exact RGB decoded pixels merged; conflicts quarantine all affected registrations',
                'split': 'SHA256 registration grouping before augmentation; legacy validation/test and NIVU excluded from training',
                'synthetic_batch_fraction_max_recommended': 0.25,
                'unknown_origin_training': 'Separate opt-in manifest, excluded from recommended real/synthetic mixture',
                'evaluation': 'No new independently reviewed test set; holdout_needs_review is not ground truth',
                'temporal': 'These are stills, not multi-frame observations; augmentation is training-only',
                'near_duplicates': 'Same-registration variants grouped; cross-registration perceptual duplicates not comprehensively audited',
                'pretraining_overlap': 'Unknown; no assertion that publisher pretrained OCR has never seen these images',
                'licensing': 'Publisher declarations retained per observation; combined corpus is not relicensed as project MIT',
            },
            'corpus_sha256': file_hash(self.root / 'corpus.jsonl'),
        }
        (self.root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return summary


def archive_rows(root, source, *, selected_members=None):
    metadata = SOURCES[source]
    path = Path(root) / 'archives' / metadata['archive']
    receipt_path = path.with_suffix('.receipt.json')
    receipt = json.loads(receipt_path.read_text())
    if (not receipt.get('zip_crc_verified') or receipt['bytes'] != path.stat().st_size
            or receipt['sha256'] != file_hash(path)):
        raise ValueError('Archive receipt/hash mismatch')
    if source == 'bapatla' and not receipt.get('publisher_md5_verified'):
        raise ValueError('Bapatla publisher checksum must be verified before merging')
    with zipfile.ZipFile(path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) > 50_000 or sum(info.file_size for info in members) > 4 * 1024**3:
            raise ValueError('Archive expansion exceeds corpus bounds')
        for info in members:
            safe_member(info)
        labels = None
        if source == 'bapatla':
            import openpyxl
            books = [m for m in members if PurePosixPath(m.filename).name == 'number_plate.xlsx']
            if len(books) != 1:
                raise ValueError('Expected one publisher workbook')
            workbook = openpyxl.load_workbook(io.BytesIO(archive.read(books[0])), read_only=True, data_only=True)
            labels = defaultdict(set)
            for name, label, *_ in list(workbook.active.values)[1:]:
                labels[re.sub(r'[ ()]', '', Path(str(name)).stem).upper()].add(normalized_label(label))
            workbook.close()
        elif source == 'trocr':
            labels = defaultdict(set)
            text = archive.read('Exported_TROCR/lpr.csv').decode('utf-8-sig')
            for row in csv.DictReader(io.StringIO(text, newline=None)):
                if row.get('Filepath') and row.get('ObjectName'):
                    labels[Path(row['Filepath']).stem.upper()].add(normalized_label(row['ObjectName']))
        for info in members:
            if Path(info.filename).suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                continue
            if selected_members is not None and info.filename not in selected_members:
                continue
            stem = Path(info.filename).stem
            key = re.sub(r'[ ()]', '', stem).upper()
            claims = labels.get(key, set()) if labels is not None else {normalized_label(stem)}
            provenance = {'source': source, 'member': info.filename, 'archive_sha256': receipt['sha256'],
                          'license': metadata['license'], 'source_url': metadata['url'],
                          'attribution': metadata['attribution'], 'origin': metadata['origin'],
                          'label_kind': metadata['label_kind']}
            if len(claims) != 1:
                yield None, '', dict(provenance, rejection='missing_or_conflicting_workbook_mapping')
                continue
            label = next(iter(claims))
            if not indian_style(label):
                yield None, label, dict(provenance, rejection='unsupported_or_non_indian_label')
                continue
            data = archive.read(info)  # zipfile checks member CRC, never extracts paths.
            provenance['original_sha256'] = hashlib.sha256(data).hexdigest()
            yield data, label, provenance


def orient_native(image):
    """Lossless integer pixel permutations for standard EXIF orientations."""
    from PIL import ImageOps
    orientation = image.getexif().get(274, 1)
    if orientation not in range(1, 9):
        raise ValueError('Unsupported EXIF orientation')
    return ImageOps.exif_transpose(image), orientation


def verify_corpus(root):
    from PIL import Image
    root = Path(root).resolve()
    summary = json.loads((root / 'summary.json').read_text())
    if file_hash(root / 'corpus.jsonl') != summary['corpus_sha256']:
        raise ValueError('Corpus manifest checksum mismatch')
    rows, groups = {}, {}
    with (root / 'corpus.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            path = (root / row['crop_path']).resolve()
            expected_path = f"crops/{row['id'][:2]}/{row['id']}.png"
            if (not path.is_relative_to(root) or row['id'] in rows or len(rows) >= 50_000
                    or row['id'] != row['decoded_pixel_sha256'] or row['crop_path'] != expected_path):
                raise ValueError('Corpus path/identity/count bounds violated')
            if file_hash(path) != row['crop_sha256']:
                raise ValueError('Exported crop file was modified')
            with Image.open(path) as image:
                if pixel_hash(image) != row['decoded_pixel_sha256'] or image.size != (row['width'], row['height']):
                    raise ValueError('Exported native pixels were modified')
            if row['training_eligible'] != (row['split'] == 'train') or row['independent_evaluation_eligible']:
                raise ValueError('Unreviewed evaluation/training gate violated')
            if not row['provenance'] or any(
                    p['source'] not in SOURCES or p['license'] != SOURCES[p['source']]['license']
                    or p.get('source_url') != SOURCES[p['source']]['url']
                    or p.get('attribution') != SOURCES[p['source']]['attribution']
                    for p in row['provenance']):
                raise ValueError('Missing/unapproved source attribution')
            label = row['transcription']
            if label in groups and groups[label] != row['split']:
                raise ValueError('Cross-split registration leakage')
            groups[label] = row['split']
            rows[row['id']] = row
            if len(rows) % 2000 == 0:
                print(json.dumps({'integrity_checked_crops': len(rows)}), flush=True)
    if len(rows) != summary['unique_crops']:
        raise ValueError('Summary crop count mismatch')
    for name, origin in [('train_photographs.jsonl', 'photograph_publisher_declared'),
                         ('train_synthetic.jsonl', 'synthetic'),
                         ('train_origin_unverified.jsonl', 'mixed_origin_unverified')]:
        manifest_rows = [json.loads(line) for line in (root / name).read_text().splitlines()]
        expected = {r['id'] for r in rows.values() if r['training_eligible'] and r['origin'] == origin}
        actual = {r['id'] for r in manifest_rows}
        if len(actual) != len(manifest_rows) or actual != expected or any(rows[r['id']] != r for r in manifest_rows):
            raise ValueError('Training manifest inconsistent with corpus')
    result = {'verified_native_crops': len(rows), 'registration_groups': len(groups),
              'crop_and_pixel_hashes_verified': True, 'registration_split_check_passed': True,
              'source_and_training_manifest_checks_passed': True,
              'label_truth_verified_by_this_check': False,
              'corpus_sha256': summary['corpus_sha256']}
    (root / 'integrity_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def review_gallery(corpus, output):
    """Small source-balanced contact sheets; never automatic truth approval."""
    from PIL import Image, ImageDraw, ImageFont
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)
    selected = []
    buckets = defaultdict(list)
    for row in corpus.rows.values():
        buckets[row['origin']].append(row)
    for origin, rows in sorted(buckets.items()):
        selected.extend(sorted(rows, key=lambda r: r['id'])[:36])
    pages = []
    for offset in range(0, len(selected), 12):
        sheet = Image.new('RGB', (1100, 1080), '#e8edf3')
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(selected[offset:offset + 12]):
            x, y = (index % 2) * 550, (index // 2) * 180
            draw.text((x + 8, y + 5), f"{row['transcription']} | {row['width']}x{row['height']} | {row['id'][:8]}", font=font, fill='black')
            draw.text((x + 8, y + 25), row['origin'] + ' / ' + row['review'], font=font, fill='black')
            with Image.open(corpus.root / row['crop_path']) as image:
                image.thumbnail((530, 120))
                sheet.paste(image, (x + 8, y + 50))
        name = f'corpus-review-{offset // 12 + 1}.png'
        sheet.save(output / name)
        pages.append(name)
    text = '<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; style-src \'unsafe-inline\'">'
    text += '<title>Indian OCR corpus review</title><style>body{background:#152033;color:#eef;font:18px sans-serif;margin:24px}img{max-width:100%}pre{white-space:pre-wrap}</style>'
    text += '<h1>Indian OCR corpus: dataset preparation, not recognition results</h1><p>Publisher labels below are proposals, not OCR predictions. Generated plates are not genuine vehicles. Samples shown do not certify the whole dataset.</p>'
    text += '<pre>' + html.escape(json.dumps(json.loads((corpus.root / 'summary.json').read_text()), indent=2)) + '</pre>'
    text += ''.join(f'<img src="{html.escape(name)}" alt="Source-balanced crop review sheet">' for name in pages)
    (output / 'indian-corpus.html').write_text(text, encoding='utf-8')


def main():
    from PIL import Image
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sources', nargs='+', choices=list(SOURCES))
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--regression', type=Path)
    parser.add_argument('--viewer', type=Path)
    parser.add_argument('--detect', action='store_true')
    parser.add_argument('--extend', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--recover-exif', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if 'data' not in root.parts or 'benchmarks' not in root.parts:
        raise ValueError('Corpus must remain in ignored benchmark data')
    if args.verify_only:
        print(json.dumps(verify_corpus(root)), flush=True)
        return
    if not args.sources or not args.pilot:
        parser.error('Merge requires sources and the reviewed pilot manifest')
    pilot = json.loads(args.pilot.read_text())
    if pilot.get('dataset', {}).get('license') != 'CC-BY-4.0' or any(
            r.get('review') != 'visually_verified' or r.get('split') not in {'train', 'validation', 'test'}
            for r in pilot.get('samples', [])):
        raise ValueError('Legacy review status/licence/split invalid')
    prior = {r['transcription']: r['split'] for r in pilot['samples']}
    reserved = set()
    if args.regression:
        regression = json.loads(args.regression.read_text())
        reserved = {p['ground_truth'] for p in regression['passages']}
    corpus = Corpus(root, legacy=prior, reserved=reserved)
    recover_members = None
    if args.extend:
        existing_sources = corpus.load_existing()
        if args.recover_exif:
            if args.sources != ['bapatla'] or not args.detect:
                raise ValueError('Orientation recovery requires only Bapatla and its detector')
            recoverable = [r for r in corpus.rejected if r['reason'] == 'exif_orientation_needs_review'
                           and r['provenance']['source'] == 'bapatla']
            recover_members = {r['provenance']['member'] for r in recoverable}
            if not recover_members:
                raise ValueError('No pending orientation records')
            corpus.rejected = [r for r in corpus.rejected if r not in recoverable]
            corpus.counts['rejected_exif_orientation_needs_review'] -= len(recoverable)
        elif existing_sources & set(args.sources):
            raise ValueError('Source already merged; rebuild explicitly rather than double-counting observations')
    elif args.recover_exif:
        raise ValueError('Orientation recovery extends an existing merge')
    reviewed = {r['sha256']: r for r in pilot['samples']}
    quarantine_names = {Path(n).stem for n in pilot.get('quarantined', [])} | {'AN4', 'AP12'}
    detector = None
    if 'bapatla' in args.sources and args.detect:
        from .ocr_selection_benchmark import PlateDetectorExperiment
        detector = PlateDetectorExperiment()
    for source in args.sources:
        count = 0
        for data, label, provenance in archive_rows(root, source, selected_members=recover_members):
            if data is None:
                corpus.reject(provenance, provenance['rejection'])
                continue
            try:
                with Image.open(io.BytesIO(data)) as original:
                    if original.width * original.height > 20_000_000:
                        raise ValueError('Image exceeds pixel bounds')
                    if original.getexif().get(274, 1) not in range(1, 9):
                        corpus.reject(provenance, 'exif_orientation_needs_review')
                        continue
                    oriented, orientation = orient_native(original)
                    image = oriented.convert('RGB')
                    provenance['exif_orientation'] = orientation
                    provenance['native_transform'] = 'lossless_EXIF_pixel_permutation' if orientation != 1 else 'none'
                    provenance['box_coordinate_space'] = 'EXIF_normalized_native_pixels'
                verified = reviewed.get(provenance['original_sha256']) if source == 'bapatla' else None
                legacy_label_conflict = bool(verified and verified['transcription'] != label)
                if legacy_label_conflict:
                    verified = None
                if verified and verified['transcription'] == label:
                    path = (args.pilot.parent / verified['crop_path']).resolve()
                    if not path.is_relative_to(args.pilot.parent.resolve()) or file_hash(path) != verified['crop_sha256']:
                        raise ValueError('Legacy reviewed crop path/hash mismatch')
                    with Image.open(path) as crop:
                        image = crop.convert('RGB')
                    provenance['box'] = verified['box']
                elif source == 'bapatla':
                    if not detector:
                        corpus.reject(provenance, 'scene_requires_plate_localisation')
                        continue
                    import numpy as np
                    box = detector.box(np.asarray(image)[:, :, ::-1].copy())
                    if not box:
                        corpus.reject(provenance, 'no_plate_detection')
                        continue
                    box = tuple(int(v) for v in box)
                    image = image.crop(box)  # native integer pixels, no enlargement/restoration.
                    provenance['box'] = list(box)
                review = ('visually_verified' if verified else 'synthetic_generated' if source == 'abtexp'
                          else 'publisher_unverified')
                corpus.add(image, label, provenance, review=review,
                           quarantined=legacy_label_conflict or
                           (source == 'bapatla' and Path(provenance['member']).stem in quarantine_names))
            except (OSError, Image.DecompressionBombError) as error:
                corpus.reject(provenance, 'image_decode_error:' + type(error).__name__)
            count += 1
            if count % 250 == 0:
                print(json.dumps({'source': source, 'decoded': count, 'unique_so_far': len(corpus.rows)}), flush=True)
        print(json.dumps({'source_complete': source, 'decoded': count}), flush=True)
    summary = corpus.finish()
    if args.viewer:
        viewer = args.viewer.resolve()
        if not viewer.is_relative_to(root.parent):
            raise ValueError('Review gallery must remain under benchmark data')
        review_gallery(corpus, viewer)
    print(json.dumps({k: summary[k] for k in ('unique_crops', 'origin_counts', 'review_counts', 'split_counts', 'training_by_origin', 'observations')}), flush=True)


if __name__ == '__main__':
    main()
