"""Create contact sheets, or record explicit visual-review decisions.

Review is never inferred from OCR agreement. Decisions must be provided by the
reviewer, with ambiguous samples omitted. Deterministic registration splits.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from PIL import Image, ImageDraw, ImageFont
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--verified', nargs='*')
    args = parser.parse_args()
    document = json.loads((args.root / 'crop_proposals.json').read_text())
    rows = document['samples']
    if args.verified is None:
        font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 19)
        for offset in range(0, len(rows), 12):
            sheet = Image.new('RGB', (1000, 1080), '#dddddd')
            draw = ImageDraw.Draw(sheet)
            for index, row in enumerate(rows[offset:offset + 12]):
                x, y = (index % 2) * 500, (index // 2) * 180
                label = f"{Path(row['path']).stem}: {row['transcription']} ({row['width']}x{row['height']})"
                draw.text((x + 8, y + 5), label, fill='black', font=font)
                with Image.open(args.root / row['crop_path']) as image:
                    image.thumbnail((470, 140))
                    # Enlarge display only; actual crop pixels never change.
                    if image.width < 300:
                        image = image.resize((min(470, image.width * 2), min(140, image.height * 2)))
                    sheet.paste(image, (x + 8, y + 33))
            sheet.save(args.root / f'review-{offset // 12 + 1}.png')
        print(json.dumps({'sheets': (len(rows) + 11) // 12}))
        return
    verified = set(args.verified)
    known = {Path(row['path']).stem for row in rows}
    if not verified <= known:
        raise ValueError('Unknown review sample')
    selected = [dict(row, review='visually_verified') for row in rows if Path(row['path']).stem in verified]
    # Deterministic hash order, grouped by registration, before augmentation.
    plates = sorted({row['transcription'] for row in selected},
                    key=lambda label: hashlib.sha256(('sentinel-20260927-' + label).encode()).hexdigest())
    test_count = max(2, round(len(plates) * .2))
    val_count = max(2, round(len(plates) * .2))
    split = {plate: 'test' if index < test_count else 'validation' if index < test_count + val_count else 'train'
             for index, plate in enumerate(plates)}
    for row in selected:
        row['split'] = split[row['transcription']]
    document.update({'samples': selected, 'quarantined': [row['path'] for row in rows if Path(row['path']).stem not in verified],
                     'split_policy': 'seeded SHA256 registration grouping before augmentation',
                     'pretraining_overlap_unknown': True, 'not_temporal_sequences': True})
    document['dataset']['permitted_use'] = 'noncommercial_benchmark'
    document['dataset']['modifications'] = 'Native detector crops; training-only mild augmentation; original images unchanged'
    (args.root / 'training_manifest.json').write_text(json.dumps(document, indent=2))
    print(json.dumps({'reviewed': len(selected), 'unique_plates': len(plates),
                      'splits': {s: sum(row['split'] == s for row in selected) for s in ('train', 'validation', 'test')}}))


if __name__ == '__main__':
    main()
