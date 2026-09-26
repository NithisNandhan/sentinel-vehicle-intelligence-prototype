"""Produce label-hidden native-crop review sheets; never certify labels automatically."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(root, output):
    from PIL import Image, ImageDraw, ImageFont
    rows = [json.loads(line) for line in (root / 'holdout_needs_review.jsonl').read_text().splitlines()]
    rows = sorted((r for r in rows if r['origin'] == 'photograph_publisher_declared'), key=lambda r: r['id'])
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 22)
    index = []
    for offset in range(0, len(rows), 12):
        sheet = Image.new('RGB', (1200, 1200), '#e8edf3')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows[offset:offset + 12]):
            number = offset + i + 1
            path = root / row['crop_path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['crop_sha256']:
                raise ValueError('Review crop checksum mismatch')
            with Image.open(path) as image:
                native = image.convert('RGB')
                # Display-only nearest-neighbour enlargement; stored native pixels unchanged.
                scale = min(560 / native.width, 150 / native.height)
                display = native.resize((max(1, round(native.width * scale)), max(1, round(native.height * scale))), Image.Resampling.NEAREST)
            x, y = i % 2 * 600, i // 2 * 200
            draw.text((x + 10, y + 8), f"{number:03d} | native {row['width']}x{row['height']}", font=font, fill='black')
            sheet.paste(display, (x + 10, y + 40))
            index.append({'number': number, **row})
        sheet.save(output / f'blind-holdout-{offset // 12 + 1}.png')
    (output / 'review_index.json').write_text(json.dumps(index, indent=2))
    print(json.dumps({'photographic_crops': len(rows), 'registration_strings': len({r['transcription'] for r in rows}), 'pages': (len(rows) + 11) // 12, 'labels_shown': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.output)
