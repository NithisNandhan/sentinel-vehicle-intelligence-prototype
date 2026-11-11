"""Render saved crop/OCR comparisons locally, without inference or input edits.

Output must be a fresh, Git-ignored folder under data/benchmarks. Human answers
are used only to score saved reconstructed top readings. No model is imported.
"""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RULE = ('Full human label equals any active reconstruction.top anywhere in the '
        'clip. Each reference counts once; timestamps, confidence, other methods '
        'and lower candidates do not affect this score. Extra guesses are not penalised.')
PROVENANCE = ('User-supplied permitted Pexels Mumbai video, selected for local '
              'crop/OCR review; not genuine Sentinel footage.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def unique(items, field, what):
    result = {item[field]: item for item in items}
    if len(result) != len(items):
        raise ValueError('Duplicate ' + what)
    return result


def safe_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe asset path')
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError('Missing asset or path escapes its source folder')
    return target


def top(group):
    return group.get('methods', {}).get('reconstruction', {}).get('top')


def active(group):
    return not group.get('plate_validation', {}).get('excluded_from_fusion', False)


def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0


def geometry_group(anchor, groups):
    """Illustrative only: unique best IoU >= .5 at an identical source time."""
    if not anchor:
        return None
    observations = {}
    for observation in anchor.get('observed', []):
        observations.setdefault(observation['position_ms'], []).append(observation['box'])
    scores = []
    for group in groups:
        if not active(group):
            continue
        score = max((overlap(box, observation['box'])
                     for observation in group.get('observed', [])
                     for box in observations.get(observation['position_ms'], [])), default=0)
        if score >= .5:
            scores.append((score, group))
    scores.sort(key=lambda item: item[0], reverse=True)
    if not scores or (len(scores) > 1 and scores[0][0] - scores[1][0] < .05):
        return None
    return scores[0][1]


def choose(document, case, anchor=None):
    groups = document['groups']
    matches = [group for group in groups if active(group) and top(group) == case['expected']]
    group = matches[0] if matches else next((g for g in groups if g['id'] == case['group']), None)
    relation = 'Exact reconstructed reading found anywhere' if matches else 'Saved case association; illustrative only'
    if group is None:
        group = geometry_group(anchor, groups)
        relation = 'Geometry correspondence at the same source timestamp; illustrative, not proof'
    if group is None:
        relation = 'No corresponding automatic crop identified'
    sample = max(group.get('samples', []), key=lambda s: (s.get('selection_priority', 0), s.get('quality', 0), -s['position_ms']), default=None) if group else None
    return dict(found=bool(matches), reading=top(group) if group else None,
                group_id=group['id'] if group else None, correspondence=relation,
                excluded=not active(group) if group else False,
                crop=sample['path'] if sample else None,
                native_source_id=sample.get('source_id') if sample else None,
                sample_position_ms=sample['position_ms'] if sample else None)


def build_data(baseline, current, baseline_accuracy, current_accuracy, manifest):
    """Pure comparison; capture validation is performed before calling this."""
    old_cases = unique(baseline_accuracy['cases'], 'id', 'baseline reference')
    new_cases = unique(current_accuracy['cases'], 'id', 'current reference')
    if set(old_cases) != set(new_cases):
        raise ValueError('Captures must score the same reference set')
    references = unique(manifest['samples'], 'id', 'manifest reference')
    old_groups = unique(baseline['groups'], 'id', 'baseline group')
    unique(current['groups'], 'id', 'current group')
    results = []
    for number, case in enumerate(baseline_accuracy['cases'], 1):
        new_case = new_cases[case['id']]
        if new_case['expected'] != case['expected']:
            raise ValueError('Human answers differ between captures')
        previous = choose(baseline, case)
        # The baseline association is preferred; group IDs are never reused
        # across captures. Geometry cannot change the found-anywhere boolean.
        anchor = old_groups.get(case['group']) or old_groups.get(previous['group_id'])
        new = choose(current, new_case, anchor)
        results.append(dict(number=number, reference_id=case['id'], actual=case['expected'],
                            reference_crop=references[case['id']]['crop'], previous=previous, current=new))
    total = len(results)
    scores = {}
    for side in ('previous', 'current'):
        correct = sum(row[side]['found'] for row in results)
        scores[side] = dict(correct=correct, total=total, percent=round(100*correct/total, 1) if total else 0)
    return dict(results=results, scores=scores, rule=RULE, provenance=PROVENANCE,
                source_sha256=manifest['source_sha256'])


def validate_capture(capture, review, manifest, labels):
    prediction = safe_file(capture, 'predictions.json')
    document, accuracy = load(prediction), load(safe_file(capture, 'accuracy.json'))
    if accuracy['predictions_sha256'] != sha(prediction) or accuracy['labels_sha256'] != sha(review/'labels.json'):
        raise ValueError('Prediction or human-label hash binding mismatch')
    if document['source_sha256'] != manifest['source_sha256'] or labels['source_sha256'] != manifest['source_sha256']:
        raise ValueError('Source hash binding mismatch')
    if labels['review_manifest_sha256'] != sha(review/'manifest.json'):
        raise ValueError('Human labels are bound to a different review manifest')
    human = unique(labels['labels'], 'id', 'human label')
    references = unique(manifest['samples'], 'id', 'manifest reference')
    for case in accuracy['cases']:
        label = human[case['id']]
        if label['status'] != 'readable' or label['plate'] != case['expected'] or not case['expected']:
            raise ValueError('Saved score does not match the actual readable human label')
        sample = references[case['id']]
        if sha(safe_file(review, sample['crop'])) != sample['crop_sha256']:
            raise ValueError('Reference image hash binding mismatch')
    return document, accuracy


def html(data, output):
    embedded = copy.deepcopy(data)
    for row in embedded['results']:
        paths = [(row, 'reference_crop'), (row['previous'], 'crop'), (row['current'], 'crop')]
        for obj, key in paths:
            if obj[key]:
                path = safe_file(output, obj[key])
                obj[key] = 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
    payload = json.dumps(embedded, ensure_ascii=False, allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crop comparison</title><style>
:root{color-scheme:light;--teal:#0F766E;--ink:#193b36;--paper:#f7f7f1}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,sans-serif}main{max-width:1050px;margin:36px auto;padding:0 24px}h1{margin:0;font-size:29px}p{margin:10px 0}.muted,figcaption{color:#526660;font-size:14px}select{font:inherit;padding:8px 24px 8px 12px;margin:18px 0;border:1px solid #809c94;border-radius:5px;background:white;color:var(--ink)}select:focus-visible{outline:3px solid #0F766E;outline-offset:3px}.panel{padding:24px;border:1px solid #cedbd4;border-radius:8px;background:white}h2{margin:0 0 14px;font-size:22px}figure{margin:0;min-width:0}figcaption{margin:0 0 8px}.image{height:170px;background:#eff2ec;display:flex;align-items:center;justify-content:center;border-radius:5px;overflow:hidden}.image img{object-fit:contain;max-width:100%;max-height:160px;width:auto;height:auto}.reference{max-width:450px;margin:0 auto 24px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:22px}.reading{font:600 23px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}.status{color:var(--teal);font-weight:650}.actual{color:var(--teal);font-weight:650;font-size:21px}.empty{padding:18px;color:#526660;font-size:14px}.note{font-size:13px;color:#526660}.scores{font-size:20px;font-weight:650;color:var(--teal);margin:16px 0}footer{margin-top:18px;font-size:13px;color:#526660}@media(max-width:620px){main{padding:0 14px;margin:20px auto}.panel{padding:16px}.columns{grid-template-columns:1fr}.reading{font-size:20px}}
</style><main><h1>Crop comparison</h1><p class="muted" id="provenance"></p><div class="scores" id="scores"></div><p class="muted" id="rule"></p><label for="case">Choose reference </label><select id="case" aria-controls="panel"></select>
<section class="panel" id="panel" aria-live="polite"><h2 id="title"></h2><figure class="reference"><figcaption>Human reference image</figcaption><div class="image" id="reference"></div><p class="actual" id="actual"></p></figure><div class="columns"><div><figure><figcaption>Previous automatic crop</figcaption><div class="image" id="previous-image"></div></figure><p class="status" id="previous-status"></p><div class="reading" id="previous-reading"></div><p class="note" id="previous-note"></p></div><div><figure><figcaption>New automatic crop</figcaption><div class="image" id="current-image"></div></figure><p class="status" id="current-status"></p><div class="reading" id="current-reading"></div><p class="note" id="current-note"></p></div></div></section>
<footer>Readings are saved reconstructed top outputs. Each automatic image is one original from its selected group's multi-frame evidence, not a full reconstruction. Geometry correspondence is illustrative only. Found-anywhere scores describe these selected human references, not full-video accuracy. Images are capped at twice native size.</footer><noscript>Enable JavaScript to select a reference.</noscript>
<script type="application/json" id="data">''' + payload + '''</script><script>
(()=>{const d=JSON.parse(document.getElementById('data').textContent),e=id=>document.getElementById(id),sel=e('case');e('provenance').textContent=d.provenance;e('rule').textContent=d.rule;const s=x=>x.correct+' / '+x.total+' ('+x.percent.toFixed(1)+'%)';e('scores').textContent='Previous '+s(d.scores.previous)+' · New '+s(d.scores.current);for(const r of d.results){const o=document.createElement('option');o.value=r.number;o.textContent=r.number;sel.append(o);}function image(id,src,alt){const box=e(id);box.replaceChildren();if(!src){const p=document.createElement('span');p.className='empty';p.textContent='No corresponding automatic crop identified.';box.append(p);return;}const img=document.createElement('img');img.alt=alt;img.onload=()=>{img.style.maxWidth=Math.min(440,img.naturalWidth*2)+'px';img.style.maxHeight=Math.min(160,img.naturalHeight*2)+'px';};img.src=src;box.append(img);}function show(){const r=d.results.find(r=>r.number===Number(sel.value));if(!r)return;e('title').textContent='Reference '+r.number+' of '+d.results.length;e('actual').textContent='Actual human label: '+r.actual;image('reference',r.reference_crop,'Human reference image '+r.number);for(const side of ['previous','current']){const v=r[side];image(side+'-image',v.crop,(side==='previous'?'Previous':'New')+' automatic crop');e(side+'-status').textContent=v.found?'Found anywhere in clip':'Not found anywhere in clip';e(side+'-reading').textContent=v.reading||'No reconstructed reading available';e(side+'-note').textContent=v.correspondence+(v.excluded?'. Excluded from fusion; not scored.':'')+(v.sample_position_ms!==null?' · Example crop at '+v.sample_position_ms+' ms':'');}}sel.addEventListener('change',show);show();})();
</script></main></html>'''


def contact_sheets(data, output):
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=14)
    files = []
    for offset in range(0, len(data['results']), 6):
        rows = data['results'][offset:offset+6]
        canvas = Image.new('RGB', (1230, 105+230*len(rows)), '#f7f7f1')
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 14), 'Crop comparison | selected human references', fill='#0F766E', font=font)
        score = data['scores']
        draw.text((20, 42), f"Found anywhere: previous {score['previous']['correct']}/{score['previous']['total']} | new {score['current']['correct']}/{score['current']['total']}", fill='#193b36', font=font)
        draw.text((20, 70), 'Permitted Pexels review video; not genuine Sentinel footage. Geometry correspondence is not proof.', fill='#526660', font=small)
        for index, row in enumerate(rows):
            y = 105 + index*230
            columns = [('Human reference', row['reference_crop'], 'Actual: '+row['actual'], ''),
                       ('Previous automatic', row['previous']['crop'], row['previous']['reading'], row['previous']),
                       ('New automatic', row['current']['crop'], row['current']['reading'], row['current'])]
            for column, (label, path, reading, evidence) in enumerate(columns):
                x = 20 + column*405
                draw.text((x, y), f"#{row['number']} | {label}", fill='#0F766E', font=font)
                draw.rectangle((x, y+30, x+380, y+155), fill='#e9eee8')
                if path:
                    with Image.open(safe_file(output, path)) as original:
                        image = original.convert('RGB')
                        scale = min(2, 370/image.width, 115/image.height)
                        image = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))), Image.Resampling.LANCZOS)
                        canvas.paste(image, (x+(380-image.width)//2, y+30+(125-image.height)//2))
                else:
                    draw.text((x+9, y+78), 'No corresponding automatic crop', fill='#526660', font=small)
                draw.text((x, y+161), reading or 'No reconstructed reading available', fill='#193b36', font=font)
                if evidence:
                    status = 'Found anywhere' if evidence['found'] else 'Not found anywhere'
                    draw.text((x, y+187), status, fill='#0F766E', font=small)
                    if evidence['correspondence'].startswith('Geometry'):
                        draw.text((x, y+207), 'Geometry correspondence only; not proof', fill='#526660', font=small)
                    elif evidence['excluded']:
                        draw.text((x, y+207), 'Excluded from fusion; illustrative only', fill='#526660', font=small)
        filename = f'comparison-{offset//6+1:02}.png'
        canvas.save(output/filename)
        files.append(filename)
    return files


def render(baseline, current, review, output):
    baseline, current, review, output = (Path(p).resolve() for p in (baseline, current, review, output))
    if output.exists():
        raise ValueError('Output must be a fresh folder')
    if not output.is_relative_to(ROOT/'data'/'benchmarks'):
        raise ValueError('Output must stay under the ignored data/benchmarks folder')
    ignored = subprocess.run(['git', 'check-ignore', '-q', '--', str(output/'index.html')], cwd=ROOT, check=False)
    if ignored.returncode != 0:
        raise ValueError('Output must be Git-ignored')
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (baseline, current, review)):
        raise ValueError('Output must be separate from every input')
    manifest, labels = load(review/'manifest.json'), load(review/'labels.json')
    old, old_accuracy = validate_capture(baseline, review, manifest, labels)
    new, new_accuracy = validate_capture(current, review, manifest, labels)
    data = build_data(old, new, old_accuracy, new_accuracy, manifest)
    data['bindings'] = {name: dict(predictions_sha256=sha(folder/'predictions.json'), accuracy_sha256=sha(folder/'accuracy.json'))
                        for name, folder in [('previous', baseline), ('current', current)]}
    data['bindings'].update(labels_sha256=sha(review/'labels.json'), manifest_sha256=sha(review/'manifest.json'))
    # Native source_id hashes decoded uint8 BGR pixels, not the PNG encoding.
    # Verify it when present, and also bind the exact copied PNG bytes.
    copies = []
    for row in data['results']:
        reference = safe_file(review, row['reference_crop'])
        destination = f"assets/reference-{row['number']:02}.png"
        copies.append((reference, destination))
        row['reference_crop'], row['reference_sha256'] = destination, sha(reference)
        for side, capture in [('previous', baseline), ('current', current)]:
            evidence = row[side]
            if evidence['crop']:
                source = safe_file(capture, evidence['crop'])
                if not source.is_relative_to((capture/'assets').resolve()):
                    raise ValueError('Automatic crop must be in capture/assets')
                with Image.open(source) as image:
                    if image.format != 'PNG':
                        raise ValueError('Automatic crop must be a saved PNG')
                    native_hash = hashlib.sha256(image.convert('RGB').tobytes('raw', 'BGR')).hexdigest()
                evidence['native_source_id_verified'] = bool(evidence['native_source_id'])
                if evidence['native_source_id'] and evidence['native_source_id'] != native_hash:
                    raise ValueError('Automatic crop native source hash binding mismatch')
                destination = f"assets/{side}-{row['number']:02}.png"
                copies.append((source, destination))
                evidence['crop'], evidence['crop_sha256'] = destination, sha(source)
    data['asset_hash_scope'] = ('Reference hashes verified against manifest. Automatic native source_id, when present, '
                                'is verified as SHA256 of decoded BGR pixels; copied PNG bytes are separately hashed. '
                                'The native_source_id_verified field reports each automatic crop binding.')
    (output/'assets').mkdir(parents=True)
    for source, relative in copies:
        shutil.copyfile(source, output/relative)
    data['contact_sheets'] = contact_sheets(data, output)
    (output/'comparison.json').write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    (output/'index.html').write_text(html(data, output), encoding='utf-8')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'current', 'review', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = render(args.baseline, args.current, args.review, args.output)
    print(json.dumps(dict(stage='crop_comparison_rendered', scores=result['scores'], contact_sheets=result['contact_sheets'])))
