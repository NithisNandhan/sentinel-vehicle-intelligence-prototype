"""Local saved-output review for the text/contrast plate-search experiment.

The inherited comparison owns label scoring and source bindings. Additional
search candidates remain unverified; this viewer never changes those scores.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image

from render_crop_comparison import ROOT, active, html as base_html, load, render as base_render, safe_file, sha, top, unique


NEW_SOURCES = frozenset(('character_group', 'contrast_plate'))
DEVICE_FIELDS = frozenset(('ocr_device', 'plate_device', 'vehicle_device', 'detector_device'))
WARNING = ('Character-group and contrast proposals are search candidates, not verified plates. '
           'Extra guesses do not reduce the selected-reference coverage score. '
           'Unlabelled candidates have unknown correctness.')


def compare_models(previous, current):
    """Permit only explicit runtime-device differences, never weight changes."""
    def identity(models):
        return {k: v for k, v in models.items() if k not in DEVICE_FIELDS} if isinstance(models, dict) else models
    if identity(previous) != identity(current):
        raise ValueError('Frozen detector/OCR model identity changed')
    devices = {side: {k: models[k] for k in sorted(DEVICE_FIELDS) if k in models}
               for side, models in [('previous', previous or {}), ('current', current or {})]}
    changed = devices['previous'] != devices['current']
    return dict(**devices, devices_changed=changed, model_identity_equal=True,
                note=('Runtime devices differ. CPU/GPU numerical differences and wall-clock search budgets can affect outputs; '
                      'processing times are not a controlled device-performance comparison.' if changed else
                      'Recorded runtime devices match. This is not a separate speed benchmark.'))


def sources_for_sample(group, sample):
    """Attribute the displayed native original, never every source in its track."""
    values = set(sample.get('proposal_sources', []))
    if not values:
        digest = sample.get('source_id')
        for observation in group.get('observed', []):
            if digest and observation.get('source_id') == digest:
                values.update(observation.get('proposal_sources', []))
    return sorted(values)


def selected_sample(group, evidence=None):
    samples = group.get('samples', [])
    if evidence and evidence.get('native_source_id'):
        matching = [s for s in samples if s.get('source_id') == evidence['native_source_id']
                    and s['position_ms'] == evidence['sample_position_ms']]
        if len(matching) == 1:
            return matching[0]
    return max(samples, key=lambda s: (s.get('selection_priority', 0), s.get('quality', 0), -s['position_ms']), default=None)


def candidate_sample(group):
    additional = [s for s in group.get('samples', []) if NEW_SOURCES.intersection(sources_for_sample(group, s))]
    return max(additional, key=lambda s: (s.get('selection_priority', 0), s.get('quality', 0), -s['position_ms']), default=None) or selected_sample(group)


def verified_native(capture, sample):
    path = safe_file(capture, sample['path'])
    if not path.is_relative_to((Path(capture)/'assets').resolve()):
        raise ValueError('Automatic crop must be in capture/assets')
    with Image.open(path) as image:
        if image.format != 'PNG':
            raise ValueError('Automatic crop must be a saved PNG')
        digest = hashlib.sha256(image.convert('RGB').tobytes('raw', 'BGR')).hexdigest()
    if sample.get('source_id') and sample['source_id'] != digest:
        raise ValueError('Automatic crop native source hash binding mismatch')
    return path


def contrast_preview(source, destination):
    """Same enhancement function, applied to a crop for illustration only."""
    import numpy as np
    sys.path.insert(0, str(ROOT/'edge'))
    from sentinel_edge.low_contrast_search import contrast_view
    with Image.open(source) as image:
        bgr = np.ascontiguousarray(np.array(image.convert('RGB'))[:, :, ::-1])
    enhanced = contrast_view(bgr)
    Image.fromarray(enhanced[:, :, ::-1]).save(destination)


def data_url(root, relative):
    return 'data:image/png;base64,' + base64.b64encode(safe_file(root, relative).read_bytes()).decode('ascii') if relative else None


def extended_html(data, output):
    page = base_html(data, output)
    payload = dict(experiment=data['experiment'], references=[], candidates=[])
    for row in data['results']:
        payload['references'].append(dict(number=row['number'], sources=row['current']['proposal_sources'],
            contrast=data_url(output, row['current'].get('contrast_crop'))))
    for candidate in data['new_candidates']:
        payload['candidates'].append({**candidate, 'crop': data_url(output, candidate['crop'])})
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    styles = '''<style>
.lc-section{margin:22px 0;padding:20px;background:white;border:1px solid #cedbd4;border-radius:8px}.lc-warning{border-left:4px solid #a36a19;padding-left:14px;color:#624c23}.lc-tags{color:#0f766e;font-weight:600}.lc-toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:14px 0}.lc-toolbar button,.lc-toolbar select{font:inherit;margin:0;padding:7px 12px;background:white;color:#193b36;border:1px solid #809c94;border-radius:5px}.lc-toolbar button:focus-visible{outline:3px solid #0f766e;outline-offset:3px}.lc-toolbar button:disabled{opacity:.4}.lc-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.lc-card{border:1px solid #cedbd4;border-radius:6px;padding:14px}.lc-card h3{margin:0 0 8px;font-size:17px}.lc-card p{font-size:14px}.lc-card .reading{font-size:20px}pre.lc-stats{font-size:13px;background:#eff2ec;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:620px){.lc-grid{grid-template-columns:1fr}}
</style>'''
    controls = '''<section class="lc-section"><h2>Search experiment</h2><p class="lc-warning" id="lc-warning"></p><p id="lc-config"></p><p id="lc-devices"></p><p class="muted" id="lc-runtime-note"></p><p class="muted">Proposal sources describe the displayed original. They do not prove that this method was necessary or that the reading is correct.</p><details><summary>Detector diagnostics</summary><pre class="lc-stats" id="lc-stats"></pre></details></section>'''
    detail = '''<section class="lc-section" aria-live="polite"><h2>Search support for this reference</h2><p class="lc-tags" id="lc-sources"></p><figure><figcaption>Contrast illustration on the new crop — search only, not the OCR input</figcaption><div class="image" id="lc-contrast"></div></figure><p class="note">The detector enhances its larger search region. Applying the same CLAHE function to this crop can produce a different result because the surrounding pixels differ. The new automatic crop above preserves native source pixels.</p></section>
<section class="lc-section"><h2>Additional-source candidate review</h2><p class="muted">These histories contain character-group or contrast-plate proposals. They can include plates also found by the original detector and false detections.</p><div class="lc-toolbar"><label for="lc-filter">Source</label><select id="lc-filter"><option value="all">All additional sources</option><option value="character_group">Character group</option><option value="contrast_plate">Contrast plate detector</option></select><button id="lc-prev" type="button">Previous</button><span id="lc-page" aria-live="polite"></span><button id="lc-next" type="button">Next</button></div><div class="lc-grid" id="lc-candidates"></div></section>'''
    script = '''<script type="application/json" id="lc-data">''' + encoded + '''</script><script>
(()=>{const d=JSON.parse(document.getElementById('lc-data').textContent),e=id=>document.getElementById(id);let page=0;const size=8;
const img=(box,src,alt)=>{box.replaceChildren();if(!src){box.textContent='No corresponding automatic crop identified.';return;}const im=document.createElement('img');im.alt=alt;im.onload=()=>{im.style.maxWidth=Math.min(440,im.naturalWidth*2)+'px';im.style.maxHeight=Math.min(160,im.naturalHeight*2)+'px';};im.src=src;box.append(im);};
e('lc-warning').textContent=d.experiment.warning;e('lc-config').textContent='Character-guided search: '+(d.experiment.character_guided?'enabled':'disabled')+' · Contrast search: '+(d.experiment.contrast_pass?'enabled':'disabled');e('lc-stats').textContent=JSON.stringify(d.experiment.diagnostics,null,2);const runtime=d.experiment.model_runtime,describe=x=>Object.entries(x).map(([k,v])=>k.replace('_device','')+': '+String(v)).join(', ')||'not recorded';e('lc-devices').textContent='Previous devices — '+describe(runtime.previous)+' · New devices — '+describe(runtime.current);e('lc-runtime-note').textContent=runtime.note;
function reference(){const r=d.references.find(r=>r.number===Number(e('case').value));if(!r)return;e('lc-sources').textContent=r.sources.length?'Displayed crop sources: '+r.sources.join(', '):'Displayed crop source unavailable';img(e('lc-contrast'),r.contrast,'Illustrative contrast-enhanced crop');}
function candidates(){const f=e('lc-filter').value,rows=d.candidates.filter(c=>f==='all'||c.history_sources.includes(f)),pages=Math.max(1,Math.ceil(rows.length/size));page=Math.min(page,pages-1);e('lc-page').textContent=rows.length+' histories · page '+(page+1)+' of '+pages;e('lc-prev').disabled=page===0;e('lc-next').disabled=page>=pages-1;const root=e('lc-candidates');root.replaceChildren();if(!rows.length){root.textContent='No retained candidate histories for this filter.';return;}for(const c of rows.slice(page*size,(page+1)*size)){const card=document.createElement('article');card.className='lc-card';const h=document.createElement('h3');h.textContent=c.group_id;card.append(h);const pic=document.createElement('div');pic.className='image';img(pic,c.crop,'Native candidate crop');card.append(pic);for(const [text,cls] of [[c.reading||'No reconstructed reading','reading'],['History sources: '+c.history_sources.join(', '),'lc-tags'],['Displayed original sources: '+(c.sample_sources.join(', ')||'not recorded'),''],['Triage: '+c.status+(c.excluded?' · excluded from fusion':''),''],[c.correctness,'']]){const p=document.createElement('p');p.textContent=text;p.className=cls;card.append(p);}root.append(card);}}
e('case').addEventListener('change',reference);e('lc-filter').addEventListener('change',()=>{page=0;candidates();});e('lc-prev').addEventListener('click',()=>{page--;candidates();});e('lc-next').addEventListener('click',()=>{page++;candidates();});reference();candidates();})();
</script>'''
    page = page.replace('<title>Crop comparison</title>', '<title>Plate search comparison</title>', 1)
    page = page.replace('<h1>Crop comparison</h1>', '<h1>Plate search comparison</h1>', 1)
    page = page.replace('</style><main>', '</style>'+styles+'<main>', 1)
    page = page.replace('<label for="case">', controls+'<label for="case">', 1)
    return page.replace('</main></html>', detail+script+'</main></html>', 1)


def render(baseline, current, review, output):
    baseline, current, output = map(lambda p: Path(p).resolve(), (baseline, current, output))
    old, new = load(baseline/'predictions.json'), load(current/'predictions.json')
    model_runtime = compare_models(old.get('models'), new.get('models'))
    flags = new.get('low_contrast_search')
    if not isinstance(flags, dict) or not any(flags.get(k) is True for k in ('character_guided', 'contrast_pass')):
        raise ValueError('An explicitly enabled text/contrast experiment is required')
    groups = unique(new['groups'], 'id', 'current group')
    pending = []
    # Preflight additional candidates before the inherited renderer writes output.
    for group in new['groups']:
        history_sources = sorted(set().union(*(set(o.get('proposal_sources', [])) for o in group.get('observed', [])),
                                            *(set(s.get('proposal_sources', [])) for s in group.get('samples', []))))
        if not NEW_SOURCES.intersection(history_sources):
            continue
        sample = candidate_sample(group)
        if sample is not None:
            pending.append((group, sample, history_sources, verified_native(current, sample)))
    data = base_render(baseline, current, review, output)
    for row in data['results']:
        evidence = row['current']
        group = groups.get(evidence['group_id'])
        sample = selected_sample(group, evidence) if group else None
        evidence['proposal_sources'] = sources_for_sample(group, sample) if sample else []
        evidence['contrast_crop'] = None
        if evidence['crop']:
            relative = f"assets/contrast-illustration-{row['number']:02}.png"
            contrast_preview(safe_file(output, evidence['crop']), output/relative)
            evidence['contrast_crop'], evidence['contrast_sha256'] = relative, sha(output/relative)
    attributed = {row['current']['group_id']: row['number'] for row in data['results'] if row['current']['group_id']}
    candidates = []
    for index, (group, sample, history_sources, source) in enumerate(pending):
        relative = f'assets/candidate-{index:04}.png'
        shutil.copyfile(source, output/relative)
        reference = attributed.get(group['id'])
        candidates.append(dict(group_id=group['id'], crop=relative, crop_sha256=sha(output/relative),
            source_id=sample.get('source_id'), position_ms=sample['position_ms'], reading=top(group),
            history_sources=history_sources, sample_sources=sources_for_sample(group, sample),
            status=group.get('plate_validation', {}).get('status', 'unverified'), excluded=not active(group),
            reference_number=reference,
            correctness=('Selected-reference association '+str(reference)+': inspect the numbered comparison; identity correspondence may be illustrative.'
                         if reference else 'Unlabelled candidate: correctness unknown.')))
    data['new_candidates'] = candidates
    counts = Counter(source for c in candidates for source in c['history_sources'] if source in NEW_SOURCES)
    data['experiment'] = dict(character_guided=flags.get('character_guided', False), contrast_pass=flags.get('contrast_pass', False),
        diagnostics=new.get('search_diagnostics', {}).get('low_contrast', {}), warning=WARNING,
        candidate_histories=len(candidates), candidate_history_source_counts=dict(counts),
        preview_scope='CLAHE on selected crop for illustration; not the exact full-ROI search input or OCR input.',
        model_weights_changed=False, scoring_changed=False, model_runtime=model_runtime)
    (output/'comparison.json').write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    (output/'index.html').write_text(extended_html(data, output), encoding='utf-8')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'current', 'review', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = render(args.baseline, args.current, args.review, args.output)
    print(json.dumps(dict(stage='plate_search_comparison_rendered', scores=result['scores'],
                          candidate_histories=result['experiment']['candidate_histories'])))
