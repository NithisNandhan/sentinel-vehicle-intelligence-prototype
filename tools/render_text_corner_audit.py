"""Local, label-after-inference review of text corners and pre-OCR verification.

No inference, scoring, training or event delivery. Human references establish
illustrative correspondence only; diagnostic single-frame readings are not a
new reconstruction benchmark. All generated content stays Git-ignored locally.
"""
import argparse
import ast
import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image

from render_crop_comparison import ROOT, choose, load, overlap, safe_file, sha, unique, validate_capture


STATES = frozenset(('likely_plate', 'uncertain', 'likely_non_plate'))
PROVISIONAL_POLICY = 'uncertain_registration_shape_v1'
NOTICE = ('Four corners are an estimated text enclosure, not verified physical plate corners. '
          'Classifier scores are not verified real-world probabilities. OCR shown here is diagnostic single-frame output, '
          'not the final reconstructed reading or a new accuracy score. Registration resemblance does not confirm '
          'vehicle identity and does not trigger an automatic watchlist alert.')


def _check_reading(verdict, reading):
    """Recompute versioned interpretation using the inference policy itself."""
    edge_path=str(ROOT/'edge')
    if edge_path not in sys.path:
        sys.path.insert(0,edge_path)
    from sentinel_edge.provisional_plate_reading import check_reading
    return check_reading(verdict,reading)


def _image(root, relative, file_hash, native_hash=None):
    path = safe_file(root, relative)
    if not path.is_relative_to((root/'assets').resolve()):
        raise ValueError('Capture images must stay under assets')
    if not file_hash or sha(path) != file_hash:
        raise ValueError('Capture PNG hash mismatch')
    with Image.open(path) as image:
        if image.format != 'PNG':
            raise ValueError('Capture image must be PNG')
        if native_hash and hashlib.sha256(image.convert('RGB').tobytes('raw', 'BGR')).hexdigest() != native_hash:
            raise ValueError('Capture native BGR hash mismatch')
        return path, list(image.size)


def _quad(points, size):
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise ValueError('Four finite native corners required')
    try:
        result = [[float(x), float(y)] for x, y in points]
    except (ValueError, TypeError):
        raise ValueError('Four finite native corners required') from None
    # Perspective inversion can put a true boundary at -2e-16. Permit only
    # numerical roundoff, not a genuinely out-of-frame enclosure.
    epsilon=1e-6
    if any(not math.isfinite(x+y) or not -epsilon <= x < size[0]+epsilon or not -epsilon <= y < size[1]+epsilon for x, y in result):
        raise ValueError('Corners outside native context')
    return [[min(size[0]-1,max(0,x)),min(size[1]-1,max(0,y))] for x,y in result]


def validate_audit(capture):
    capture = Path(capture).resolve()
    document = load(safe_file(capture, 'audit.json'))
    policy = document.get('provisional_ocr_policy')
    if policy not in (None, PROVISIONAL_POLICY):
        raise ValueError('Unsupported provisional OCR policy')
    if document.get('labels_used_during_inference') is not False:
        raise ValueError('Explicit label-free inference declaration required')
    if not document.get('source_sha256') or not document.get('schedule_sha256'):
        raise ValueError('Source and schedule bindings required')
    if (capture/'schedule.json').is_file() and sha(capture/'schedule.json') != document['schedule_sha256']:
        raise ValueError('Schedule hash mismatch')
    frames = document.get('frames', [])
    if len({f['position_ms'] for f in frames}) != len(frames):
        raise ValueError('Duplicate frame timestamp')
    assets = {}
    for frame in frames:
        if not isinstance(frame['position_ms'], (int, float)) or not math.isfinite(frame['position_ms']) or frame['position_ms'] < 0:
            raise ValueError('Invalid source timestamp')
        region_ids = set()
        for region in frame['regions']:
            if region['id'] in region_ids:
                raise ValueError('Duplicate vehicle region')
            region_ids.add(region['id'])
            path, size = _image(capture, region['path'], region['sha256'], region.get('native_source_id'))
            assets[region['path']] = path
            x1, y1, x2, y2 = region['box']
            if (x1 < 0 or y1 < 0 or x2-x1 != size[0] or y2-y1 != size[1]):
                raise ValueError('Native context shape mismatch')
            for polygon in region['polygons']:
                _quad(polygon, size)
            proposals = region['proposals']
            if len({p['id'] for p in proposals}) != len(proposals):
                raise ValueError('Duplicate proposal ID')
            for proposal in proposals:
                _quad(proposal['quad'], size)
                _quad(proposal['seed_quad'], size)
                if proposal.get('geometry', {}).get('boundary') != 'estimated_text_enclosure':
                    raise ValueError('Estimated corner declaration required')
                verdict = proposal['verdict']
                if verdict['status'] not in STATES or not math.isfinite(verdict['classifier_score']):
                    raise ValueError('Invalid classifier verdict')
                if not isinstance(proposal['ocr_ran'], bool):
                    raise ValueError('Explicit OCR execution state required')
                allowed_ocr = verdict['status'] == 'likely_plate' or (policy == PROVISIONAL_POLICY and verdict['status'] == 'uncertain')
                if not allowed_ocr and proposal['ocr_ran']:
                    raise ValueError('Uncertain or rejected candidate must not run OCR')
                if not proposal['ocr_ran'] and proposal.get('top') is not None:
                    raise ValueError('Skipped OCR cannot have a reading')
                if proposal.get('top') is not None and not isinstance(proposal['top'],str):
                    raise ValueError('Raw OCR must be text or null')
                if policy == PROVISIONAL_POLICY:
                    reading_check=proposal.get('reading_check')
                    if (not isinstance(reading_check,dict) or reading_check.get('operational_eligible') is not False
                            or reading_check != _check_reading(verdict,proposal.get('top'))):
                        raise ValueError('Provisional reading interpretation does not match raw OCR/policy')
                if policy is None and proposal.get('reading_check') is not None:
                    raise ValueError('Provisional reading interpretation requires the versioned policy')
                if any(not isinstance(i, int) or not 0 <= i < len(region['polygons']) for i in proposal['text_members']):
                    raise ValueError('Unknown text-outline member')
                native, native_size = _image(capture, proposal['native_path'], proposal['native_sha256'], proposal.get('native_source_id'))
                rectified, _ = _image(capture, proposal['rectified_path'], proposal['rectified_sha256'])
                box = proposal['box']
                if (len(box) != 4 or not 0 <= box[0] < box[2] <= size[0]
                        or not 0 <= box[1] < box[3] <= size[1]
                        or [box[2]-box[0], box[3]-box[1]] != native_size):
                    raise ValueError('Native candidate box mismatch')
                assets[proposal['native_path']], assets[proposal['rectified_path']] = native, rectified
    return document, assets


def _reference_box(sample, corners):
    points = corners.get(sample['id'])
    if not points:
        return sample['box']
    x, y = sample['box'][:2]
    return [x+min(p[0] for p in points), y+min(p[1] for p in points),
            x+max(p[0] for p in points), y+max(p[1] for p in points)]


def associate(reference, audit, corners):
    expected = _reference_box(reference, corners)
    options = []
    frame_indexes = [i for i, f in enumerate(audit['frames']) if abs(f['position_ms']-reference['position_ms']) <= 20]
    for frame_index in frame_indexes:
        for region_index, region in enumerate(audit['frames'][frame_index]['regions']):
            x, y = region['box'][:2]
            for proposal_index, proposal in enumerate(region['proposals']):
                a, b, c, d = proposal['box']
                score = overlap(expected, [a+x, b+y, c+x, d+y])
                if score >= .3:
                    options.append((score, frame_index, region_index, proposal_index))
    options.sort(reverse=True)
    # Do not use the OCR answer or classifier score to break an identity tie.
    match = options[0][1:] if options and (len(options) == 1 or options[0][0]-options[1][0] >= .05) else None
    overlapping_alternatives = False
    if options and match is None:
        tied = [item for item in options if options[0][0]-item[0] < .05]
        same_region = len({(item[1], item[2]) for item in tied}) == 1
        if same_region:
            frame_index, region_index = tied[0][1:3]
            proposals = audit['frames'][frame_index]['regions'][region_index]['proposals']
            boxes = [proposals[item[3]]['box'] for item in tied]
            overlapping_alternatives = all(overlap(a,b) >= .7 for i,a in enumerate(boxes) for b in boxes[i+1:])
            if overlapping_alternatives:
                match = options[0][1:]
    return dict(frame_indexes=frame_indexes, match=list(match) if match else None,
                inspection=list(options[0][1:]) if options and match is None else None,
                correspondence=('Overlapping crop alternatives; highest geometric overlap shown. Illustrative only; identity is not verified.'
                                if overlapping_alternatives else
                                'Unique geometry correspondence at the reference time; illustrative only.' if match else
                                'No unambiguous association; highest geometric overlap is shown for inspection only, not verified identity.' if options else
                                'No unambiguous automatic correspondence; inspect all vehicle regions.'))


def html(data, output):
    embedded = copy.deepcopy(data)
    def inline(relative):
        return ('data:image/png;base64,'+base64.b64encode(safe_file(output, relative).read_bytes()).decode('ascii')) if relative else None
    for case in embedded['references']:
        case['reference_crop'] = inline(case['reference_crop'])
        case['previous']['crop'] = inline(case['previous']['crop'])
    for frame in embedded['frames']:
        for region in frame['regions']:
            region['path'] = inline(region['path'])
            for proposal in region['proposals']:
                proposal['native_path'] = inline(proposal['native_path'])
                proposal['rectified_path'] = inline(proposal['rectified_path'])
    encoded = json.dumps(embedded, ensure_ascii=False, allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel · Text corners and plate verification</title><style>
:root{--paper:#f6f7f2;--ink:#183b35;--teal:#0f766e;--line:#cbd8d1;--muted:#526a62}*{box-sizing:border-box}body{margin:24px auto;padding:0 20px;max-width:1420px;font:16px/1.55 system-ui;background:var(--paper);color:var(--ink)}h1,h2,h3{line-height:1.2}h1{font-size:30px}h2{font-size:22px;color:var(--teal)}h3{font-size:16px}p{overflow-wrap:anywhere}.muted,.note{color:var(--muted);font-size:14px}.note{border-left:3px solid #b28229;padding-left:12px}.panel{background:white;border:1px solid var(--line);border-radius:8px;padding:20px;margin:18px 0}.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin:16px 0}select{font:inherit;padding:8px 12px;background:white;color:var(--ink);border:1px solid #7e9a91;border-radius:5px;max-width:100%}select:focus-visible{outline:3px solid var(--teal)}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px}.image{display:flex;align-items:center;justify-content:center;min-height:145px;background:#edf1e9;border-radius:5px;padding:12px}.image img{max-width:100%;max-height:200px;object-fit:contain}.reading{font:600 19px/1.4 ui-monospace,Consolas,monospace}.context{width:100%;max-height:650px;background:#edf1e9;display:block}.status{font-weight:600}.likely_plate{color:#0f766e}.uncertain{color:#9a6700}.likely_non_plate{color:#b42318}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}ul{padding-left:22px}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px}.raw{color:#2563eb}details{margin:14px 0}@media(max-width:900px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:520px){.cards{grid-template-columns:1fr}body{padding:0 12px}}
</style><h1>Text corners &amp; plate verification</h1><p class="note" id="notice"></p><p class="muted" id="scope"></p>
<div class="toolbar"><label for="reference">Reference</label><select id="reference"></select><span id="time" class="muted"></span></div>
<section class="panel"><h2 id="reference-title"></h2><p id="correspondence" class="muted"></p><div class="cards">
<article><h3>Human reference</h3><div class="image" id="reference-image"></div><p class="reading" id="actual"></p></article>
<article><h3>Previous automatic crop</h3><div class="image" id="previous-image"></div><p class="reading" id="previous-reading"></p><p class="muted">Saved reconstructed top output</p></article>
<article><h3>New expanded native crop</h3><div class="image" id="native-image"></div><p id="verdict" class="status"></p></article>
<article><h3>Rectified working copy</h3><div class="image" id="rectified-image"></div><p class="reading" id="reading"></p><p id="raw-reading" class="muted"></p><p id="ocr-note" class="muted"></p></article></div><p id="score-note" class="note"></p><ul id="reasons"></ul></section>
<section class="panel"><h2>All automatic vehicle regions</h2><div class="toolbar"><label for="frame">Audited frame</label><select id="frame"></select><label for="region">Vehicle region</label><select id="region"></select></div>
<div class="legend"><span class="raw">Blue dashed: raw text outline</span><span class="likely_plate">Teal: likely plate</span><span class="uncertain">Amber: uncertain</span><span class="likely_non_plate">Red: likely non-plate</span></div><div id="context"></div><p class="muted" id="unretained"></p>
<div class="toolbar"><label for="status-filter">Status</label><select id="status-filter"><option value="all">All candidates</option><option value="likely_plate">Likely plate</option><option value="uncertain">Uncertain</option><option value="likely_non_plate">Likely non-plate</option></select><label for="candidate">Candidate</label><select id="candidate"></select></div><p class="note">Geometry correspondence is illustrative. Selecting a different candidate does not change any accuracy score.</p></section>
<details class="panel"><summary>Capture and verifier metadata</summary><pre id="metadata"></pre></details>
<script type="application/json" id="audit-data">'''+encoded+'''</script><script>
(()=>{'use strict';const d=JSON.parse(document.getElementById('audit-data').textContent),e=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';
const colours={likely_plate:'#0f766e',uncertain:'#b7791f',likely_non_plate:'#b42318'};
const option=(select,value,label)=>{const o=document.createElement('option');o.value=String(value);o.textContent=label;select.append(o);};
const image=(id,src)=>{const box=e(id);box.replaceChildren();if(!src){box.textContent='No corresponding crop';return;}const im=document.createElement('img');im.src=src;im.alt=id.replaceAll('-',' ');box.append(im);};
const currentRegion=()=>{if(e('frame').value===''||e('region').value==='')return null;return d.frames[Number(e('frame').value)]?.regions[Number(e('region').value)];};
function showCandidate(){const region=currentRegion(),p=region?.proposals[Number(e('candidate').value)];image('native-image',p?.native_path);image('rectified-image',p?.rectified_path);e('reasons').replaceChildren();if(!p){e('verdict').textContent='No candidate selected';e('reading').textContent='';e('raw-reading').textContent='';e('ocr-note').textContent='OCR was not run for this selection.';e('score-note').textContent='';return;}const v=p.verdict;e('verdict').className='status '+v.status;e('verdict').textContent=v.status.replaceAll('_',' ');const checked=p.reading_check;const provisional=checked?.state==='provisional_plate'&&checked.display_plate;e('reading').textContent=p.ocr_ran?(provisional?'Provisional plate: '+checked.display_plate:(p.top||'No diagnostic OCR reading')):'OCR skipped';e('reading').className='reading '+(provisional?'likely_plate':checked?'uncertain':'');e('raw-reading').textContent=checked&&p.ocr_ran?'Raw OCR: '+(p.top||'No text')+' · '+checked.state.replaceAll('_',' '):'';e('ocr-note').textContent=p.ocr_ran?(checked?'Registration resemblance is not identity confirmation. This provisional/review output cannot trigger an automatic watchlist alert.':'Single-frame diagnostic reading; not reconstructed output.'):(p.ocr_skip_reason||(v.status==='likely_plate'?'Classified as likely plate; OCR was not run in this capture.':(d.provisional_ocr_policy?'OCR was not run in this capture; visually rejected regions stay excluded.':'Only likely-plate candidates reach OCR. Uncertain/rejected evidence remains visible.')));e('score-note').textContent='Classifier score: '+Number(v.classifier_score).toFixed(4)+' · This is not a verified real-world probability. Corners are an estimated text enclosure.';for(const reason of [...(p.reasons||[]),...(v.reasons||[]),...(checked?.reasons||[])]){const li=document.createElement('li');li.textContent=reason;e('reasons').append(li);}}
function candidates(){const region=currentRegion();e('candidate').replaceChildren();for(const [i,p] of (region?.proposals||[]).entries()){if(e('status-filter').value==='all'||p.verdict.status===e('status-filter').value)option(e('candidate'),i,p.id+' · '+p.verdict.status.replaceAll('_',' ')+(p.reading_check?.display_plate?' · provisional':''));}if(!e('candidate').options.length)option(e('candidate'),-1,'No candidates in this status');showCandidate();}
function draw(){const region=currentRegion(),host=e('context');host.replaceChildren();if(!region){host.textContent='No vehicle region captured at this frame.';e('unretained').textContent='';candidates();return;}const svg=document.createElementNS(ns,'svg');svg.classList.add('context');svg.setAttribute('viewBox','0 0 '+region.size.join(' '));svg.setAttribute('role','img');svg.setAttribute('aria-label','Native vehicle context with raw text and estimated plate outlines');const im=document.createElementNS(ns,'image');im.setAttribute('href',region.path);im.setAttribute('width',region.size[0]);im.setAttribute('height',region.size[1]);svg.append(im);const used=new Set(region.proposals.flatMap(p=>p.text_members));const polygon=(points,colour,dashed)=>{const q=document.createElementNS(ns,'polygon');q.setAttribute('points',points.map(p=>p.join(',')).join(' '));q.setAttribute('fill','none');q.setAttribute('stroke',colour);q.setAttribute('stroke-width',Math.max(1.5,region.size[0]/400));q.setAttribute('vector-effect','non-scaling-stroke');if(dashed)q.setAttribute('stroke-dasharray','5 4');svg.append(q);};region.polygons.forEach(p=>polygon(p,'#2563eb',true));for(const p of region.proposals)polygon(p.quad,colours[p.verdict.status],false);host.append(svg);e('unretained').textContent=region.polygons.length+' raw text outlines · '+region.polygons.filter((_,i)=>!used.has(i)).length+' not retained by bounded geometry search. This does not establish that those outlines are non-plates.';candidates();}
function regions(){const frame=d.frames[Number(e('frame').value)];e('region').replaceChildren();(frame?.regions||[]).forEach((r,i)=>option(e('region'),i,r.id+' · '+r.proposals.length+' candidates'));draw();}
function reference(){const r=d.references[Number(e('reference').value)];if(!r)return;e('reference-title').textContent='Reference '+r.number+' of '+d.references.length;e('actual').textContent=r.actual;e('previous-reading').textContent=r.previous.reading||'No saved reconstructed reading';image('reference-image',r.reference_crop);image('previous-image',r.previous.crop);e('time').textContent=(r.position_ms/1000).toFixed(3)+' s';e('correspondence').textContent=r.correspondence;const selection=r.match||r.inspection,fi=selection?selection[0]:r.frame_indexes[0];if(fi!==undefined){e('frame').value=String(fi);regions();if(selection){e('region').value=String(selection[1]);draw();e('status-filter').value='all';candidates();e('candidate').value=String(selection[2]);showCandidate();}}else{e('frame').value='-1';e('region').replaceChildren();e('context').textContent='No audited frame at this reference timestamp.';e('candidate').replaceChildren();showCandidate();}}
e('notice').textContent=d.notice;e('scope').textContent=d.scope;e('metadata').textContent=JSON.stringify({source_sha256:d.source_sha256,schedule_sha256:d.schedule_sha256,elapsed_seconds:d.elapsed_seconds,summary:d.summary,verifier:d.verifier_metadata,provisional_ocr_policy:d.provisional_ocr_policy,baseline_coverage:d.baseline_coverage,labels_used_during_inference:false},null,2);d.references.forEach((r,i)=>option(e('reference'),i,String(r.number)));d.frames.forEach((f,i)=>option(e('frame'),i,(f.position_ms/1000).toFixed(3)+' s · '+f.regions.length+' regions'));const manual=()=>{e('correspondence').textContent='Manual candidate inspection: this selection does not establish identity correspondence to the human reference.';};e('reference').addEventListener('change',reference);e('frame').addEventListener('change',()=>{manual();regions();});e('region').addEventListener('change',()=>{manual();draw();});e('candidate').addEventListener('change',()=>{manual();showCandidate();});e('status-filter').addEventListener('change',()=>{manual();candidates();});reference();
})();</script></html>'''


def render(capture, baseline, review, output):
    capture, baseline, review, output = (Path(p).resolve() for p in (capture, baseline, review, output))
    if output.exists():
        raise ValueError('Output must be a fresh folder')
    if not output.is_relative_to(ROOT/'data'/'benchmarks') or any(output.is_relative_to(p) or p.is_relative_to(output) for p in (capture, baseline, review)):
        raise ValueError('Output must be separate local benchmark data')
    if subprocess.run(['git','check-ignore','-q','--',str(output/'index.html')],cwd=ROOT,check=False).returncode != 0:
        raise ValueError('Output must be Git-ignored')
    audit, audit_assets = validate_audit(capture)
    manifest, labels = load(review/'manifest.json'), load(review/'labels.json')
    previous, accuracy = validate_capture(baseline, review, manifest, labels)
    if audit['source_sha256'] != manifest['source_sha256']:
        raise ValueError('Audit source differs from human references')
    corners = {}
    rectify = review/'ocr-model-dev'/'rectify.py'
    if rectify.is_file():
        for node in ast.parse(rectify.read_text(encoding='utf-8')).body:
            if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in ('CORNERS','TEXT_REGIONS') for target in node.targets):
                corners.update(ast.literal_eval(node.value))
    references = unique(manifest['samples'],'id','human reference')
    data = {key:copy.deepcopy(audit.get(key)) for key in ('source_sha256','scope','elapsed_seconds','verifier_metadata','schedule_sha256','summary','frames','provisional_ocr_policy')}
    data['notice'], data['references'] = NOTICE, []
    copies = dict(audit_assets)
    remap = {relative:f'assets/capture-{i:05}.png' for i, relative in enumerate(copies)}
    baseline_found = 0
    for number, case in enumerate(accuracy['cases'],1):
        sample = references[case['id']]
        old = choose(previous,case)
        baseline_found += old['found']
        row = dict(number=number,reference_id=case['id'],actual=case['expected'],position_ms=sample['position_ms'],
                   previous=old,reference_crop=f'assets/reference-{number:02}.png',**associate(sample,audit,corners))
        copies['reference:'+case['id']] = safe_file(review,sample['crop'])
        remap['reference:'+case['id']] = row['reference_crop']
        if old['crop']:
            source = safe_file(baseline,old['crop'])
            with Image.open(source) as image:
                if old['native_source_id'] and hashlib.sha256(image.convert('RGB').tobytes('raw','BGR')).hexdigest() != old['native_source_id']:
                    raise ValueError('Baseline native hash mismatch')
            key='baseline:'+str(number)
            copies[key]=source;remap[key]=f'assets/baseline-{number:02}.png';row['previous']['crop']=remap[key]
        data['references'].append(row)
    for frame in data['frames']:
        for region in frame['regions']:
            with Image.open(audit_assets[region['path']]) as image:
                region['size']=list(image.size)
            region['path']=remap[region['path']]
            for proposal in region['proposals']:
                for key in ('native_path','rectified_path'):
                    proposal[key]=remap[proposal[key]]
    data['baseline_coverage']=dict(correct=baseline_found,total=len(data['references']),
        rule='Saved active reconstructed top found anywhere; extra guesses not penalised. New audit is not scored.')
    data['bindings']=dict(audit_sha256=sha(capture/'audit.json'),baseline_predictions_sha256=sha(baseline/'predictions.json'),
                          labels_sha256=sha(review/'labels.json'),manifest_sha256=sha(review/'manifest.json'))
    (output/'assets').mkdir(parents=True)
    for key, source in copies.items():
        shutil.copyfile(source,output/remap[key])
    (output/'audit-view.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (output/'index.html').write_text(html(data,output),encoding='utf-8')
    return data


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','baseline','review','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=render(args.capture,args.baseline,args.review,args.output)
    print(json.dumps(dict(stage='text_corner_audit_rendered',references=len(result['references']),frames=len(result['frames']))))
