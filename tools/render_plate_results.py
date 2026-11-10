"""Simple local review of labelled plates and reconstructed whole-clip results.

No inference or label changes. Count each reference once when its complete text
appears as an active reconstructed top reading anywhere in the saved capture.
Keep generated crops, actual plate text and this local HTML outside Git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_results(document,accuracy,manifest):
    if document['source_sha256']!=manifest['source_sha256']:
        raise ValueError('Reference images belong to a different source')
    references={s['id']:s for s in manifest['samples']}
    active={g['id']:g for g in document['groups'] if not g.get('plate_validation',{}).get('excluded_from_fusion')}
    results=[]
    for number,case in enumerate(accuracy['cases'],1):
        matches=[g for g in active.values() if g['methods']['reconstruction']['top']==case['expected']]
        # Choosing an exact match implements the user's explicit found-anywhere
        # metric. Do not suggest that other guesses were absent or verified.
        group=matches[0] if matches else active.get(case['group'])
        sample=max(group['samples'],key=lambda s:(s['quality'],-s['position_ms'])) if group and group['samples'] else None
        results.append(dict(number=number,reference_id=case['id'],actual=case['expected'],found=bool(matches),
            ocr=group['methods']['reconstruction']['top'] if group else None,
            reference_crop='reference-crops/'+Path(references[case['id']]['crop']).name,
            automatic_crop=sample['path'] if sample else None,history=group['id'] if group else None,
            note='Correct reading found at least once in the clip.' if matches else 'No exact reconstructed reading found anywhere in the clip.'))
    count=sum(r['found'] for r in results)
    return dict(results=results,correct=count,total=len(results),percent=round(100*count/len(results),1) if results else 0,
        rule='Full-number match in any active reconstructed top output anywhere in the clip; count each labelled example once. Extra wrong guesses are not penalised.')


def render_html(data):
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    return '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><title>Sentinel · Plate results</title>
<style>
:root{--paper:#f6f7f2;--ink:#18352e;--muted:#59695f;--teal:#0f766e;--line:#d5ddd3}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:1050px;margin:44px auto;padding:0 24px}header{display:flex;justify-content:space-between;align-items:center;gap:24px}
h1{font-size:28px;letter-spacing:-.8px;margin:0}header p{margin:6px 0 0;color:var(--muted)}.score{white-space:nowrap;color:var(--teal);font-weight:700;font-size:24px}
.rule{font-size:14px;color:var(--muted);margin:22px 0 30px}.toolbar{display:flex;align-items:center;gap:16px;margin:0 0 20px}
label{font-weight:600}select{font:inherit;background:white;border:1px solid #93aaa1;border-radius:6px;padding:9px 38px 9px 12px;min-width:105px;color:var(--ink)}
select:focus-visible,a:focus-visible{outline:3px solid #79b9b0;outline-offset:3px}section{background:#fff;border:1px solid var(--line);border-radius:10px;padding:26px}
.title-row{display:flex;justify-content:space-between;gap:12px;align-items:center}h2{margin:0;font-size:21px}.badge{font-size:13px;font-weight:650;background:#e7f3ed;color:#126353;padding:5px 10px;border-radius:5px}.badge.miss{color:#8e482c;background:#f9ede5}
.crops{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin:25px 0}figure{margin:0;min-width:0}figcaption{font-size:13px;color:var(--muted);margin-bottom:9px}.image-area{height:190px;display:grid;place-items:center;background:#f1f3ed;border:1px solid #e2e7de;border-radius:6px;padding:12px}
img{width:100%;height:100%;max-height:164px;object-fit:contain}.empty{padding:20px;font-size:14px;text-align:center;color:var(--muted)}
.readings{display:grid;grid-template-columns:1fr 1fr;gap:22px;border-top:1px solid var(--line);padding-top:22px}dt{font-size:13px;color:var(--muted);margin-bottom:8px}dd{margin:0;font:600 clamp(18px,3vw,26px)/1.3 ui-monospace,Consolas,monospace;overflow-wrap:anywhere}
.note{font-size:14px;color:var(--muted);margin:22px 0 0}footer{font-size:13px;color:var(--muted);margin-top:18px}a{color:var(--teal)}[hidden]{display:none!important}
@media(max-width:600px){main{margin:24px auto;padding:0 16px}header{align-items:flex-start;flex-direction:column;gap:10px}h1{font-size:25px}.score{font-size:21px}.crops,.readings{grid-template-columns:1fr;gap:18px}section{padding:18px}.image-area{height:165px}img{max-height:139px}.title-row{align-items:flex-start}.rule{margin:16px 0 22px}}
</style></head><body><main>
<header><div><h1>Plate results</h1><p>Reconstructed multi-frame OCR</p></div><div class="score" id="score"></div></header>
<p class="rule">A plate counts when its complete correct number is found anywhere in the clip. Extra wrong guesses are not penalised.</p>
<div class="toolbar"><label for="plate-select">Choose plate</label><select id="plate-select" aria-controls="plate-panel"></select></div>
<section id="plate-panel" aria-live="polite" aria-atomic="true">
<div class="title-row"><h2 id="plate-title"></h2><span class="badge" id="status"></span></div>
<div class="crops"><figure><figcaption>Your labelled plate crop</figcaption><div class="image-area"><img id="reference-crop" alt="Your manually labelled plate crop"></div></figure>
<figure><figcaption>Example automatic crop used by OCR</figcaption><div class="image-area"><img id="automatic-crop" alt="Automatic crop from the selected OCR history"><span id="no-crop" class="empty" hidden>No corresponding automatic crop identified.</span></div></figure></div>
<dl class="readings"><div><dt>OCR reading · reconstructed</dt><dd id="ocr-reading"></dd></div><div><dt>Actual reading · your label</dt><dd id="actual-reading"></dd></div></dl>
<p id="result-note" class="note"></p></section>
<footer>Each labelled example counts once. The automatic image is one original crop from its multi-frame evidence. <a id="diagnostics" href="engineering.html">Detailed diagnostics</a></footer>
<noscript>Enable JavaScript to select a plate and compare its readings.</noscript>
<script id="results-data" type="application/json">'''+payload+'''</script>
<script>
(()=>{
 const data=JSON.parse(document.getElementById('results-data').textContent),select=document.getElementById('plate-select');
 const el=id=>document.getElementById(id);
 el('score').textContent=data.correct+' / '+data.total+' found · '+data.percent.toFixed(1)+'%';
 for(const row of data.results){const option=document.createElement('option');option.value=String(row.number);option.textContent=String(row.number);select.append(option);}
 function show(number){
  const row=data.results.find(r=>r.number===number)||data.results[0];if(!row)return;
  select.value=String(row.number);el('plate-title').textContent='Plate '+row.number+' of '+data.total;
  el('status').textContent=row.found?'Found':'Not found';el('status').className='badge'+(row.found?'':' miss');
  el('reference-crop').src=row.reference_crop;el('automatic-crop').hidden=!row.automatic_crop;el('no-crop').hidden=!!row.automatic_crop;
  if(row.automatic_crop)el('automatic-crop').src=row.automatic_crop;else el('automatic-crop').removeAttribute('src');
  el('ocr-reading').textContent=row.ocr||'No exact reading found';el('actual-reading').textContent=row.actual;el('result-note').textContent=row.note;
  el('diagnostics').href='engineering.html'+(row.history?'#'+encodeURIComponent(row.history):'');
 }
 function fromHash(){const match=location.hash.match(/^#plate-([0-9]+)$/);show(match?Number(match[1]):1);}
 select.addEventListener('change',()=>{history.replaceState(null,'','#plate-'+select.value);show(Number(select.value));});
 window.addEventListener('hashchange',fromHash);fromHash();
})();
</script></main></body></html>'''


def render(capture,review):
    capture,review=Path(capture).resolve(),Path(review).resolve()
    prediction=capture/'predictions.json';document=json.loads(prediction.read_bytes());accuracy=json.loads((capture/'accuracy.json').read_bytes())
    manifest=json.loads((review/'manifest.json').read_bytes())
    if accuracy['predictions_sha256']!=sha(prediction) or accuracy['labels_sha256']!=sha(review/'labels.json'):
        raise ValueError('Score, source capture and human labels must match')
    data=build_results(document,accuracy,manifest)
    reference_by_id={s['id']:s for s in manifest['samples']}
    copies=[]
    for row in data['results']:
        sample=reference_by_id[row['reference_id']];source=(review/sample['crop']).resolve()
        if not source.is_relative_to(review) or sha(source)!=sample['crop_sha256']:
            raise ValueError('Labelled crop changed or escapes the review folder')
        copies.append((source,capture/row['reference_crop']))
        if row['automatic_crop']:
            path=(capture/row['automatic_crop']).resolve()
            if not path.is_relative_to(capture/'assets') or not path.is_file():raise ValueError('Automatic crop is not a saved source asset')
    for source,target in copies:
        target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
    # Preserve the full original viewer once, before replacing its landing page.
    if (capture/'index.html').exists() and not (capture/'engineering.html').exists():
        shutil.copyfile(capture/'index.html',capture/'engineering.html')
    (capture/'simple-results.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    page=capture/'index.html.tmp';page.write_text(render_html(data),encoding='utf-8');page.replace(capture/'index.html')
    print(json.dumps(dict(stage='simple_results_rendered',correct=data['correct'],total=data['total'],percent=data['percent'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True);args=parser.parse_args();render(args.capture,args.review)
