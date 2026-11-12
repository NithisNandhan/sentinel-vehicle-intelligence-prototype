"""Local-only side-by-side model results; no frame/plate text publication."""
import argparse
import json
import shutil
from pathlib import Path


def render(results, reference_view, output):
    report=json.loads((results/'comparison.json').read_bytes())
    refs=json.loads(reference_view.read_bytes())['references']
    if output.exists():raise ValueError('Fresh viewer directory required')
    (output/'assets').mkdir(parents=True)
    data=dict(report=report,references=refs)
    for ref in refs:
        relative=ref['reference_crop'];source=reference_view.parent/relative
        target=f"assets/reference-{ref['number']:02}.png"
        shutil.copyfile(source,output/target);ref['image']=target
    for model in report['models']:
        for case in model['cases']:
            if not case['found']:continue
            crop=case['crop']
            for kind in ('native','rectified'):
                key=kind+'_path'
                if key not in crop:continue
                source=(results/model['model']/crop[key]).resolve()
                if not source.is_relative_to(results.resolve()):raise ValueError('Asset outside results')
                target=f"assets/{model['model']}-{case['number']:02}-{kind}.png"
                shutil.copyfile(source,output/target);crop[key]=target
    (output/'comparison-view.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    payload=json.dumps(data).replace('<','\\u003c')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sentinel · plate detector comparison</title>
<style>:root{font:16px/1.5 system-ui;color:#183b35;background:#f6f7f2}body{max-width:1400px;margin:28px auto;padding:0 20px}h1,h2{line-height:1.2}h1{font-size:28px}h2{font-size:20px}p{max-width:1050px}.muted{color:#526a62;font-size:14px}.panel{background:white;border:1px solid #cbd8d1;border-radius:8px;padding:18px;margin:18px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #dce4dc;text-align:left;white-space:nowrap}th{font-size:14px}select{font:inherit;padding:7px 12px;border:1px solid #7e9a91;background:white;border-radius:5px}select:focus-visible{outline:3px solid #0f766e}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card{background:white;border:1px solid #cbd8d1;padding:16px;border-radius:8px}.views{display:grid;grid-template-columns:1fr 1fr;gap:12px}.image{min-height:110px;background:#edf1e9;display:flex;align-items:center;justify-content:center;padding:8px}.image img{max-width:100%;max-height:150px;object-fit:contain}code{font-size:17px;overflow-wrap:anywhere}.ok{color:#0f766e;font-weight:600}.bad{color:#a8432e}.badge{font-size:13px}.truth{display:flex;align-items:center;gap:20px;flex-wrap:wrap}.truth img{max-width:330px;max-height:160px}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:800px){.cards{grid-template-columns:1fr}.views{grid-template-columns:1fr 1fr}}@media(max-width:420px){.views{grid-template-columns:1fr}}</style>
<h1>Plate detection: same frames, same OCR</h1><p>149 saved automatic vehicle regions · 14 timestamps · 18 reviewed development references. Models receive identical vehicle images. Correct readings and manual boundaries are used only after inference for evaluation.</p>
<p class="muted">This measures reference coverage and OCR on a selected clip. Other predictions are unlabelled, so false-positive precision is unavailable. The crop displayed is the best geometric match (IoU ≥ 0.30), selected without looking at the OCR answer. Results are single-frame, not the previous full-video reconstructed score.</p>
<div class="panel scroll"><table id="summary"></table></div><div class="panel"><label for="ref">Reference </label><select id="ref"></select><div class="truth" id="truth"></div></div><div class="cards" id="cards"></div>
<p class="muted">A detection is not a confirmed identity. All readings here are diagnostic; none generate watchlist alerts. YOLO11 publisher warns of contaminated training/test splits; published accuracy claims are not used. Detector timings exclude warmup and OCR. CPU IWPOD and GPU YOLO timings are labelled separately.</p>
<script>const D=__PAYLOAD__;const e=id=>document.getElementById(id);const text=(tag,value,cls)=>{const n=document.createElement(tag);n.textContent=value;if(cls)n.className=cls;return n;};const head=['Model','Device','Found / 18','≥95% reference box covered','Native exact / 18','Rectified exact / 18','Median / ROI'];const table=e('summary');const tr=document.createElement('tr');head.forEach(v=>tr.append(text('th',v)));table.append(tr);D.report.models.forEach(m=>{const r=document.createElement('tr');[m.model,m.device,m.found_iou30,m.complete_reference_box_95,m.native_exact,m.model==='iwpod'?m.rectified_exact:'—',m.median_ms.toFixed(1)+' ms'].forEach(v=>r.append(text('td',String(v))));table.append(r)});D.references.forEach(r=>{const o=document.createElement('option');o.value=r.number;o.textContent=r.number;e('ref').append(o)});
function draw(){const number=Number(e('ref').value),r=D.references.find(v=>v.number===number);e('truth').replaceChildren();const im=document.createElement('img');im.src=r.image;im.alt='Manual reference crop '+number;e('truth').append(im,text('p','Actual: '+r.actual));e('cards').replaceChildren();for(const m of D.report.models){const c=m.cases.find(v=>v.number===number),host=document.createElement('section');host.className='card';host.append(text('h2',m.model));if(!c.found){host.append(text('p','No qualifying plate detection','bad'));host.append(text('p',c.roi_available?'A covering automatic vehicle region was available.':'No automatic vehicle region covered at least 95% of the reference.','muted'));}else{host.append(text('p','IoU '+c.iou.toFixed(2)+' · reference box coverage '+(100*c.reference_box_coverage).toFixed(1)+'% · detector score '+c.confidence.toFixed(3),'badge'));const grid=document.createElement('div');grid.className='views';for(const k of ['native','rectified']){if(!c.crop[k+'_path'])continue;const v=document.createElement('div');v.append(text('p',k==='native'?'Native crop':'Model rectification','muted'));const ih=document.createElement('div');ih.className='image';const img=document.createElement('img');img.src=c.crop[k+'_path'];img.alt=m.model+' '+k+' crop';ih.append(img);v.append(ih);const reading=c.crop[k+'_reading'];v.append(text('code',reading||'(no text)',c[k+'_exact']?'ok':'bad'));v.append(text('p',c[k+'_exact']?'Exact match':'Does not match reference','muted'));grid.append(v)}host.append(grid)}e('cards').append(host)}}e('ref').addEventListener('change',draw);draw();</script></html>'''
    (output/'index.html').write_text(page.replace('__PAYLOAD__',payload),encoding='utf-8')
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--reference-view',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=render(a.results,a.reference_view,a.output);print(json.dumps(dict(models=len(d['report']['models']),references=len(d['references']))))
