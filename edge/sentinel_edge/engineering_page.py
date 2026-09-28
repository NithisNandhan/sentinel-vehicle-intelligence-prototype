"""Standalone, memory-only local engineering console.

This renderer contains no inference logic and loads no external resources.
The companion HTTP server owns authentication, shared capture and expiration.
"""
from __future__ import annotations

import html
import json


def engineering_page(token: str, detector_profile: str | None = None) -> str:
    """Render the engineering UI without interpolating executable model text."""
    # JSON escaping alone does not protect the HTML script closing delimiter.
    safe_token = json.dumps(str(token)).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    profile = html.escape(str(detector_profile or "Profile reported by the shared session"))
    before, _, after = PAGE.partition("__TOKEN__")
    return before.replace("__PROFILE__", profile) + safe_token + after.replace("__PROFILE__", profile)


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentinel · Engineering console</title>
<style>
:root{color-scheme:dark;--bg:#080e17;--panel:#101b2a;--panel2:#132235;--line:#25364d;--text:#edf3fa;--muted:#a1b0c4;--cyan:#68e2da;--amber:#ffc76a;--red:#ff8c94;--green:#88dba0;--mono:ui-monospace,SFMono-Regular,Consolas,monospace}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 80% 0%,#13263c 0,transparent 50%),var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}button,input,select{font:inherit}button,a{touch-action:manipulation}button{border:1px solid var(--line);background:#1b3048;color:var(--text);padding:10px 15px;border-radius:8px;cursor:pointer;font-weight:600}button:hover:not(:disabled){background:#27435f;border-color:#577996}button:disabled{opacity:.45;cursor:not-allowed}.primary{background:#176e6b;border-color:#40b5ac}.primary:hover:not(:disabled){background:#238b85}button:focus-visible,input:focus-visible,select:focus-visible,a:focus-visible{outline:2px solid var(--cyan);outline-offset:3px}input,select{background:#0b1522;border:1px solid #3a526b;border-radius:8px;padding:10px 12px;color:var(--text);min-width:0}input{width:260px}select{max-width:440px;width:100%}a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}[hidden]{display:none!important}h1,h2,h3,p{margin-top:0}h1{font-size:30px;line-height:1.2;letter-spacing:-.8px;margin-bottom:8px}h2{font-size:17px;margin-bottom:14px}h3{font-size:14px}.muted{color:var(--muted)}.mono{font-family:var(--mono)}.eyebrow{color:var(--cyan);font-size:10px;letter-spacing:2px;font-weight:700;text-transform:uppercase}.small{font-size:12px}.shell{max-width:1720px;margin:auto;padding:24px 30px 36px}.topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;padding-bottom:20px;border-bottom:1px solid var(--line);margin-bottom:22px}.brand{display:flex;align-items:center;gap:12px;font-weight:700;letter-spacing:.3px}.brandmark{background:#1b4748;color:var(--cyan);border:1px solid #2d6967;border-radius:10px;width:38px;height:38px;display:grid;place-items:center;font-size:20px}.toplinks{display:flex;gap:20px;align-items:center;font-size:13px}.pill{display:inline-flex;align-items:center;gap:6px;padding:4px 9px;border:1px solid var(--line);border-radius:20px;background:#152136;font-size:11px;white-space:nowrap}.pill.warn{color:var(--amber);border-color:#715939;background:#332a1f}.pill.live{color:var(--green);border-color:#3a6848;background:#182d23}.pill.error{color:var(--red);border-color:#6e3745}.dot{width:6px;height:6px;background:currentColor;border-radius:50%}.intro{display:flex;align-items:flex-start;justify-content:space-between;gap:20px}.intro p{max-width:750px}.panel{background:linear-gradient(145deg,#122034,#101b2a);border:1px solid var(--line);border-radius:13px;padding:18px}.controls{display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin:18px 0}.controlfield{flex:1 1 300px}.controlfield label{display:block;font-size:11px;color:var(--muted);margin-bottom:5px}.controlbuttons{display:flex;gap:8px;align-self:flex-end}.catalogue{flex-basis:100%;margin:0;font-size:12px}.auth{display:flex;gap:12px;align-items:center;flex-wrap:wrap;background:#0b1726}.authcopy{flex:1 1 280px}.authcopy p{margin:0}.auth form{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.auth label{position:absolute;width:1px;height:1px;overflow:hidden}.statusline{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:18px 0 12px;color:var(--muted)}#message{margin:0;flex:1}.metrics{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin-bottom:18px}.metric{padding:13px 15px;background:#0e1a29;border:1px solid var(--line);border-radius:10px}.metric .value{font:600 24px/1.25 var(--mono);margin-top:5px;display:block}.metric .label{font-size:10px;letter-spacing:.7px;text-transform:uppercase;color:var(--muted)}.flow{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;margin:0 0 18px}.step{background:#0e1b2a;border:1px solid var(--line);border-radius:9px;padding:11px 12px;position:relative}.step:after{content:'›';position:absolute;right:-7px;top:15px;color:#6f91ac;z-index:1}.step:last-child:after{display:none}.step b{display:block;font-size:12px;margin-top:4px}.step span{font-size:11px;color:var(--muted)}.workgrid{display:grid;grid-template-columns:minmax(0,2fr) minmax(300px,1fr);gap:18px;align-items:start}.panelhead{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:12px}.panelhead h2{margin:0}.viewport{position:relative;display:grid;place-items:center;aspect-ratio:16/9;background:#030910;border:1px solid #1c3045;border-radius:9px;overflow:hidden}.viewport img{display:block;width:100%;height:100%;object-fit:contain;max-height:66vh}.placeholder{padding:32px;text-align:center;color:var(--muted);max-width:550px}.placeholder strong{display:block;color:var(--text);font-size:17px;margin-bottom:8px}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin:12px 0 0}.legend .vehicle{color:var(--green)}.legend .plate{color:var(--amber)}.facts{display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;margin:15px 0 0}.facts dt{color:var(--muted)}.facts dd{margin:0;text-align:right;font-family:var(--mono);overflow-wrap:anywhere}.tracklist{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:14px;max-height:190px;overflow:auto}.trackbutton{padding:7px 10px;font:12px var(--mono);text-align:left;background:#101c2e}.trackbutton.selected{border-color:var(--cyan);background:#153e44}.trackbutton small{display:block;font:10px system-ui;color:var(--muted);margin-top:3px}.trackidentity{display:flex;justify-content:space-between;align-items:center;gap:10px;padding-bottom:10px;border-bottom:1px solid var(--line);margin-bottom:12px}.trackidentity strong{font-family:var(--mono);font-size:18px}.cropview{background:#040b13;border:1px solid var(--line);border-radius:8px;padding:14px;display:grid;place-items:center;min-height:95px;overflow:auto}.cropview img{image-rendering:pixelated;max-width:100%;display:block}.callout{padding:11px 13px;border-left:3px solid #668aa6;background:#10253b;border-radius:0 7px 7px 0;font-size:12px;color:#c1d0e2;margin-top:13px}.callout.warn{border-color:var(--amber);background:#2b271f;color:#ead7b7}.gates{padding-left:18px;font-size:12px;color:var(--amber);margin:8px 0}.gates li{margin:5px 0}.below{margin-top:18px}.evidencegrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.evidencecard{border:1px solid var(--line);background:#0a1624;border-radius:9px;overflow:hidden}.evidencetop{padding:10px 12px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:8px;font-size:10px;color:var(--muted)}.evidenceimage{min-height:90px;padding:12px;display:grid;place-items:center;background:#040c15;overflow:auto}.evidenceimage img{max-width:100%;image-rendering:pixelated}.evidencebody{padding:12px}.frametext{font:600 17px var(--mono);overflow-wrap:anywhere;margin-bottom:10px}.characters{display:flex;gap:4px;flex-wrap:wrap}.character{padding:5px 6px;border-radius:5px;border:1px solid #34465a;text-align:center;min-width:27px;background:#122538;font:14px var(--mono)}.character small{display:block;font:9px system-ui;color:var(--muted);margin-top:4px}.character.weak{border-color:#715a35;color:var(--amber)}.evidencefoot{font-size:10px;color:var(--muted);margin:10px 0 0}.tablewrap{overflow:auto}.candidatetable{width:100%;border-collapse:collapse;text-align:left;font-size:12px;white-space:nowrap}.candidatetable th{color:var(--muted);font-weight:500;font-size:10px;letter-spacing:.5px;text-transform:uppercase;padding:10px;border-bottom:1px solid var(--line)}.candidatetable td{padding:10px;border-bottom:1px solid #1b2b3f;font-family:var(--mono)}.candidatetable tr:first-child td{color:var(--cyan)}.emptyrow{padding:18px!important;color:var(--muted)!important;font-family:system-ui!important;white-space:normal}.dual{display:grid;grid-template-columns:minmax(0,3fr) minmax(280px,2fr);gap:18px;margin-top:18px}.stagebars{display:grid;gap:12px;margin:16px 0}.stagebar{display:grid;grid-template-columns:70px 1fr 70px;gap:12px;align-items:center;font-size:12px}.bartrack{height:7px;background:#203046;border-radius:8px;overflow:hidden}.barfill{height:100%;background:var(--cyan);border-radius:8px;width:0}.stagebar .time{text-align:right;font-family:var(--mono)}.footer{border-top:1px solid var(--line);margin-top:24px;padding-top:18px;font-size:11px;color:var(--muted)}.lockedbanner{border:1px dashed #33536c;border-radius:13px;padding:55px 20px;text-align:center;margin-top:18px;background:#0b1623}.lockedbanner p{margin:8px auto 0;max-width:620px;color:var(--muted)}#hypothesis-note{margin:0;font-size:11px;color:var(--amber)}#candidate-count{font-size:11px;color:var(--muted)}@media(max-width:1150px){.workgrid{grid-template-columns:1fr}.evidencegrid{grid-template-columns:repeat(3,minmax(0,1fr))}.metrics{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:750px){.shell{padding:16px}.topbar,.intro{align-items:flex-start}.toplinks{gap:10px;flex-wrap:wrap;justify-content:flex-end}.flow{grid-template-columns:repeat(3,minmax(0,1fr))}.step:nth-child(3):after{display:none}.dual{grid-template-columns:1fr}.evidencegrid{grid-template-columns:repeat(2,minmax(0,1fr))}.auth form{width:100%}.auth input{flex:1;width:auto}.controlbuttons{flex-wrap:wrap}.metrics{gap:7px}.metric{padding:10px}.metric .value{font-size:20px}}@media(max-width:440px){.evidencegrid{grid-template-columns:1fr}.toplinks .pill{display:none}.brand{font-size:12px}}
.bestreading{border:1px solid #6e603d;background:linear-gradient(135deg,#342b1c,#1a2531);border-radius:10px;padding:13px 15px;margin-bottom:14px}.bestreading strong{display:block;font:700 clamp(20px,3vw,29px)/1.25 var(--mono);overflow-wrap:anywhere;margin:5px 0}.bestreading p{margin:0;color:var(--muted);font-size:11px}
</style>
</head>
<body><main class="shell">
<header class="topbar"><div class="brand"><span class="brandmark" aria-hidden="true">S</span><span>SENTINEL <span class="muted">/ ENGINEERING</span></span></div><nav class="toplinks" aria-label="Console navigation"><span class="pill">LOCALHOST · MEMORY ONLY</span><a href="http://127.0.0.1:5173/" target="_blank" rel="noopener noreferrer">Operator dashboard ↗</a></nav></header>
<section class="intro"><div><div class="eyebrow">Observable evidence · shared inference session</div><h1>Inside the recognition pipeline</h1><p class="muted">Inspect localisation, native pixels and individual OCR observations before trusting a result. A model-accepted hypothesis is not an independently verified registration.</p></div><span id="connection-state" class="pill"><span class="dot"></span><span id="connection-label">Locked</span></span></section>
<section class="panel auth" aria-label="Protected engineering access"><div class="authcopy"><strong id="auth-title">Unlock sensitive local evidence</strong><p class="small muted" id="auth-description">Use the existing operator key. It stays in this page's memory, never browser storage.</p></div><form id="unlock-form"><label for="operator-key">Operator key</label><input id="operator-key" type="password" autocomplete="off" placeholder="Operator key" spellcheck="false" required><button type="submit" class="primary" id="unlock">Unlock console</button></form><button id="lock" hidden>Lock & clear evidence</button></section>
<section class="panel controls" aria-label="Shared camera controls"><div class="controlfield"><label for="camera-select">Authorised camera — selection does not start a capture</label><select id="camera-select" disabled><option>Loading authorised catalogue…</option></select></div><div class="controlbuttons"><button id="open-camera" class="primary" disabled>Start selected camera</button><button id="pause" disabled>Pause shared session</button></div><p id="catalogue-status" class="muted catalogue">Loading catalogue status…</p><p class="small muted catalogue">One shared diagnostic camera. Starting another camera replaces this session for both interfaces. Opening either interface does not duplicate inference. Other production workers are separate.</p></section>
<div class="statusline"><span id="session-state" class="pill">No active session</span><p id="message" role="status" aria-live="polite">Unlock to inspect image evidence. No camera starts automatically.</p></div>
<section class="metrics" aria-label="Shared processing counters"><div class="metric"><span class="label">Decoded frames</span><span id="decoded" class="value">—</span></div><div class="metric"><span class="label">Analysed frames</span><span id="analysed" class="value">—</span></div><div class="metric"><span class="label">Vehicle boxes now</span><span id="vehicles" class="value">—</span></div><div class="metric"><span class="label">Plate crops</span><span id="plates" class="value">—</span></div><div class="metric"><span class="label">Distinct frames / track</span><span id="observations" class="value">—</span></div><div class="metric"><span class="label">Accepted · unverified</span><span id="accepted" class="value">—</span></div></section>
<section class="flow" aria-label="Pipeline stages"><div class="step"><span>01 · Source</span><b>Original frame</b></div><div class="step"><span>02 · Tracking</span><b>Vehicle localisation</b></div><div class="step"><span>03 · Native pixels</span><b>Plate crop</b></div><div class="step"><span>04 · Deep OCR</span><b>Per-frame evidence</b></div><div class="step"><span>05 · Distinct frames</span><b>Temporal fusion</b></div><div class="step"><span>06 · Decision</span><b>Candidate & gates</b></div></section>
<div id="locked-banner" class="lockedbanner"><div class="eyebrow">Protected local inspection</div><h2>Images and full hypotheses are hidden</h2><p>Unlock with the operator key to inspect the currently active shared session or explicitly start a camera. The console does not save media or registrations.</p></div>
<div id="inspection" hidden>
<div class="workgrid"><section class="panel"><div class="panelhead"><h2>Analysed frame</h2><span id="camera-label" class="pill">No camera</span></div><div role="group" aria-label="Frame display mode" style="display:flex;gap:8px;margin-bottom:12px"><button id="show-overlays" aria-pressed="true">Detection overlays</button><button id="show-original" aria-pressed="false" disabled>Original scene</button></div><div class="viewport"><img id="frame" alt="Current analysed live frame with aligned vehicle and plate boxes" hidden><div id="frame-placeholder" class="placeholder"><strong>Waiting for live evidence</strong><span>Select a camera and start the shared session.</span></div></div><p class="legend"><span class="vehicle">Green · vehicle prediction</span><span class="plate">Amber · plate proposal</span><span>Sampled inference · scene displays use JPEG; native crops below use PNG</span></p><dl class="facts"><dt>Model profile</dt><dd id="model-profile">__PROFILE__</dd><dt>Native frame resolution</dt><dd id="resolution">—</dd><dt>Video PTS</dt><dd id="pts">—</dd><dt>Displayed frame age</dt><dd id="age">—</dd><dt>Latest processing time</dt><dd id="processing">—</dd><dt>Overwritten decoder frames</dt><dd id="overwritten">—</dd></dl></section>
<aside class="panel"><div class="panelhead"><h2>Tracked vehicle evidence</h2><span id="track-count" class="pill">0 tracks</span></div><div id="track-list" class="tracklist" aria-label="Select a tracked vehicle"></div><div id="track-empty" class="muted small">No recent tracked plate evidence. Vehicle boxes alone do not mean a plate was readable.</div><div id="track-detail" hidden><div class="trackidentity"><strong id="track-id"></strong><span id="track-state" class="pill warn"></span></div><div class="bestreading"><span id="best-reading-label" class="eyebrow">Best available guess · unverified</span><strong id="best-reading-value">—</strong><p id="best-reading-note">A model suggestion, not a confirmed registration.</p></div><div class="cropview"><img id="track-crop" alt="Native plate crop from the selected vehicle, lossless PNG" hidden><span id="track-crop-empty" class="muted small">No native crop supplied</span></div><p id="crop-meta" class="muted small"></p><dl class="facts"><dt>Vehicle class</dt><dd id="vehicle-class">—</dd><dt>Vehicle box</dt><dd id="vehicle-box">—</dd><dt>Distinct observations</dt><dd id="track-observations">—</dd><dt>Last evidence PTS</dt><dd id="track-pts">—</dd><dt>Masked hypothesis</dt><dd id="masked-plate">—</dd></dl><div class="callout warn"><strong>Acceptance diagnostics</strong><ul id="gate-list" class="gates"></ul></div><div class="callout"><strong>Reconstruction</strong><p id="reconstruction-state" class="small" style="margin:6px 0 0"></p></div></div></aside></div>
<section class="panel below"><div class="panelhead"><h2>Distinct native crops & per-frame OCR</h2><span id="evidence-count" class="pill">0 frames</span></div><p class="muted small">Lossless crops retain their native pixels; on-screen enlargement adds no detail. Character scores are model outputs, not calibrated probabilities. Hover or focus a character to inspect alternatives.</p><div id="evidence-grid" class="evidencegrid"></div><p id="evidence-empty" class="muted small">Select a track with OCR observations to inspect its contributing frames.</p></section>
<div class="dual"><section class="panel"><div class="panelhead"><h2>Ranked plate hypotheses</h2><span id="candidate-count">0 retained candidates</span></div><p id="hypothesis-note">Full text is an unverified hypothesis. Ranking scores do not establish correctness.</p><div class="tablewrap"><table class="candidatetable"><thead><tr><th scope="col">Rank</th><th scope="col">Hypothesis</th><th scope="col">Model score</th><th scope="col">Min character</th><th scope="col">Log score</th><th scope="col">Observations</th></tr></thead><tbody id="candidate-body"><tr><td class="emptyrow" colspan="6">No complete candidate yet. Partial frame evidence may still be available above.</td></tr></tbody></table></div><button id="expand-candidates" hidden style="margin-top:12px">Show all retained candidates</button></section><section class="panel"><div class="panelhead"><h2>Processing budget</h2><span id="roi-budget" class="pill">ROI budget —</span></div><p class="muted small">Last analysed frame's measured stages. Bars compare these stages with each other, not accuracy. Missing timings are shown as unavailable.</p><div id="stage-bars" class="stagebars"></div><div class="callout"><strong>Evidence boundaries</strong><p class="small" style="margin:6px 0 0">Repeated frames, resized copies and reconstructed images must not count as additional independent observations. A scene cut or track discontinuity clears temporal evidence.</p></div></section></div>
</div>
<footer class="footer">Protected local research view · No downloads, recording, browser storage or central plate text. Images and OCR hypotheses live in server/browser memory and clear on lock, stale evidence, camera changes or errors. A hidden page stops polling and heartbeats; capture expires after 60 seconds without an active interface heartbeat. Locking or closing this page does not stop a session another interface is using. Model acceptance is not human verification.</footer>
<section class="panel below" aria-label="Frame collection budget"><h2>Frame collection</h2><p id="collection-status" class="muted small">Collection measurements become available after unlocking a live session.</p></section>
</main>
<script>
'use strict';
const viewerToken=__TOKEN__;
const $=id=>document.getElementById(id);
let operatorKey='',activeCamera=null,latestSnapshot=null,selectedTrack=null;
let generation=0,busy=false,catalogueBusy=false,expanded=false,lastGoodEvidence=0,frameMode='overlay';
let catalogueAvailable=false,catalogueCount=0,unlockBusy=false;
const knownStatuses=new Set(['paused','connecting','live','unavailable']);
const safeNumber=(value,digits=0)=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(digits):'—';
const text=(id,value)=>{$(id).textContent=value==null?'—':String(value);};
const node=(tag,className,content)=>{const el=document.createElement(tag);if(className)el.className=className;if(content!=null)el.textContent=String(content);return el;};
function setImage(element,image,mime){
  element.removeAttribute('src');element.hidden=true;
  if(typeof image!=='string'||image.length>12000000||!image.length||!/^[A-Za-z0-9+/]*={0,2}$/.test(image))return false;
  element.src='data:image/'+mime+';base64,'+image;element.hidden=false;return true;
}
function clearEvidence(reason){
  latestSnapshot=null;selectedTrack=null;expanded=false;lastGoodEvidence=0;frameMode='overlay';
  for(const id of ['frame','track-crop']){$(id).removeAttribute('src');$(id).hidden=true;}
  for(const id of ['track-list','evidence-grid','candidate-body','gate-list','stage-bars'])$(id).replaceChildren();
  $('candidate-body').append(emptyCandidateRow('No complete candidate yet. Partial frame evidence may still become available.'));
  for(const id of ['track-id','track-state','crop-meta','vehicle-class','vehicle-box','track-observations','track-pts','masked-plate','reconstruction-state','best-reading-label','best-reading-value','best-reading-note'])text(id,'');
  text('track-count','0 tracks');text('evidence-count','0 frames');text('candidate-count','0 retained candidates');
  $('track-detail').hidden=true;$('track-empty').hidden=false;$('track-crop-empty').hidden=false;
  $('evidence-empty').hidden=false;$('expand-candidates').hidden=true;$('frame-placeholder').hidden=false;
  $('show-original').disabled=true;$('show-original').setAttribute('aria-pressed','false');$('show-overlays').setAttribute('aria-pressed','true');
  const placeholder=$('frame-placeholder');placeholder.replaceChildren(node('strong',null,reason||'Waiting for live evidence'),node('span',null,'No stale images or full hypotheses are retained in this view.'));
}
function emptyCandidateRow(message){const row=node('tr');const cell=node('td','emptyrow',message);cell.colSpan=6;row.append(cell);return row;}
function updateControls(){
  $('open-camera').disabled=!operatorKey||!catalogueCount||unlockBusy;
  $('pause').disabled=!operatorKey||!activeCamera||unlockBusy;
  $('camera-select').disabled=!catalogueCount;
}
function setLocked(reason){
  generation++;operatorKey='';activeCamera=null;clearEvidence(reason||'Console locked');
  $('operator-key').value='';$('unlock-form').hidden=false;$('lock').hidden=true;
  $('inspection').hidden=true;$('locked-banner').hidden=false;
  text('auth-title','Unlock sensitive local evidence');text('auth-description',"Use the existing operator key. It stays in this page's memory, never browser storage.");
  text('connection-label','Locked');$('connection-state').className='pill';
  if(reason)text('message',reason);updateControls();
}
async function request(path,options={}){
  const headers={'X-Viewer-Token':viewerToken};if(operatorKey)headers['X-Operator-Key']=operatorKey;
  if(options.method)headers['Content-Type']='application/json';
  const controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),8000);
  try{const response=await fetch(path,{...options,headers,cache:'no-store',credentials:'same-origin',signal:controller.signal});
    if(!response.ok){const error=new Error(response.status===401||response.status===403?'Access rejected':'Local service unavailable');error.status=response.status;throw error;}
    return await response.json();
  }finally{clearTimeout(timeout);}
}
function handleError(error){
  generation++;clearEvidence('Evidence unavailable');
  if(error&&[401,403].includes(error.status))setLocked('Access rejected. Re-enter the operator key or reload if the viewer session expired.');
  else{ text('message','Local inference service unavailable. Sensitive evidence has been cleared; retrying while this page is visible.');text('connection-label','Unavailable');$('connection-state').className='pill error';}
}
async function catalogue(){
  if(!operatorKey||catalogueBusy||document.visibilityState!=='visible')return;catalogueBusy=true;const epoch=generation;
  try{const result=await request('/cameras');if(epoch!==generation)return;
    const cameras=Array.isArray(result.cameras)?result.cameras.slice(0,10000):[];
    const previous=$('camera-select').value;$('camera-select').replaceChildren();catalogueCount=0;
    for(const camera of cameras){const id=camera.id||camera.camera_id;if(typeof id!=='string'||!id.length)continue;
      const option=node('option',null,id.toUpperCase()+' · '+String(camera.name||camera.location||'Authorised camera'));option.value=id;$('camera-select').append(option);catalogueCount++;}
    if(!catalogueCount){const option=node('option',null,'Catalogue unavailable');option.value='';$('camera-select').append(option);}
    if([...$('camera-select').options].some(option=>option.value===previous))$('camera-select').value=previous;
    catalogueAvailable=result.available===true;
    text('catalogue-status',result.scope==='configured_relays'?'Configured relays only; full catalogue unavailable':catalogueAvailable?catalogueCount+' authorised live cameras · choose explicitly to start':'Catalogue unavailable. Any listed fallback cameras are not the full catalogue.');updateControls();
  }catch(error){text('catalogue-status','Authorised catalogue unavailable. No new camera paths are guessed.');if(error.status===401||error.status===403)handleError(error);}
  finally{catalogueBusy=false;}
}
function summary(snapshot){
  const camera=typeof snapshot.camera_id==='string'?snapshot.camera_id:null;
  if(camera!==activeCamera){generation++;clearEvidence('Shared camera changed');activeCamera=camera;}
  const status=knownStatuses.has(snapshot.status)?snapshot.status:'unavailable';
  text('session-state',(camera?camera.toUpperCase()+' · ':'')+status);text('camera-label',camera?camera.toUpperCase():'No camera');
  text('decoded',safeNumber(snapshot.decoded_frames));text('analysed',safeNumber(snapshot.analysed_frames));text('vehicles',safeNumber(snapshot.vehicle_boxes));text('plates',safeNumber(snapshot.plate_crops));
  text('observations',safeNumber(snapshot.maximum_distinct_observations));text('accepted',safeNumber(snapshot.unique_model_accepted_unverified));
  text('resolution',Array.isArray(snapshot.resolution)?snapshot.resolution.map(value=>safeNumber(value)).join(' × '):'—');
  text('pts',safeNumber(snapshot.pts_ms)+' ms');text('age',safeNumber(snapshot.frame_age_seconds,1)+' s');text('processing',safeNumber(snapshot.processing_ms,1)+' ms');
  if(typeof snapshot.model_profile==='string')text('model-profile',snapshot.model_profile);
  const overwritten=snapshot.debug&&snapshot.debug.overwritten_frames!=null?snapshot.debug.overwritten_frames:snapshot.overwritten_frames;
  text('overwritten',safeNumber(overwritten));updateControls();return status;
}
function renderStageTimes(debug){
  $('stage-bars').replaceChildren();const values=['vehicle','plate','ocr','reconstruction','fusion'];const times=debug.stage_ms||{};
  const maximum=Math.max(1,...values.map(name=>typeof times[name]==='number'&&Number.isFinite(times[name])?Math.max(0,times[name]):0));
  for(const name of values){const row=node('div','stagebar');row.append(node('span',null,name==='ocr'?'OCR':name.charAt(0).toUpperCase()+name.slice(1)));const track=node('div','bartrack');const fill=node('div','barfill');
    const measured=times[name];fill.style.width=(typeof measured==='number'&&Number.isFinite(measured)?Math.max(0,Math.min(100,measured/maximum*100)):0)+'%';track.append(fill);row.append(track,node('span','time',safeNumber(measured,1)+' ms'));$('stage-bars').append(row);}
  text('roi-budget','ROI budget '+safeNumber(debug.roi_budget));
  if(!$('collection-status'))return;
  text('collection-status','Collected native crops: '+safeNumber(debug.collected_native_crops)+' · sampled views: '+safeNumber(debug.sampled_frames)+' · detector queue drops: '+safeNumber(debug.collector_input_drops)+' · OCR queue drops: '+safeNumber(debug.dropped_sampled_frames)+' · crop queue peak: '+safeNumber(typeof debug.decode_queue_peak_bytes==='number'?debug.decode_queue_peak_bytes/(1024*1024):null,1)+' MiB. Decode, vehicle/plate collection and OCR run in separate bounded stages. Video timing uses PTS; display age includes queue waiting.');
}
function paintFrame(){
  if(!operatorKey||!latestSnapshot||document.visibilityState!=='visible')return false;
  if(Date.now()-lastGoodEvidence+latestSnapshot.frame_age_seconds*1000>5000){generation++;clearEvidence('Evidence expired — waiting for a current frame');return false;}
  const original=latestSnapshot.debug&&latestSnapshot.debug.original_frame;$('show-original').disabled=!original;
  if(frameMode==='original'&&!original)frameMode='overlay';
  const image=frameMode==='original'?original:latestSnapshot.frame;
  const valid=setImage($('frame'),image,'jpeg');$('frame-placeholder').hidden=valid;
  if(!valid){clearEvidence('Invalid image evidence — waiting for a current frame');return false;}
  $('frame').alt=frameMode==='original'?'Original analysed camera scene, display JPEG without detection overlays':'Current analysed live frame with aligned vehicle and plate boxes';
  $('show-original').setAttribute('aria-pressed',String(frameMode==='original'));$('show-overlays').setAttribute('aria-pressed',String(frameMode==='overlay'));
  return true;
}
function render(snapshot){
  const status=summary(snapshot);const age=snapshot.frame_age_seconds;
  if(!operatorKey||document.visibilityState!=='visible')return;
  if(status!=='live'||typeof age!=='number'||!Number.isFinite(age)||age>5||age<0){
    clearEvidence(status==='connecting'?'Connecting / loading models':status==='paused'?'Shared session paused':'No current live evidence');
    text('message',status==='connecting'?'Connecting or loading models. No stale evidence is displayed.':status==='paused'?'Shared session paused. Choose a camera and explicitly start it.':'Feed unavailable or older than five seconds. Images and full hypotheses have been cleared.');
    text('connection-label',status==='connecting'?'Connecting':status==='paused'?'Paused':'Stale / unavailable');$('connection-state').className='pill'+(status==='unavailable'?' error':'');return;
  }
  if(!snapshot.frame){clearEvidence('Waiting for a current analysed frame');text('message','The session is live but no image evidence was provided.');return;}
  lastGoodEvidence=Date.now();latestSnapshot=snapshot;
  if(!paintFrame())return;
  text('message','Live evidence · hypotheses are unverified · full text stays in this protected local view.');text('connection-label','Live inspection');$('connection-state').className='pill live';
  const debug=snapshot.debug&&typeof snapshot.debug==='object'?snapshot.debug:{};renderStageTimes(debug);
  const tracks=Array.isArray(debug.tracks)?debug.tracks.slice(0,128):[];
  if(!tracks.some(track=>String(track.track_id)===selectedTrack)){selectedTrack=tracks.length?String(tracks[0].track_id):null;expanded=false;}
  $('track-list').replaceChildren();text('track-count',tracks.length+' tracks');
  for(const track of tracks){const id=String(track.track_id);const button=node('button','trackbutton'+(id===selectedTrack?' selected':''),'Track '+id);
    button.type='button';button.setAttribute('aria-pressed',String(id===selectedTrack));button.append(node('small',null,safeNumber(track.observations)+' distinct frames'));
    button.addEventListener('click',()=>{selectedTrack=id;expanded=false;renderSelectedTrack();for(const child of $('track-list').children){const picked=child===button;child.classList.toggle('selected',picked);child.setAttribute('aria-pressed',String(picked));}});$('track-list').append(button);}
  renderSelectedTrack();
}
function renderSelectedTrack(){
  const debug=latestSnapshot&&latestSnapshot.debug;const tracks=debug&&Array.isArray(debug.tracks)?debug.tracks:[];
  const track=tracks.find(item=>String(item.track_id)===selectedTrack);
  $('track-detail').hidden=!track;$('track-empty').hidden=!!track;$('evidence-grid').replaceChildren();$('candidate-body').replaceChildren();$('gate-list').replaceChildren();
  if(!track){$('track-crop').removeAttribute('src');$('track-crop').hidden=true;for(const id of ['track-id','track-state','crop-meta','vehicle-class','vehicle-box','track-observations','track-pts','masked-plate','reconstruction-state','best-reading-label','best-reading-value','best-reading-note'])text(id,'');$('candidate-body').append(emptyCandidateRow('No plate-bearing track selected. Vehicle boxes alone are not OCR readings.'));text('candidate-count','0 retained candidates');text('evidence-count','0 frames');$('evidence-empty').hidden=false;$('expand-candidates').hidden=true;return;}
  text('track-id','Track '+String(track.track_id));text('track-state',track.recognition_state||'No complete candidate');text('vehicle-class',track.vehicle_class);text('vehicle-box',Array.isArray(track.vehicle_box)?track.vehicle_box.map(value=>safeNumber(value)).join(', '):'—');
  text('track-observations',safeNumber(track.observations));text('track-pts',safeNumber(track.seen_pts_ms)+' ms');text('masked-plate',track.masked_plate||'No complete hypothesis');
  const frames=Array.isArray(track.frames)?track.frames.slice(-8):[];
  const ranked=Array.isArray(track.candidates)?track.candidates:[];
  const guess=track.best_guess&&typeof track.best_guess.plate==='string'?track.best_guess:ranked[0];
  if(guess&&typeof guess.plate==='string'&&guess.plate){
    text('best-reading-label','Best available guess · unverified');text('best-reading-value',guess.plate);
    text('best-reading-note',safeNumber(guess.observations)+' distinct observation(s) · uncalibrated model score '+safeNumber(guess.score,3)+(track.recognition_state==='model_accepted'?' · model gates passed, not independently verified.':' · review only; no watchlist alert.'));
  }else if(frames.length&&typeof frames[frames.length-1].text==='string'&&frames[frames.length-1].text){
    text('best-reading-label','Partial OCR · not a complete plate');text('best-reading-value',frames[frames.length-1].text);text('best-reading-note','No complete supported plate hypothesis. Do not use this as a registration.');
  }else{text('best-reading-label','No plate reading yet');text('best-reading-value','—');text('best-reading-note','No usable OCR evidence for this track. No plate number has been invented.');}
  const best=frames.length?frames.reduce((a,b)=>(typeof b.quality==='number'?b.quality:0)>(typeof a.quality==='number'?a.quality:0)?b:a):null;
  const crop=best||(latestSnapshot.crop&&latestSnapshot.crop.track_id!=null&&String(latestSnapshot.crop.track_id)===selectedTrack?latestSnapshot.crop:null);
  const cropVisible=crop&&setImage($('track-crop'),crop.image,'png');if(!cropVisible){$('track-crop').removeAttribute('src');$('track-crop').hidden=true;}$('track-crop-empty').hidden=!!cropVisible;
  if(crop){$('track-crop').style.width=(Math.min(3000,Math.max(1,typeof crop.width==='number'?crop.width:1))*3)+'px';text('crop-meta',safeNumber(crop.width)+' × '+safeNumber(crop.height)+' native pixels · lossless PNG · display enlarged only');}else text('crop-meta','No native crop supplied for this track');
  const reasons=Array.isArray(track.gate_reasons)?track.gate_reasons.slice(0,20):[];
  for(const reason of reasons)$('gate-list').append(node('li',null,reason));
  if(!reasons.length)$('gate-list').append(node('li',null,'No gate details supplied. Do not infer correctness from the absence of a rejection.'));
  const reconstruction=track.reconstruction||{status:'not_run',reason:'Reconstruction is not enabled in this session'};
  text('reconstruction-state',String(reconstruction.status||'not_run')+' · '+String(reconstruction.reason||'No reconstruction evidence supplied'));
  if(reconstruction.status==='reconstructed'){
    const card=node('article','evidencecard');const top=node('div','evidencetop');top.append(node('span',null,'RECONSTRUCTED · DERIVED'),node('span','mono','0 extra observations'));card.append(top);
    const wrap=node('div','evidenceimage');const image=node('img');image.alt='Aligned multi-frame reconstruction, not an independent observation';
    if(setImage(image,reconstruction.image,'png'))wrap.append(image);else wrap.append(node('span','muted small','Reconstruction image unavailable'));card.append(wrap);
    const body=node('div','evidencebody');body.append(node('div','frametext',reconstruction.text||'No text decoded'));
    const chars=node('div','characters');for(const item of (Array.isArray(reconstruction.characters)?reconstruction.characters.slice(0,24):[])){
      const element=node('span','character');element.append(node('span',null,item.character||'?'),node('small',null,safeNumber(item.score,2)));element.title='Derived-view character score, not an additional temporal vote';chars.append(element);}
    body.append(chars,node('p','evidencefoot',safeNumber(reconstruction.width)+' × '+safeNumber(reconstruction.height)+' reference-grid pixels · '+safeNumber(reconstruction.contributing_originals)+' aligned originals · '+safeNumber(reconstruction.elapsed_ms,1)+' ms · learned restoration not enabled'));
    for(const item of (Array.isArray(reconstruction.candidates)?reconstruction.candidates.slice(0,5):[]))body.append(node('p','small mono','Derived suggestion: '+String(item.plate||'')+' · score '+safeNumber(item.score,3)));
    body.append(node('p','small muted','Suggestions are rescored on original frames. Reconstruction is not super-resolution or a verified reading.'));card.append(body);$('evidence-grid').append(card);
  }
  text('evidence-count',frames.length+' retained distinct frames');$('evidence-empty').hidden=!!frames.length;
  frames.forEach((frame,index)=>{
    const card=node('article','evidencecard');const top=node('div','evidencetop');top.append(node('span',null,'FRAME '+(index+1)),node('span','mono','PTS '+safeNumber(frame.pts_ms)+' ms'));card.append(top);
    const imageWrap=node('div','evidenceimage');const image=node('img');image.alt='Native plate crop for distinct observation '+(index+1);if(!setImage(image,frame.image,'png'))imageWrap.append(node('span','muted small','No crop supplied'));else imageWrap.append(image);card.append(imageWrap);
    const body=node('div','evidencebody');body.append(node('div','frametext',frame.text||'No text decoded'));
    const chars=node('div','characters');const characters=Array.isArray(frame.characters)?frame.characters.slice(0,24):[];
    for(const item of characters){const element=node('span','character'+(typeof item.score==='number'&&item.score<.75?' weak':''));element.tabIndex=0;element.append(node('span',null,item.character||'?'),node('small',null,safeNumber(item.score,2)));
      const alternatives=Array.isArray(item.alternatives)?item.alternatives.slice(0,5):[];element.title='Character model score: '+safeNumber(item.score,3)+(alternatives.length?' · Alternatives: '+alternatives.map(alt=>String(alt.character)+' '+safeNumber(alt.score,3)).join(', '):' · No alternatives supplied');element.setAttribute('aria-label',String(item.character||'?')+'; '+element.title);chars.append(element);}
    body.append(chars,node('p','evidencefoot',safeNumber(frame.width)+' × '+safeNumber(frame.height)+' native pixels · image quality '+safeNumber(frame.quality,3)+' (heuristic)'));card.append(body);$('evidence-grid').append(card);
  });
  const candidates=Array.isArray(track.candidates)?track.candidates.slice(0,200):[];
  text('candidate-count',candidates.length+' retained candidates · '+(expanded?candidates.length:Math.min(20,candidates.length))+' shown');
  if(!candidates.length)$('candidate-body').append(emptyCandidateRow('No complete candidate yet. Per-frame partial text is evidence, not a confirmed registration.'));
  for(const [index,candidate] of candidates.slice(0,expanded?200:20).entries()){
    const row=node('tr');for(const value of [index+1,candidate.plate||'Incomplete',safeNumber(candidate.score,3),safeNumber(candidate.minimum_character_score,3),safeNumber(candidate.log_score,3),safeNumber(candidate.observations)])row.append(node('td',null,value));$('candidate-body').append(row);
  }
  $('expand-candidates').hidden=candidates.length<=20;text('expand-candidates',expanded?'Show first 20':'Show all '+candidates.length+' retained candidates');
}
async function refresh(){
  if(!operatorKey||busy||unlockBusy||document.visibilityState!=='visible')return;busy=true;const epoch=generation;const key=operatorKey;
  try{const snapshot=await request('/engineering');if(epoch!==generation||key!==operatorKey||document.visibilityState!=='visible')return;
    render(snapshot);
  }catch(error){if(epoch===generation&&key===operatorKey)handleError(error);}finally{busy=false;}
}
async function selectCamera(camera){
  if(!operatorKey)return;generation++;clearEvidence(camera?'Switching shared camera':'Pausing shared session');const epoch=generation;const key=operatorKey;
  text('message',camera?'Starting selected camera for the shared inference session…':'Pausing the shared session…');
  try{await request('/session',{method:'POST',body:JSON.stringify({camera_id:camera})});if(epoch!==generation||key!==operatorKey)return;activeCamera=camera;updateControls();await refresh();}
  catch(error){if(epoch===generation&&key===operatorKey)handleError(error);}
}
$('unlock-form').addEventListener('submit',async event=>{
  event.preventDefault();if(unlockBusy)return;const entered=$('operator-key').value;$('operator-key').value='';if(!entered)return;
  generation++;clearEvidence('Checking protected access');operatorKey=entered;unlockBusy=true;$('unlock').disabled=true;updateControls();const epoch=generation;
  try{const snapshot=await request('/engineering');if(epoch!==generation||operatorKey!==entered)return;
    $('unlock-form').hidden=true;$('lock').hidden=false;$('inspection').hidden=false;$('locked-banner').hidden=true;
    text('auth-title','Engineering evidence unlocked');text('auth-description','Protected local RAM only. Lock clears images, hypotheses and the operator key from this page.');render(snapshot);catalogue();
  }catch(error){if(epoch===generation)setLocked(error.status===401||error.status===403?'Access rejected. Re-enter the operator key or reload if the viewer session expired.':'Unable to validate protected access. The key and evidence were cleared; retry when the service is available.');}finally{unlockBusy=false;$('unlock').disabled=false;updateControls();}
});
$('lock').addEventListener('click',()=>setLocked('Console locked. Sensitive evidence cleared; other interfaces may keep the shared session running.'));
$('open-camera').addEventListener('click',()=>{const camera=$('camera-select').value;if(camera)selectCamera(camera);});
$('pause').addEventListener('click',()=>selectCamera(null));
$('expand-candidates').addEventListener('click',()=>{expanded=!expanded;renderSelectedTrack();});
$('show-original').addEventListener('click',()=>{frameMode='original';paintFrame();});
$('show-overlays').addEventListener('click',()=>{frameMode='overlay';paintFrame();});
document.addEventListener('visibilitychange',()=>{generation++;clearEvidence(document.visibilityState==='visible'?'Refreshing current evidence':'Page hidden — evidence cleared');if(document.visibilityState==='visible'){refresh();catalogue();}});
window.addEventListener('pagehide',()=>{generation++;operatorKey='';clearEvidence('Page closed');});
setInterval(()=>{if(document.visibilityState==='visible'){if(lastGoodEvidence&&latestSnapshot&&Date.now()-lastGoodEvidence+latestSnapshot.frame_age_seconds*1000>5000){generation++;clearEvidence('Evidence expired — waiting for a current frame');text('message','Evidence older than five seconds was cleared.');}refresh();}},1000);
setInterval(()=>{if(document.visibilityState!=='visible'||!operatorKey||!activeCamera||unlockBusy)return;const epoch=generation;const key=operatorKey;
  request('/heartbeat',{method:'POST',body:JSON.stringify({camera_id:activeCamera})}).catch(error=>{if(epoch===generation&&key===operatorKey)handleError(error);});},20000);
setInterval(catalogue,15000);
catalogue();refresh();
</script>
</body></html>'''
