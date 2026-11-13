/* Rehearse, then record the real local dashboard. No mocked detections or truth. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'C:/Users/NITHIS/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base = process.env.DASHBOARD_URL || 'http://127.0.0.1:5173';
const out = path.resolve(process.env.REPLAY_DEMO_OUTPUT || 'output/demos/analysis-replay-20261113');
const rehearseOnly = process.argv.includes('--rehearse');
fs.mkdirSync(out, { recursive: true });
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));

async function visible(locator, label) {
  await locator.waitFor({ state: 'visible', timeout: 30000 });
  console.log('VISIBLE ' + label);
}
async function click(page, locator, label) {
  await visible(locator, label); await locator.scrollIntoViewIfNeeded();
  const box = await locator.boundingBox();
  if (box) await page.mouse.move(box.x+box.width/2, box.y+box.height/2, { steps: 10 });
  await wait(250); await locator.click(); await wait(400);
}
async function caption(page, text) {
  await page.evaluate(text => {
    let bar = document.getElementById('sentinel-demo-caption');
    if (!bar) {
      bar = document.createElement('div'); bar.id = 'sentinel-demo-caption';
      bar.style.cssText = 'position:fixed;bottom:0;left:0;right:0;z-index:999999;padding:10px 18px;background:rgba(8,22,24,.88);color:white;text-align:center;font:16px Segoe UI,sans-serif;pointer-events:none';
      document.body.appendChild(bar);
      const cursor = document.createElement('div');
      cursor.innerHTML = '<svg width="24" height="24"><path d="M5 3L19 12L12 13L9 20Z" fill="white" stroke="black" stroke-width="1.5"/></svg>';
      cursor.style.cssText = 'position:fixed;z-index:1000000;pointer-events:none;left:0;top:0';
      document.body.appendChild(cursor);
      document.addEventListener('mousemove', e => {cursor.style.left=e.clientX+'px';cursor.style.top=e.clientY+'px';});
    }
    bar.textContent = text; bar.style.display = text ? 'block' : 'none';
  }, text);
}
async function status(page) {
  for(let attempt=0;attempt<6;attempt++) {
    const response = await page.request.get(base+'/v1/recognition/status');
    if(response.status()===200)return response.json();
    if(response.status()!==503)throw new Error('Dashboard status rejected: HTTP '+response.status());
    console.warn('Temporary status-poll timeout; retaining the same run and retrying ('+(attempt+1)+'/6)');
    await wait(Math.min(2000,500*(attempt+1)));
  }
  throw new Error('Dashboard status remained unavailable after bounded retries');
}
async function enter(page) {
  await page.goto(base); await visible(page.getByRole('heading',{name:'Analyse footage',exact:true}),'Analyse page');
}
async function ready(page) {
  let s = await status(page);
  if (s.status !== 'completed' || s.camera_id !== 'rec-6b7064605b3d') {
    if (s.camera_id && s.camera_id !== 'rec-6b7064605b3d' && s.status === 'processing') throw new Error('Another source is being analysed');
    if (s.status !== 'processing' && s.status !== 'connecting') {
      await page.getByRole('combobox',{name:'Recording',exact:true}).selectOption({label:'Mumbai street traffic (4K, 31 seconds)'});
      await wait(300);
      await click(page,page.getByRole('button',{name:/^(Start analysis|Analyse again)$/}),'Start recorded analysis');
    }
    const deadline=Date.now()+900000; let last=-1;
    while (Date.now()<deadline) {
      s=await status(page);
      if (s.status==='completed') break;
      if (s.status==='unavailable') throw new Error('Analysis unavailable: '+(s.failure_code||'unknown'));
      if (s.analysed_frames!==last && s.analysed_frames%10===0) {last=s.analysed_frames;console.log(JSON.stringify({stage:'analysing',samples:last,position:s.video_position_ms,device:s.inference_device}));}
      await wait(1000);
    }
  }
  assert.equal(s.status,'completed','Complete recording required before filming');
  assert.equal(s.camera_id,'rec-6b7064605b3d');
  await visible(page.getByRole('button',{name:'Replay at 1×',exact:true}),'Replay at 1×');
  return s;
}
async function replay(page) {
  await click(page,page.getByRole('button',{name:'Replay at 1×',exact:true}),'Replay at 1×');
  await visible(page.getByRole('button',{name:'Pause replay',exact:true}),'Loaded replay controls');
}
async function seek(page, key) {const slider=page.getByRole('slider',{name:'Replay source position'});await slider.focus();await slider.press(key);await wait(300);return Number(await slider.inputValue());}

(async()=>{
  const browser=await chromium.launch({channel:'chrome',headless:true});
  const context=await browser.newContext({viewport:{width:1280,height:720}});
  const page=await context.newPage(); page.on('dialog',d=>d.dismiss());
  const networkFailures=[];
  page.on('response', response=>{if(response.status()>=400&&response.url().includes('/v1/'))networkFailures.push({status:response.status(),path:new URL(response.url()).pathname});});
  let recordingContext;
  try {
    await enter(page); const before=await ready(page);
    const controls=await page.evaluate(()=>Array.from(document.querySelectorAll('button,input,select')).filter(e=>!e.closest('.reading-list')).map(e=>({tag:e.tagName,type:e.type,label:e.getAttribute('aria-label')||e.textContent?.trim().slice(0,70)})));
    fs.writeFileSync(path.join(out,'discovery.json'),JSON.stringify({viewport:[1280,720],controls},null,2));
    const mutations=[]; const monitor=request=>{if(request.method()==='POST'&&/\/v1\/recognition\/(session|control)(?:\?|$)/.test(request.url()))mutations.push(request.url().split('?')[0]);};
    page.on('request',monitor);
    await replay(page); await click(page,page.getByRole('button',{name:'Pause replay',exact:true}),'Pause replay');
    const slider=page.getByRole('slider',{name:'Replay source position'});
    const paused=Number(await slider.inputValue()); await wait(600); assert.equal(Number(await slider.inputValue()),paused,'Paused clock moved');
    await seek(page,'Home'); assert.equal(Number(await slider.inputValue()),0);
    const early=await page.locator('.reading-row').count(); assert(early>0 && early<before.vehicles_seen.length,'Historical rows should not contain the final inventory');
    await click(page,page.getByRole('button',{name:'Play replay',exact:true}),'Play at source speed');
    const t0=Number(await slider.inputValue()); const wall=Date.now(); await wait(1100);
    const delta=Number(await slider.inputValue())-t0; assert(Math.abs(delta-(Date.now()-wall))<350,'Replay is not 1× source time');
    await click(page,page.getByRole('button',{name:'Pause replay',exact:true}),'Pause again');
    const end=await seek(page,'End'); assert.equal(end,before.video_duration_ms);
    assert.equal(await page.locator('.reading-row').count(),before.vehicles_seen.length);
    await seek(page,'Home'); await page.screenshot({path:path.join(out,'rehearsal.png')});
    await click(page,page.getByRole('button',{name:'Exit replay',exact:true}),'Exit saved playback');
    const after=await status(page); assert.equal(after.session_id,before.session_id);assert.equal(after.analysed_frames,before.analysed_frames);
    assert.equal(mutations.length,0,'Playback attempted to restart or control inference'); page.off('request',monitor);
    const result={passed:true,session_id:before.session_id,device:before.inference_device,decoded_frames:before.decoded_frames,analysed_frames:before.analysed_frames,final_rows:before.vehicles_seen.length,early_rows:early,paused_clock_stable:true,scrubbing:true,no_future_inventory:true,source_time_clock:true,inference_mutations:0};
    fs.writeFileSync(path.join(out,'rehearsal.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({stage:'rehearsal_passed',...result}));
    if(rehearseOnly)return;
    const destination=path.join(out,'sentinel-analysis-replay.webm');if(fs.existsSync(destination))throw new Error('Recording already exists; choose a fresh output directory');
    recordingContext=await browser.newContext({viewport:{width:1280,height:720},recordVideo:{dir:path.join(out,'raw'),size:{width:1280,height:720}}});
    const film=await recordingContext.newPage();await enter(film);await ready(film);
    await caption(film,'Step 1 — completed GPU analysis, ready to review');await wait(3000);
    await replay(film);await click(film,film.getByRole('button',{name:'Pause replay',exact:true}),'Pause for introduction');await seek(film,'Home');
    await caption(film,'Step 2 — sampled detections replayed at 1× video time');await wait(2500);
    await click(film,film.getByRole('button',{name:'Play replay',exact:true}),'Play normal-speed analysis');
    await film.evaluate(()=>window.scrollTo({top:document.querySelector('.analysis-workspace').getBoundingClientRect().top+window.scrollY-82,behavior:'smooth'}));
    await wait(700);await film.screenshot({path:path.join(out,'playback.png')});
    await wait(before.video_duration_ms+1200);
    await caption(film,'Step 3 — best readings retained; uncertain results stay labelled');await wait(3500);
    await click(film,film.getByRole('button',{name:'Exit replay',exact:true}),'Exit replay');
    const accepted=film.locator('.reading-row').filter({hasText:'Model-accepted'}).first();
    await click(film,accepted,'Inspect a model-accepted reading');await visible(film.locator('dialog[open]'),'Recognition Inspector');
    await caption(film,'Step 4 — inspect original crops and model evidence');await wait(5500);
    await caption(film,'Model acceptance is not independently verified accuracy');await wait(3000);
    await recordingContext.close();recordingContext=null;await film.video().saveAs(destination);
    console.log(JSON.stringify({stage:'recorded',path:destination,bytes:fs.statSync(destination).size}));
  } catch(error) {
    const diagnostic=await page.evaluate(()=>({headings:[...document.querySelectorAll('h1,h2')].map(e=>e.textContent),notices:[...document.querySelectorAll('.notice')].map(e=>e.textContent),controls:[...document.querySelectorAll('button')].filter(e=>!e.closest('.reading-list')).map(e=>e.textContent?.trim())})).catch(()=>({page_closed:true}));
    fs.writeFileSync(path.join(out,'failure.json'),JSON.stringify({message:error.message,networkFailures,diagnostic},null,2));
    await page.screenshot({path:path.join(out,'failure.png')}).catch(failure=>console.warn('Failure screenshot unavailable: '+failure.message));
    throw error;
  } finally {if(recordingContext)await recordingContext.close();await context.close();await browser.close();}
})().catch(error=>{console.error('REPLAY DEMO FAILED: '+error.message);process.exitCode=1;});
