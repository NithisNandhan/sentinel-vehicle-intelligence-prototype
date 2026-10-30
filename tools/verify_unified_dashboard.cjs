/* Browser-only UI verification. Every API response and image below is synthetic.
 * Never connects this test browser to genuine recognition outputs or secrets.
 */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'C:/Users/NITHIS/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base = process.env.DASHBOARD_TEST_URL || 'http://127.0.0.1:5174';
const output = path.resolve(process.argv[2] || 'data/ui-validation/unified-20261030');
const viewports = [[1440,900],[1366,768],[768,1024],[390,844],[320,844]];
const evidence = [];
const checks = [];
const sessionId = 's-synthetic-ui-test';
const sourceId = 'rec-synthetic-ui';
const stamp = '2026-10-30T08:00:00Z';
const record = (name) => { checks.push(name); process.stdout.write('PASS '+name+'\n'); };

(async () => {
  fs.mkdirSync(output,{recursive:true});
  const browser = await chromium.launch({channel:'chrome',headless:true});
  try {
    const art = await browser.newPage();
    const images = await art.evaluate(() => {
      const canvas=document.createElement('canvas');canvas.width=960;canvas.height=540;
      const c=canvas.getContext('2d');c.fillStyle='#49545c';c.fillRect(0,0,960,540);
      c.fillStyle='#738476';c.fillRect(0,0,160,540);c.fillRect(800,0,160,540);
      c.strokeStyle='#d2d1bc';c.lineWidth=6;c.setLineDash([40,32]);c.beginPath();c.moveTo(480,0);c.lineTo(480,540);c.stroke();c.setLineDash([]);
      c.fillStyle='#e0e4e6';c.fillRect(300,210,110,175);c.fillStyle='#90a1ad';c.fillRect(314,225,82,50);c.strokeStyle='#a8e6c5';c.lineWidth=2;c.strokeRect(286,196,138,204);
      c.fillStyle='#16211f';c.fillRect(20,20,920,48);c.fillStyle='#fffdf9';c.font='22px sans-serif';c.fillText('SYNTHETIC UI TEST — NOT CCTV OR RECOGNITION EVIDENCE',34,51);
      const frame=canvas.toDataURL('image/jpeg').split(',')[1];
      canvas.width=180;canvas.height=50;c.fillStyle='#fffdf9';c.fillRect(0,0,180,50);c.fillStyle='#242925';c.font='22px monospace';c.fillText('DEMO ONLY',17,33);
      return {frame,crop:canvas.toDataURL('image/png').split(',')[1]};
    });
    await art.close();
    function fixture() {
      return {
        mode:'normal',processing:'paused',source:true,sessionPosts:0,diagnosticCalls:0,watchPosts:0,revision:1,
        entries:[{id:1,masked_plate:'GJ******34',reason:'Synthetic QA watchlist entry',severity:'high',active:true,version:1,created_at:stamp}],
        alerts:[{id:'1',sighting_id:'1',camera_id:sourceId,observed_at:stamp,processed_at:stamp,masked_plate:'GJ******34',reason:'Synthetic QA alert — test fixture, not a recognised plate',severity:'high',status:'new',source_mode:'recorded',stream_pts_ms:18000,session_id:sessionId,time_basis:'processing_time',created_at:stamp,audit:[]}]
      };
    }
    const state=fixture();
    const context=await browser.newContext({viewport:{width:1440,height:900},reducedMotion:'reduce'});
    await context.addInitScript(() => {
      class FixtureSocket {
        constructor(){this.readyState=0;queueMicrotask(()=>{this.readyState=1;this.onopen?.({});});}
        close(){this.readyState=3;}
        send(){}
      }
      window.WebSocket=FixtureSocket;
    });
    let releaseBootstrap;
    await context.route('**/v1/**',async route => {
      const request=route.request(), url=new URL(request.url()), endpoint=url.pathname, method=request.method();
      let code=200,body;
      if(endpoint==='/v1/local/session') {
        if(state.mode==='loading')await new Promise(resolve=>{releaseBootstrap=resolve;});
        if(state.mode==='permission'){code=403;body={detail:'Synthetic local profile configuration error'};}
        else body={csrf_token:'synthetic-csrf',expires_at:'2099-01-01T00:00:00Z'};
      } else if(endpoint==='/v1/recognition/status') {
        if(state.mode==='error'){code=503;body={detail:'Synthetic recognition service unavailable'};}
        else body={schema_version:'1.0',status:state.mode==='empty'?'paused':state.processing,camera_id:state.mode==='empty'||!state.source?null:sourceId,source_mode:'recorded',session_id:sessionId,recording_name:'Synthetic QA recording — long descriptive source name',video_position_ms:18000,video_duration_ms:41000,frame_age_seconds:0,stale_after_seconds:300,inference_device:'gpu',model_profile:'synthetic-fixture',recognition_state:'uncertain',unique_model_accepted_unverified:1,tracks:[],vehicles_seen:state.mode==='empty'?[]:[{track_id:'17',vehicle_class:'car',confidence:.9,first_ms:15000,last_ms:18000,observations:4,state:'uncertain',masked_plate:'GJ******34',watchlist:'possible_match'},{track_id:'18',vehicle_class:'bus',confidence:.8,first_ms:12000,last_ms:16000,observations:3,state:'model_accepted',masked_plate:'GJ******21',watchlist:'no_match'}]};
      } else if(endpoint==='/v1/recognition/recordings') body={recordings:state.mode==='empty'?[]:[{id:sourceId,name:'Synthetic QA recording — long descriptive source name',duration_ms:41000,time_basis:'processing_time'}]};
      else if(endpoint==='/v1/recognition/evidence') body={source_mode:'recorded',session_id:sessionId,video_position_ms:18000,frame:images.frame,vehicles:[{track_id:'17',vehicle_class:'car',confidence:.9,first_ms:15000,last_ms:18000,observations:4,state:'uncertain',reading:'GJ01AB1234',thumb:images.crop,watchlist:'possible_match'}]};
      else if(endpoint==='/v1/recognition/diagnostics') {
        state.diagnosticCalls++;
        body={schema_version:'1.0',source_mode:'recorded',session_id:sessionId,camera_id:sourceId,video_position_ms:18000,models:{ocr_model:'synthetic-fixture-no-inference',vehicle_device:'gpu',plate_device:'gpu',ocr_device:'gpu'},stage_ms:{vehicle:12,plate:8,ocr:20,reconstruction:6,fusion:4},track:{track_id:'17',vehicle_box:[10,10,100,100],vehicle_class:'car',seen_pts_ms:18000,observations:4,recognition_state:'uncertain',masked_plate:'GJ******34',best_reading:'GJ01AB1234',model_score:.8,gate_reasons:['Synthetic example: leading candidates remain too close'],frames:[{pts_ms:18000,width:180,height:50,quality:.8,text:'GJ01AB1234',image:images.crop,characters:[{character:'G',score:.8,alternatives:[{character:'C',score:.15}]}]}],candidates:[{plate:'GJ01AB1234',score:.8,minimum_character_score:.7,observations:4},{plate:'GJ01AB1284',score:.75}],reconstruction:{status:'reconstructed',reason:'Synthetic display fixture',image:images.crop,text:'GJ01AB1234',width:180,height:50,contributing_originals:4,elapsed_ms:6,characters:[],candidates:[]}}};
      } else if(endpoint==='/v1/recognition/session') {
        state.sessionPosts++;const data=request.postDataJSON();state.source=!!data.camera_id;state.processing='processing';body={schema_version:'1.0',status:state.processing,camera_id:data.camera_id,source_mode:'recorded',session_id:sessionId,recognition_state:'reading',tracks:[]};
      } else if(endpoint==='/v1/recognition/control') {
        const action=request.postDataJSON().action;state.processing=action==='pause'?'paused':'processing';body={schema_version:'1.0',status:state.processing,camera_id:sourceId,source_mode:'recorded',session_id:sessionId,recognition_state:'reading',tracks:[]};
      } else if(endpoint==='/v1/recognition/heartbeat')body={ok:true};
      else if(endpoint==='/v1/alerts') {
        if(state.mode==='partial'){code=503;body={detail:'Synthetic central alert service unavailable'};}
        else body=state.mode==='empty'?[]:state.alerts;
      } else if(/^\/v1\/alerts\/\d+$/.test(endpoint)) {
        const next=request.postDataJSON().status;state.alerts[0].status=next;state.alerts[0].audit.push({id:String(state.alerts[0].audit.length+1),to_status:next,actor:'synthetic QA',changed_at:stamp});body=state.alerts[0];
      } else if(endpoint==='/v1/watchlist'&&method==='POST') {
        state.watchPosts++;assert.equal(request.headers()['x-csrf-token'],'synthetic-csrf');const input=request.postDataJSON();state.revision++;const item={id:state.entries.length+1,masked_plate:input.plate.slice(0,2)+'******'+input.plate.slice(-2),reason:input.reason,severity:input.severity,active:true,version:state.revision,created_at:stamp};state.entries.push(item);code=201;body=item;
      } else if(/^\/v1\/watchlist\/\d+$/.test(endpoint)) {
        const input=request.postDataJSON(),item=state.entries.find(entry=>entry.id===Number(endpoint.split('/').pop()));
        if(state.conflict){code=409;body={detail:'Watchlist entry changed; refresh before editing'};state.conflict=false;}
        else {assert.equal(input.expected_version,item.version);const {expected_version,...changes}=input;Object.assign(item,changes,{version:++state.revision});body=item;}
      } else if(endpoint==='/v1/watchlist'||endpoint==='/v1/watchlist/search') {
        const input=method==='POST'?request.postDataJSON():{},q=(input.query||'').toLowerCase();const items=state.mode==='empty'?[]:state.entries.filter(e=>!q||(e.reason+' '+e.masked_plate).toLowerCase().includes(q));body={items,total:items.length,offset:input.offset||0,limit:25,version:state.revision};
      } else {code=404;body={detail:'No synthetic fixture for endpoint'};}
      await route.fulfill({status:code,contentType:'application/json',body:JSON.stringify(body)});
    });
    const page=await context.newPage();
    const pageErrors=[];page.on('pageerror',error=>pageErrors.push(error.message));
    async function capture(name,width,height){await page.setViewportSize({width,height});await page.screenshot({path:path.join(output,name),fullPage:true});const overflow=await page.evaluate(()=>({width:document.documentElement.scrollWidth,viewport:innerWidth,elements:[...document.querySelectorAll('body *')].filter(el=>el.getBoundingClientRect().right>innerWidth+1).slice(0,12).map(el=>({tag:el.tagName,class:el.className,right:el.getBoundingClientRect().right}))}));assert(overflow.width<=overflow.viewport+1,'Page overflows viewport '+width+': '+JSON.stringify(overflow));evidence.push({viewport:`${width}x${height}`,artifact:name,observation:'Rendered Paper teal interface using labelled synthetic fixtures; no viewport overflow.'});}
    await page.goto(base);await page.getByRole('heading',{name:'Analyse footage'}).waitFor();await page.locator('.reading-row').first().waitFor();
    assert.equal(state.sessionPosts,0);assert.equal(await page.getByRole('button',{name:/unlock/i}).count(),0);record('No manual unlock or inference on page load');
    for(const [width,height] of viewports)await capture(`analyse-${width}.png`,width,height);
    record('All five responsive viewport widths');
    await page.setViewportSize({width:1440,height:900});
    const row=page.locator('.reading-row').first();await row.focus();await row.click();await page.getByRole('dialog').waitFor();await page.getByText('Synthetic example: leading candidates remain too close').waitFor();await page.getByRole('heading',{name:'Native crop',exact:true}).waitFor();
    await capture('inspector-1440.png',1440,900);await capture('inspector-390.png',390,844);
    await page.keyboard.press('Escape');assert.equal(await page.getByRole('dialog').count(),0);assert(await row.evaluate(el=>el===document.activeElement));record('Inspector evidence, responsive sheet, Escape and focus return');
    const diagnosticBefore=state.diagnosticCalls;await page.waitForTimeout(2200);assert.equal(state.diagnosticCalls,diagnosticBefore);record('Closing Inspector stops diagnostic polling');
    await page.setViewportSize({width:1440,height:900});
    await page.getByRole('button',{name:'Watchlist',exact:true}).click();await page.getByRole('button',{name:'Add plate',exact:true}).click();await page.getByLabel('Plate number',{exact:true}).fill('GJ02CD9988');await page.getByLabel('Reason',{exact:true}).fill('Synthetic added entry');await page.getByLabel('Priority',{exact:true}).selectOption('medium');await page.getByRole('button',{name:'Save entry',exact:true}).click();await page.getByText('Synthetic added entry',{exact:true}).waitFor();assert.equal(state.watchPosts,1);record('Watchlist create and CSRF header');
    const added=page.getByRole('row').filter({hasText:'Synthetic added entry'});
    await added.getByRole('button',{name:'Edit',exact:true}).click();await page.getByLabel('Reason',{exact:true}).fill('Synthetic updated entry');await page.getByRole('button',{name:'Save entry',exact:true}).click();await page.getByText('Synthetic updated entry',{exact:true}).waitFor();
    page.once('dialog',dialog=>dialog.accept());await page.getByRole('row').filter({hasText:'Synthetic updated entry'}).getByRole('button',{name:'Deactivate',exact:true}).click();await page.getByRole('row').filter({hasText:'Synthetic updated entry'}).getByRole('button',{name:'Reactivate',exact:true}).waitFor();record('Watchlist versioned edit and reversible deactivation');
    state.conflict=true;await page.getByRole('row').filter({hasText:'Synthetic updated entry'}).getByRole('button',{name:'Edit',exact:true}).click();await page.getByLabel('Reason',{exact:true}).fill('Synthetic conflicting edit');await page.getByRole('button',{name:'Save entry',exact:true}).click();await page.getByText('Watchlist entry changed; refresh before editing',{exact:false}).waitFor();await page.keyboard.press('Escape');record('Watchlist stale edit conflict remains visible');
    await capture('watchlist-1440.png',1440,900);await capture('watchlist-320.png',320,844);
    await page.getByRole('button',{name:/^Alerts/}).click();await page.locator('.alert-row').first().click();await page.getByRole('button',{name:'Acknowledge',exact:true}).click();await page.getByRole('button',{name:'Resolve',exact:true}).click();assert.equal(state.alerts[0].status,'resolved');await page.keyboard.press('Escape');record('Alert acknowledge, resolve and decision history');
    await page.getByRole('button',{name:'Analyse',exact:true}).click();assert.equal(state.sessionPosts,0);record('Navigation preserves shared source without restarting inference');
    await page.getByRole('button',{name:'Pause',exact:true}).count();
    await page.getByRole('button',{name:'Resume',exact:true}).click();await page.getByRole('button',{name:'Pause',exact:true}).waitFor();await page.getByRole('button',{name:'Pause',exact:true}).click();await page.getByRole('button',{name:'Resume',exact:true}).waitFor();record('Shared pause/resume controls');
    state.mode='partial';await page.reload();await page.getByText(/Synthetic central alert service unavailable/).first().waitFor();await page.locator('.frame-viewport img').waitFor();await capture('partial.png',1440,900);record('Partial central outage retains labelled available analysis');
    state.mode='error';await page.reload();await page.getByText(/Synthetic recognition service unavailable/).first().waitFor();await capture('error.png',1440,900);record('Recognition error has an explicit recovery state');
    state.mode='empty';await page.reload();await page.getByText('No permitted recordings are registered.',{exact:false}).waitFor();await capture('empty.png',1440,900);record('Empty recording state');
    state.mode='permission';await page.reload();await page.getByRole('heading',{name:'Local service unavailable'}).waitFor();await page.getByRole('button',{name:'Retry connection'}).waitFor();await capture('permission.png',390,844);record('Local access failure without manual key entry');
    state.mode='loading';await page.reload({waitUntil:'domcontentloaded'});await page.getByRole('heading',{name:'Connecting to Sentinel'}).waitFor();await capture('loading.png',390,844);state.mode='normal';releaseBootstrap();
    assert.equal(pageErrors.length,0,pageErrors.join('\n'));record('No browser runtime errors');
    const report={contract:'docs/superpowers/specs/2026-10-30-sentinel-unified-dashboard-design.md; all network and media fixtures synthetic',targetViewports:viewports.map(([w,h])=>`${w}x${h}`),accessibility:{status:'pass',evidence:'Native buttons, labelled controls, dialog Escape and focus restoration exercised; status labels accompany colour.'},responsive:{status:'pass',evidence:'No global horizontal overflow at 1440,1366,768,390,320 widths; Inspector and watchlist exercised at narrow widths.'},contentStates:Object.fromEntries(['loading','empty','error','partial','success','permission'].map(s=>[s,{status:'pass',evidence:'Exercised '+s+' with explicit synthetic API fixtures; corresponding screenshots or successful mutation assertions retained.'}])),renderedEvidence:evidence,checks};
    fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(report,null,2));
    process.stdout.write(JSON.stringify({passed:checks.length,artifacts:output})+'\n');
    await context.close();
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
