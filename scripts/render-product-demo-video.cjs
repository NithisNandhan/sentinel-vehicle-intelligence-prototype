const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const workDirectory = path.resolve(process.argv[2]);
const outputDirectory = path.resolve(process.argv[3]);
const videoPath = path.join(outputDirectory, 'Sentinel_Vehicle_Intelligence_Demo.webm');
const captionsPath = path.join(outputDirectory, 'Sentinel_Vehicle_Intelligence_Demo.vtt');
const scenes = JSON.parse(fs.readFileSync(path.join(workDirectory, 'scenes.json'), 'utf8').replace(/^\uFEFF/, ''));

function clock(seconds) {
  const ms = Math.round(seconds * 1000);
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  return [h, m, s].map((n) => String(n).padStart(2, '0')).join(':') + '.' + String(ms % 1000).padStart(3, '0');
}

function addWebmDuration(source, durationMilliseconds) {
  const infoId = Buffer.from([0x15, 0x49, 0xa9, 0x66]);
  const infoAt = source.indexOf(infoId);
  if (infoAt < 0) throw new Error('The recorded WebM is missing its Segment Info element.');
  const sizeAt = infoAt + infoId.length;
  const first = source[sizeAt];
  let sizeLength = 1;
  let marker = 0x80;
  while (sizeLength <= 8 && (first & marker) === 0) { marker >>= 1; sizeLength += 1; }
  if (sizeLength > 8) throw new Error('The Segment Info size is not a valid EBML value.');
  let infoSize = BigInt(first & (marker - 1));
  for (let i = 1; i < sizeLength; i += 1) infoSize = (infoSize << 8n) | BigInt(source[sizeAt + i]);
  const contentAt = sizeAt + sizeLength;
  const endAt = contentAt + Number(infoSize);
  const durationId = Buffer.from([0x44, 0x89]);
  if (source.subarray(contentAt, endAt).indexOf(durationId) >= 0) return source;

  const durationElement = Buffer.alloc(11);
  durationElement.set(durationId, 0);
  durationElement[2] = 0x88;
  durationElement.writeDoubleBE(durationMilliseconds, 3);
  const nextInfoSize = infoSize + BigInt(durationElement.length);
  let newSizeLength = sizeLength;
  while (nextInfoSize >= (1n << BigInt(7 * newSizeLength)) - 1n) newSizeLength += 1;
  if (newSizeLength > 8) throw new Error('Updated Segment Info is too large.');
  const encodedSize = Buffer.alloc(newSizeLength);
  let remaining = nextInfoSize;
  for (let i = newSizeLength - 1; i >= 0; i -= 1) {
    encodedSize[i] = Number(remaining & 0xffn);
    remaining >>= 8n;
  }
  encodedSize[0] |= 0x80 >> (newSizeLength - 1);
  const prefix = source.subarray(0, sizeAt);
  const content = source.subarray(contentAt, endAt);
  const suffix = source.subarray(endAt);
  return Buffer.concat([prefix, encodedSize, content, durationElement, suffix]);
}

const timeline = [];
let totalDuration = 0;
const sceneDuration = 9.5;
scenes.forEach((scene) => {
  timeline.push({
    id: scene.id, title: scene.title, caption: scene.voiceover,
    start: totalDuration, speechDuration: sceneDuration - 0.5, duration: sceneDuration,
  });
  totalDuration += sceneDuration;
});
const vtt = ['WEBVTT', ''];
timeline.forEach((scene) => vtt.push(
  clock(scene.start) + ' --> ' + clock(scene.start + scene.speechDuration + 0.72),
  scene.caption,
  '',
));
fs.writeFileSync(captionsPath, vtt.join('\n'), 'utf8');

function createPainter() {
  const W = 1280, H = 720;
  const accent = '#52dbb4', blue = '#58a7ff', gold = '#f3c76c';
  const ink = '#f0f5ff', muted = '#a3b1c8', panel = '#111e31';
  function rr(c, x, y, w, h, r, fill, stroke) {
    c.beginPath(); c.roundRect(x, y, w, h, r);
    if (fill) { c.fillStyle = fill; c.fill(); }
    if (stroke) { c.strokeStyle = stroke; c.lineWidth = 1; c.stroke(); }
  }
  function tx(c, value, x, y, size, color, weight = 500, align = 'left') {
    c.fillStyle = color; c.font = weight + ' ' + size + 'px "Segoe UI", Arial, sans-serif'; c.textAlign = align; c.fillText(value, x, y); c.textAlign = 'left';
  }
  function wrap(c, value, x, y, max, step, lines, font, color) {
    c.font = font; c.fillStyle = color; let line = ''; let count = 0;
    for (const word of value.split(/\s+/)) {
      const next = line ? line + ' ' + word : word;
      if (c.measureText(next).width > max && line) { c.fillText(line, x, y + count * step); count++; line = word; if (count >= lines) break; }
      else line = next;
    }
    if (line && count < lines) c.fillText(line, x, y + count * step);
  }
  function card(c, x, y, w, h, tag, title, detail, color, fill = panel) {
    rr(c, x, y, w, h, 14, fill, '#263a55');
    c.fillStyle = color; c.fillRect(x + 20, y + 20, 4, 24);
    tx(c, tag.toUpperCase(), x + 37, y + 37, 10, color, 700);
    tx(c, title, x + 21, y + 83, 18, ink, 650);
    wrap(c, detail, x + 21, y + 112, w - 42, 19, 3, '450 12px "Segoe UI", Arial, sans-serif', muted);
  }
  function arrow(c, x1, y1, x2, y2, color, p) {
    const ex = x1 + (x2 - x1) * p, ey = y1 + (y2 - y1) * p;
    c.strokeStyle = color; c.lineWidth = 2; c.beginPath(); c.moveTo(x1,y1); c.lineTo(ex,ey); c.stroke();
    if (p > .55) { c.fillStyle = color; c.beginPath(); c.moveTo(ex,ey); c.lineTo(ex-9,ey-5); c.lineTo(ex-9,ey+5); c.closePath(); c.fill(); }
  }
  function heading(c, section, title) {
    tx(c, section.toUpperCase(), 80, 124, 12, accent, 700);
    tx(c, title, 80, 172, 32, ink, 650);
    tx(c, 'Prototype workflow • scope and evidence are shown separately', 82, 199, 14, muted, 450);
  }
  function camera(c, x, y, color, label) {
    rr(c, x, y, 72, 56, 9, '#13233a', color);
    c.fillStyle = color; c.fillRect(x + 11, y + 12, 50, 3);
    c.beginPath(); c.arc(x + 36, y + 30, 10, 0, Math.PI * 2); c.strokeStyle = color; c.lineWidth = 2; c.stroke();
    tx(c, label, x + 36, y + 52, 9, '#d5e5f5', 600, 'center');
  }
  function mapDot(c, x, y, color, label) {
    c.beginPath(); c.arc(x,y,8,0,Math.PI*2); c.fillStyle = color; c.fill();
    c.beginPath(); c.arc(x,y,17,0,Math.PI*2); c.strokeStyle = color; c.globalAlpha = .42; c.stroke(); c.globalAlpha = 1;
    tx(c, label, x + 15, y - 8, 9, '#d4e1ef', 600);
  }
  function backdrop(c) {
    const bg = c.createLinearGradient(0,0,W,H); bg.addColorStop(0,'#08111f'); bg.addColorStop(.52,'#0c182b'); bg.addColorStop(1,'#11192b');
    c.fillStyle = bg; c.fillRect(0,0,W,H);
    const glow = c.createRadialGradient(1100,60,12,1100,60,530); glow.addColorStop(0,'rgba(44,121,169,.18)'); glow.addColorStop(1,'rgba(44,121,169,0)');
    c.fillStyle = glow; c.fillRect(600,0,680,520);
    c.save(); c.globalAlpha = .07; c.strokeStyle = '#7ca3c7'; c.lineWidth = 1;
    for (let x=0;x<=W;x+=80) { c.beginPath(); c.moveTo(x,0); c.lineTo(x,H); c.stroke(); }
    for (let y=0;y<=H;y+=80) { c.beginPath(); c.moveTo(0,y); c.lineTo(W,y); c.stroke(); }
    c.restore();
  }
  function topbar(c) {
    c.beginPath(); c.arc(72,63,24,0,Math.PI*2); c.fillStyle='#10233a'; c.fill(); c.strokeStyle=accent; c.lineWidth=2; c.stroke();
    tx(c,'SVI',72,68,14,accent,700,'center');
    tx(c,'SENTINEL VEHICLE INTELLIGENCE',111,58,18,ink,650);
    tx(c,'REGIONAL EDGE  /  CENTRAL COMMAND',112,79,11,muted,500);
    rr(c,948,42,264,34,17,'#10233a','#28415d');
    tx(c,'ILLUSTRATIVE  •  SYNTHETIC DEMO DATA',1080,64,11,'#c7d7e8',600,'center');
  }
  function opening(c, p) {
    tx(c,'A REGIONAL-TO-CENTRAL PROTOTYPE',82,188,12,accent,700);
    tx(c,'See the workflow.',80,263,49,ink,700);
    tx(c,'Inspect the evidence.',80,321,42,ink,650);
    wrap(c,'Vehicle detection, multi-frame plate analysis, and an operator command centre — demonstrated without overstating recognition results.',84,376,530,25,3,'450 18px "Segoe UI", Arial, sans-serif',muted);
    rr(c,84,482,378,44,22,'#102f3a','#2d786d'); tx(c,'CAPTIONED PRODUCT WALKTHROUGH',273,509,11,'#c8fff0',700,'center');
    const cx=914, cy=352;
    for(let r=1;r<=3;r++){c.beginPath();c.strokeStyle='rgba(82,219,180,'+(.45-r*.09)+')';c.arc(cx,cy,r*91+Math.sin(p*6.28)*4,0,Math.PI*2);c.stroke();}
    c.beginPath();c.strokeStyle='rgba(88,167,255,.55)';c.arc(cx,cy,272,-1.3+p*1.5,-.55+p*1.5);c.lineWidth=5;c.stroke();
    camera(c,cx-201,cy-61,blue,'CAM01'); camera(c,cx+120,cy-61,accent,'CAM17');
    c.beginPath();c.arc(cx,cy+70,43,0,Math.PI*2);c.fillStyle='#17394a';c.fill();c.strokeStyle=accent;c.lineWidth=2;c.stroke();
    tx(c,'EDGE',cx,cy+74,14,ink,700,'center'); arrow(c,cx-126,cy-8,cx-45,cy+37,blue,Math.min(1,p*2)); arrow(c,cx+126,cy-8,cx+45,cy+37,accent,Math.min(1,p*2));
    rr(c,800,497,228,39,'#12263b','#29435e'); tx(c,'CENTRAL COMMAND CENTRE',914,522,11,'#c6d6ea',650,'center');
    tx(c,'Selected cameras • local prototype',914,562,11,muted,500,'center');
  }
  function challenge(c,p) {
    heading(c,'The problem','Turn a stream into a reviewable event');
    card(c,78,240,314,245,'Input','Selected camera feeds','Live streams are decoded at the regional edge. Only configured cameras are analysed.',blue);
    card(c,483,240,314,245,'Analysis','Find and follow vehicles','Detection and tracking select vehicle regions for careful, multi-frame plate analysis.',accent);
    card(c,888,240,314,245,'Review','Keep a person in control','The operator receives a bounded alert workflow; a weak reading is not a confirmed identity.',gold);
    arrow(c,398,362,470,362,blue,Math.min(1,p*3)); arrow(c,803,362,875,362,accent,Math.min(1,p*3));
    rr(c,81,510,1118,47,'#0d2035','#213a54'); tx(c,'SCOPE CHECK',103,539,10,gold,700);
    tx(c,'This prototype demonstrates the workflow, not 80,000-camera capacity.',221,540,15,ink,550);
  }
  function architecture(c,p) {
    heading(c,'Architecture','Regional analysis. Central command.');
    card(c,78,241,318,264,'01 • EDGE','Camera + AI worker','RTSP/TCP or HLS • vehicle detection and tracking • multi-frame OCR • SQLite outbox',accent);
    card(c,481,241,318,264,'02 • CENTRAL','API + data service','Versioned events • watchlist matching • PostgreSQL/PostGIS • duplicate-safe ingest',blue);
    card(c,884,241,318,264,'03 • OPERATOR','Command dashboard','Camera health • alert queue • routes and reports • real-time updates',gold);
    arrow(c,403,369,466,369,accent,Math.min(1,p*3)); arrow(c,807,369,869,369,blue,Math.min(1,p*3));
    rr(c,79,522,1124,38,'#0d2035','#213a54');
    tx(c,'Two-camera prototype  •  synthetic offline demo  •  no laptop-scale capacity claim',641,547,13,muted,500,'center');
  }
  function recognition(c,p) {
    heading(c,'Recognition pipeline','Evidence accumulates across frames');
    const xs=[76,303,530,757,984];
    const items=[['01','Sample + track','Vehicle follows its own history.',blue],['02','Keep crop pixels','OCR sees the native crop.',accent],['03','Read each frame','OCR returns alternatives.',gold],['04','Fuse evidence','Distinct observations combine.',blue],['05','Gate the result','Uncertain stays unverified.',gold]];
    items.forEach((v,i)=>{card(c,xs[i],278,210,198,v[0],v[1],v[2],v[3]);if(i<4)arrow(c,xs[i]+212,370,xs[i]+225,370,v[3],Math.min(1,p*4));});
    for(let i=0;i<5;i++){const x=100+i*61;rr(c,x,506,43,31,7,'#172943','#344d6a');tx(c,'f'+(i+1),x+21,526,11,i<Math.max(1,Math.floor(p*5))?accent:muted,650,'center');}
    tx(c,'Illustrative sequence  •  no super-resolution claim  •  model scores are not calibrated probabilities',640,555,11,muted,450,'center');
  }
  function operator(c,p) {
    heading(c,'Operator experience','A single place to search, review, and act');
    rr(c,79,222,1121,338,'#0c1626','#2b3d57'); rr(c,79,222,168,338,'#101f34','#2b3d57');
    tx(c,'SENTINEL',102,255,12,ink,700); tx(c,'COMMAND CENTRE',102,274,10,muted,550);
    ['Overview','Cameras','Alerts','Routes','Reports'].forEach((label,i)=>{const y=311+i*37;if(i===0)rr(c,92,y-16,142,29,6,'#17354a','#285873');tx(c,label,111,y+3,12,i===0?accent:muted,i===0?650:500);});
    tx(c,'Overview',271,254,20,ink,650); tx(c,'DEMO VIEW  •  SYNTHETIC STATUS',271,274,9,gold,650);
    rr(c,271,290,591,246,9,'#101d30','#2a3d57'); tx(c,'CAMERA LOCATIONS  •  ILLUSTRATIVE MAP',291,317,10,muted,650);
    c.strokeStyle='#263b57';c.lineWidth=11;[[316,377,544,472],[491,327,622,509],[292,446,805,360],[324,518,845,421]].forEach((r)=>{c.beginPath();c.moveTo(r[0],r[1]);c.lineTo(r[2],r[3]);c.stroke();});
    c.strokeStyle='#37506e';c.lineWidth=1;for(let x=305;x<850;x+=58){c.beginPath();c.moveTo(x,335);c.lineTo(x+26,525);c.stroke();}
    mapDot(c,403,385,blue,'CAM01');mapDot(c,683,437,accent,'CAM17');
    rr(c,882,290,293,97,9,'#111f33','#2a3d57');tx(c,'SEARCH AUTHORISED CAMERAS',900,315,9,muted,650);rr(c,900,330,257,39,7,'#0b1626','#2e4968');tx(c,'ID or location…',916,355,12,'#8498b3',500);
    rr(c,882,401,293,135,9,'#111f33','#2a3d57');tx(c,'ALERT QUEUE',900,426,10,gold,700);tx(c,'Synthetic demo event',900,452,13,ink,650);tx(c,'Pending operator review',900,474,11,muted,450);
    rr(c,900,492,123,27,6,'#142f32','#286451');tx(c,'OPEN EVENT',961,510,9,'#bcf7df',700,'center');
  }
  function syntheticAlert(c,p) {
    heading(c,'Alert workflow','The operator remains in control');
    rr(c,80,230,625,313,14,'#111e31','#30445f');rr(c,106,254,213,31,16,'#3b2f1b','#805f26');
    tx(c,'SYNTHETIC WATCHLIST MATCH',212,275,10,'#ffe0a2',700,'center');tx(c,'Demo alert • not from a real plate read',108,324,22,ink,650);
    tx(c,'SOURCE',108,362,9,muted,700);tx(c,'Synthetic offline event',108,384,14,gold,600);
    tx(c,'VEHICLE IDENTITY',108,425,9,muted,700);tx(c,'Withheld in this presentation',108,448,13,muted,500);
    tx(c,'OPERATOR ACTION',108,487,9,muted,700);rr(c,107,501,157,30,6,'#17354a','#27617a');tx(c,'ACKNOWLEDGE',185,521,9,'#c7eaff',700,'center');
    rr(c,277,501,121,30,6,'#143a32','#2d705d');tx(c,'RESOLVE',337,521,9,'#c8ffe9',700,'center');tx(c,'Audited actions stay attached to the event.',425,520,10,muted,500);
    [['MATCHED','synthetic rule'],['ACKNOWLEDGED','operator review'],['RESOLVED','action recorded']].forEach((v,i)=>{
      const x=753,y=263+i*88,active=i<Math.max(1,Math.ceil(p*3));c.beginPath();c.arc(x+18,y+15,16,0,Math.PI*2);c.fillStyle=active?'#153a34':'#17263d';c.fill();c.strokeStyle=active?accent:'#39506c';c.stroke();
      tx(c,String(i+1),x+18,y+20,11,active?accent:muted,700,'center');tx(c,v[0],x+52,y+11,12,active?ink:muted,700);tx(c,v[1],x+52,y+33,11,muted,450);
      if(i<2){c.fillStyle='#34546e';c.fillRect(x+17,y+32,2,39);}
    });
    rr(c,749,535,411,29,6,'#291f17','#604823');tx(c,'Synthetic alert proves workflow only — not recognition accuracy.',954,554,10,'#f1d5a4',550,'center');
  }
  function engineering(c,p) {
    heading(c,'Engineering console','See the model evidence — including failure');
    rr(c,78,226,617,322,10,'#0d1828','#2b3e58');tx(c,'ILLUSTRATIVE FRAME  •  NOT LIVE FOOTAGE',98,251,9,gold,700);
    c.fillStyle='#23324a';c.fillRect(99,270,575,222);c.fillStyle='#1a2c40';c.fillRect(99,388,575,104);c.fillStyle='#40516a';c.fillRect(99,438,575,54);
    for(let i=0;i<7;i++){c.strokeStyle='#33465f';c.beginPath();c.moveTo(108+i*88,289);c.lineTo(145+i*70,405);c.stroke();}
    rr(c,272,345,267,101,17,'#24364d','#70dfb9');c.fillStyle='#344b63';c.beginPath();c.ellipse(406,382,94,31,0,0,Math.PI*2);c.fill();
    c.fillStyle='#53667e';c.beginPath();c.arc(344,424,14,0,Math.PI*2);c.arc(482,424,14,0,Math.PI*2);c.fill();
    c.strokeStyle=accent;c.lineWidth=2;c.strokeRect(258,331,295,128);tx(c,'VEHICLE TRACK',271,327,9,accent,700);
    c.strokeStyle=gold;c.strokeRect(381,388,66,19);tx(c,'POSSIBLE PLATE ROI',449,403,9,gold,600);
    rr(c,725,226,477,322,10,'#111e31','#2b3e58');tx(c,'FRAME-BY-FRAME OCR EVIDENCE',749,253,10,blue,700);
    [['FRAME A','Partial / unclear'],['FRAME B','Alternative characters'],['FRAME C','Conflicting evidence']].forEach((v,i)=>{
      const y=276+i*54;rr(c,748,y,425,43,8,'#0e192b','#243952');tx(c,v[0],765,y+17,9,muted,700);tx(c,v[1],865,y+17,11,ink,550);tx(c,'UNVERIFIED',1149,y+17,8,gold,700,'right');
      for(let j=0;j<4;j++)rr(c,865+j*39,y+24,31,12,4,'#273952');
    });
    rr(c,748,454,425,68,8,'#122a34','#28624e');tx(c,'FUSED STATUS',767,480,9,accent,700);tx(c,'Candidate available • acceptance gate not met',767,504,12,ink,550);
  }
  function privacy(c,p) {
    heading(c,'Reliability and privacy','Built to show what leaves the edge');
    card(c,82,238,505,280,'Resilience','Store, retry, deduplicate','SQLite outbox holds events during a central outage. Retries drain safely; event IDs prevent duplicate central records.',blue);
    card(c,693,238,505,280,'Privacy','Keep raw evidence in memory','Genuine images and plaintext plates are not persisted. Central records carry masked or keyed identifiers and operational metadata.',accent);
    rr(c,83,531,1113,34,'#0d2035','#213a54');tx(c,'No face recognition  •  no footage recording  •  synthetic imagery is labelled',640,554,13,'#c2d1e2',550,'center');
  }
  function evidence(c,p) {
    heading(c,'Evidence boundary','The workflow is ahead of OCR accuracy');
    const items=[
      [82,accent,'OBSERVED','Vehicle detection','Vehicle boxes appear in selected test footage. Coverage still varies by scene.'],
      [469,blue,'INTERMITTENT','Plate proposals','Some plate crops are produced; missed vehicles and small or blurred crops remain.'],
      [856,gold,'NOT VERIFIED','Correct live reading','No live plate reading has yet been independently checked as correct against ground truth.'],
    ];
    items.forEach((v)=>{rr(c,v[0],248,342,223,14,panel,'#2a3d57');rr(c,v[0]+20,270,120,27,6,'#172943','#344d6a');tx(c,v[2],v[0]+80,288,9,v[1],700,'center');tx(c,v[3],v[0]+20,339,20,ink,650);wrap(c,v[4],v[0]+20,373,298,21,4,'450 13px "Segoe UI", Arial, sans-serif',muted);});
    rr(c,83,502,1110,53,8,'#2a2117','#66502e');tx(c,'No benchmark number here is a promise. Recognition needs labelled, independent evaluation before accuracy claims.',638,533,14,'#f2ddb6',550,'center');
  }
  function roadmap(c,p) {
    heading(c,'Future scope','Prove recognition before claiming scale');
    card(c,82,250,344,232,'NEXT 01','Build ground truth','Collect permitted, labelled Indian-plate frames across cameras, distance, angle, lighting, and motion.',blue);
    card(c,468,250,344,232,'NEXT 02','Measure every stage','Separate missed vehicles, missed plate crops, OCR ambiguity, and temporal-fusion outcomes.',accent);
    card(c,854,250,344,232,'NEXT 03','Expand after proof','Improve image conditions and review gates; validate load and privacy before wider deployment.',gold);
    rr(c,83,514,1112,48,'#102d35','#2d6b5e');tx(c,'A working workflow prototype. Live plate recognition remains the next proof point.',639,545,17,'#dcfff2',600,'center');
  }
  function draw(context, elapsed, timeline, total) {
    let index=timeline.findIndex((scene)=>elapsed<scene.start+scene.duration);if(index<0)index=timeline.length-1;
    const scene=timeline[index];const p=Math.max(0,Math.min(1,(elapsed-scene.start)/scene.duration));
    backdrop(context);topbar(context);
    const render={opening,challenge,architecture,recognition,operator,'synthetic-alert':syntheticAlert,engineering,privacy,evidence,roadmap}[scene.id];
    render(context,p);
    rr(context,64,593,1152,91,8,'rgba(8,17,31,.94)','#24374f');
    tx(context,'CAPTION',89,617,10,accent,700);
    wrap(context,scene.caption,89,643,1062,22,2,'500 15px "Segoe UI", Arial, sans-serif',ink);
    tx(context,'VISUALS ARE EXPLANATORY RECREATIONS • NO REAL PLATE TEXT OR CAMERA FOOTAGE',80,574,10,'#7d8fa8',500);
    tx(context,String(index+1).padStart(2,'0')+' / '+String(timeline.length).padStart(2,'0'),1200,574,10,'#7d8fa8',500,'right');
    context.fillStyle='#425873';context.fillRect(64,700,1152,3);context.fillStyle=accent;context.fillRect(64,700,1152*Math.max(0,Math.min(1,elapsed/total)),3);
  }
  return draw;
}

async function main() {
  const edgePath = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
  if (!fs.existsSync(edgePath)) throw new Error('Microsoft Edge could not be found.');
  const browser = await chromium.launch({ headless: true, executablePath: edgePath, args: ['--autoplay-policy=no-user-gesture-required'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
    await page.setContent('<!doctype html><html><head><style>html,body{margin:0;background:#08111f;overflow:hidden}canvas{display:block;width:1280px;height:720px}</style></head><body><canvas id="screen" width="1280" height="720"></canvas></body></html>');
    const painterSource = createPainter.toString();
    await page.evaluate(({ painterSource, timeline, totalDuration }) => {
      const painter = new Function('return (' + painterSource + ')()')();
      const canvas = document.getElementById('screen');
      painter(canvas.getContext('2d'), 0, timeline, totalDuration);
    }, { painterSource, timeline, totalDuration });
    await page.screenshot({ path: path.join(workDirectory, 'video-first-frame.png') });
    const captured = await page.evaluate(async ({ painterSource, timeline, totalDuration }) => {
      const painter = new Function('return (' + painterSource + ')()')();
      const canvas = document.getElementById('screen');
      const context = canvas.getContext('2d');
      const stream = canvas.captureStream(24);
      const type = 'video/webm;codecs=vp9';
      if (!MediaRecorder.isTypeSupported(type)) throw new Error('WebM video encoding is not supported by this browser.');
      const recorder = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 2700000 });
      const chunks = [];
      recorder.ondataavailable = (event) => { if (event.data && event.data.size) chunks.push(event.data); };
      return new Promise((resolve, reject) => {
        recorder.onerror = (event) => reject(event.error ? String(event.error) : 'Video recording failed.');
        recorder.onstop = async () => {
          try {
            const blob = new Blob(chunks, { type });
            const bytes = new Uint8Array(await blob.arrayBuffer());
            let binary = '';
            for (let offset = 0; offset < bytes.length; offset += 32768) binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
            resolve({ base64: btoa(binary), size: blob.size });
          } catch (error) { reject(String(error)); }
        };
        const startAt = performance.now();
        const draw = () => {
          const elapsed = Math.max(0, Math.min((performance.now() - startAt) / 1000, totalDuration));
          painter(context, elapsed, timeline, totalDuration);
          if (elapsed >= totalDuration) {
            recorder.requestData();
            recorder.stop();
            stream.getTracks().forEach((track) => track.stop());
            return;
          }
          requestAnimationFrame(draw);
        };
        recorder.start(1000);
        requestAnimationFrame(draw);
      });
    }, { painterSource, timeline, totalDuration });
    const webm = addWebmDuration(Buffer.from(captured.base64, 'base64'), totalDuration * 1000);
    fs.writeFileSync(videoPath, webm);
    console.log('Recorded ' + timeline.length + ' scenes, ' + totalDuration.toFixed(1) + ' seconds, ' + (captured.size / 1048576).toFixed(1) + ' MiB.');
    await page.close();
  } finally {
    await browser.close();
  }
  console.log('Video: ' + videoPath);
  console.log('Captions: ' + captionsPath);
}

const durationOnly = process.argv[4] === '--fix-duration-only';
const task = durationOnly
  ? Promise.resolve().then(() => {
      const existing = fs.readFileSync(videoPath);
      const updated = addWebmDuration(existing, totalDuration * 1000);
      fs.writeFileSync(videoPath, updated);
      console.log('Updated WebM duration metadata in ' + videoPath);
    })
  : main();
task.catch((error) => { console.error(error.stack || error); process.exitCode = 1; });
