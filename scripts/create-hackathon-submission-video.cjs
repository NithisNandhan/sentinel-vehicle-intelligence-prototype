const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const output = path.join(root, 'deliverables');
const videoPath = path.join(output, 'Sentinel_Hackathon_Final_Demo.mp4');
const captionsPath = path.join(output, 'Sentinel_Hackathon_Final_Demo.vtt');
const posterPath = path.join(output, 'Sentinel_Hackathon_Final_Demo_Poster.png');
const edgePath = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const envText = fs.readFileSync(path.join(root, '.env'), 'utf8');
const keyLine = envText.split(/\r?\n/).find((line) => /^OPERATOR_API_KEY=/.test(line));
const operatorKey = keyLine ? keyLine.slice(keyLine.indexOf('=') + 1).trim().replace(/^["']|["']$/g, '') : '';
if (!operatorKey) throw new Error('The local operator key is not configured; no dashboard capture was made.');
if (!fs.existsSync(edgePath)) throw new Error('Microsoft Edge could not be found.');
fs.mkdirSync(output, { recursive: true });

const smokeRender = process.argv.includes('--smoke-render');
const sceneSeconds = smokeRender ? 0.35 : 11.5;
const scenes = [
  { kind: 'hero', title: 'From camera feed to operator decision', section: 'SENTINEL VEHICLE INTELLIGENCE', caption: 'A guided tour of the operator command centre and the engineering console behind its vehicle and number-plate workflow.' },
  { kind: 'scope', title: 'A focused regional prototype', section: 'SCOPE', caption: 'This local prototype demonstrates two selected camera feeds and one shared analysis session. The 80,000-camera figure is an architectural target, not measured capacity.' },
  { kind: 'screen', shot: 'overview', section: 'OPERATOR DASHBOARD · OVERVIEW', caption: 'The command centre combines connection health, camera status, a map and the alert queue. The information shown is local prototype data, not a statewide production service.' },
  { kind: 'screen', shot: 'search', section: 'OPERATOR DASHBOARD · CAMERA SEARCH', caption: 'Search the authorised catalogue by camera ID, name or place. A preview is on-demand and does not add that camera to the AI workload.' },
  { kind: 'preview', title: 'Preview access is separate from analysis', section: 'CAMERA ACCESS', caption: 'Operators can open a temporary live preview when feed access permits. Preview sessions are bounded, are not recordings and do not change the selected analysis camera.' },
  { kind: 'screen', shot: 'cameras', section: 'OPERATOR DASHBOARD · CAMERA HEALTH', caption: 'The camera list shows source, health and last-seen time. “Offline” or “degraded” is a useful operational signal—not evidence about vehicle recognition.' },
  { kind: 'screen', shot: 'alerts', section: 'OPERATOR DASHBOARD · ALERT QUEUE', caption: 'Each watchlist item is visibly marked live or simulated. The selected example is synthetic demonstration data, not a real plate read from CCTV.' },
  { kind: 'actions', title: 'A person reviews every alert', section: 'OPERATOR WORKFLOW', caption: 'From the alert detail, an operator can acknowledge it, mark a false positive, open a preview or create an incident report. Decisions are recorded for review.' },
  { kind: 'screen', shot: 'routes', section: 'OPERATOR DASHBOARD · ROUTE SEARCH', caption: 'Route search orders available sightings by time to support an investigation. It only works when matching metadata exists; it does not establish identity by itself.' },
  { kind: 'screen', shot: 'reports', section: 'OPERATOR DASHBOARD · REPORTS', caption: 'Incident and periodic reports summarize camera, time and operator actions. They are supporting records—not independent confirmation that OCR was correct.' },
  { kind: 'architecture', title: 'Regional analysis feeds central command', section: 'SYSTEM FLOW', caption: 'At the edge, a worker analyses selected feeds and queues metadata durably. The central API deduplicates events, stores operational records and pushes updates to the dashboard.' },
  { kind: 'sampling', title: 'Decode continuously; analyse selectively', section: 'EDGE PROCESSING', caption: 'The decoder follows stream presentation timestamps. The analysis stage samples selected frames and tracks vehicles; it does not claim to run heavy inference on every frame.' },
  { kind: 'production', title: 'The production-worker OCR path', section: 'MODEL PIPELINE · PRODUCTION BASELINE', caption: 'The current production baseline combines YOLOv8n vehicle detection, ByteTrack, a YOLOv9-tiny plate detector, FastALPR CCT-S-v2 OCR and per-track temporal evidence.' },
  { kind: 'pixels', title: 'Localise first, preserve the original crop', section: 'MODEL PIPELINE · IMAGE HANDLING', caption: 'The plate detector proposes a region and the OCR reads the native crop before display annotations. Lossless cropping avoids extra compression; it cannot restore pixels the camera never captured.' },
  { kind: 'ocr', title: 'OCR returns evidence, not certainty', section: 'MODEL PIPELINE · PER-FRAME OCR', caption: 'Each crop produces character or sequence alternatives with model scores. Scores help rank hypotheses; they are not calibrated probabilities that a registration is correct.' },
  { kind: 'fusion', title: 'Combine distinct observations over time', section: 'MODEL PIPELINE · MULTI-FRAME FUSION', caption: 'Evidence is grouped by tracked vehicle and video time. Duplicate frames and processed copies do not count as new votes; scene cuts and broken tracks clear stale evidence.' },
  { kind: 'gates', title: 'Uncertain evidence stays uncertain', section: 'MODEL PIPELINE · DECISION GATE', caption: 'A candidate may remain partial, uncertain or unreadable. Even a model-accepted result is unverified until checked against independent ground truth.' },
  { kind: 'screen', shot: 'engineering', section: 'ENGINEERING CONSOLE · MODEL INSPECTION', caption: 'The engineering console exposes the selected profile, frame counts, vehicle boxes, native crops, per-frame OCR evidence, candidate ranking and timing—without claiming access to hidden model thoughts.' },
  { kind: 'profiles', title: 'The diagnostic stack is an experiment', section: 'MODEL PIPELINE · ENGINEERING PROFILE', caption: 'The separate engineering viewer uses a YOLO26s vehicle profile, ByteTrack, YOLOv9-tiny plate proposals and an adapted Awiros CTC reader. It is not the unchanged production worker.' },
  { kind: 'reconstruction', title: 'Reconstruction is a checked aid, not magic', section: 'MODEL PIPELINE · EXPERIMENTAL', caption: 'The diagnostic path can align distinct crops on a native-pixel grid and inspect reconstructed OCR suggestions. Those suggestions are checked against original-frame evidence; no super-resolution claim is made.' },
  { kind: 'benchmark', title: 'A small Indian-plate OCR pilot', section: 'MEASURED DEVELOPMENT EVIDENCE', caption: 'On one permitted Indian development passage across 21 controlled quality conditions, adapted Awiros accepted 11 correctly, made no wrong accepts and abstained 10 times. This is not a general accuracy rate.' },
  { kind: 'live', title: 'Live camera results remain the hard boundary', section: 'MEASURED LIVE EVIDENCE · 27 SEPTEMBER 2026', caption: 'A documented 90-second CAM17 check decoded 251 frames, analysed 20 and found 7 plate crops—but accepted zero identifiers. These counts do not measure recall or correctness.' },
  { kind: 'privacy', title: 'Privacy and resilience are part of the design', section: 'OPERATING SAFEGUARDS', caption: 'Genuine frames and plaintext plates are not stored in the central records. Metadata is buffered for retries; synthetic events are labelled; face recognition and recording are out of scope.' },
  { kind: 'future', title: 'What must be proven next', section: 'NEXT VALIDATION GATES', caption: 'Next: restore dependable feed availability, collect permitted labelled Indian-plate examples, measure missed vehicles and exact-string errors separately, and expand only after the evidence supports it.' },
];
scenes.forEach((scene, index) => {
  scene.start = index * sceneSeconds;
  scene.duration = sceneSeconds;
});
const totalDuration = scenes.length * sceneSeconds;

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
  if (infoAt < 0) throw new Error('Recorded WebM is missing its Segment Info element.');
  const sizeAt = infoAt + infoId.length;
  const first = source[sizeAt];
  let sizeLength = 1;
  let marker = 0x80;
  while (sizeLength <= 8 && (first & marker) === 0) { marker >>= 1; sizeLength += 1; }
  if (sizeLength > 8) throw new Error('Segment Info has an invalid EBML size.');
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
  for (let i = newSizeLength - 1; i >= 0; i -= 1) { encodedSize[i] = Number(remaining & 0xffn); remaining >>= 8n; }
  encodedSize[0] |= 0x80 >> (newSizeLength - 1);
  return Buffer.concat([source.subarray(0, sizeAt), encodedSize, source.subarray(contentAt, endAt), durationElement, source.subarray(endAt)]);
}

async function captureScreenshots(browser) {
  const shots = {};
  const context = await browser.newContext({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  await page.addInitScript((key) => {
    if (location.origin === 'http://127.0.0.1:5173') sessionStorage.setItem('sentinel.operatorKey', key);
  }, operatorKey);
  await page.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.getByText('Operator unlocked', { exact: true }).waitFor({ state: 'visible', timeout: 20000 });
  await page.getByRole('heading', { name: 'Command overview' }).waitFor({ state: 'visible', timeout: 15000 });
  await page.waitForTimeout(900);
  shots.overview = await page.screenshot({ type: 'png' });

  const query = page.locator('#catalogue-query');
  await query.scrollIntoViewIfNeeded();
  await query.fill('CAM17');
  await page.waitForTimeout(1100);
  shots.search = await page.screenshot({ type: 'png' });

  async function openSection(name, heading) {
    await page.getByRole('button', { name: new RegExp(name, 'i') }).click();
    await page.getByRole('heading', { name: new RegExp(heading, 'i') }).first().waitFor({ state: 'visible', timeout: 10000 });
    await page.waitForTimeout(500);
    return page.screenshot({ type: 'png' });
  }
  shots.alerts = await openSection('Alert queue', 'Watchlist alerts');
  shots.cameras = await openSection('Cameras', 'Monitored cameras');
  shots.routes = await openSection('Route search', 'Route');
  shots.reports = await openSection('Reports', 'Report');

  const engineering = await context.newPage();
  await engineering.goto('http://127.0.0.1:7780/', { waitUntil: 'domcontentloaded', timeout: 30000 });
  await engineering.locator('#operator-key').fill(operatorKey);
  await engineering.locator('#unlock').click();
  await engineering.getByText('Engineering evidence unlocked', { exact: true }).waitFor({ state: 'visible', timeout: 20000 });
  await engineering.getByRole('heading', { name: 'Inside the recognition pipeline' }).waitFor({ state: 'visible', timeout: 10000 });
  await engineering.waitForTimeout(900);
  shots.engineering = await engineering.screenshot({ type: 'png' });
  await context.close();
  return Object.fromEntries(Object.entries(shots).map(([name, image]) => [name, 'data:image/png;base64,' + image.toString('base64')]));
}

function painterFactory() {
  const W = 1280, H = 720;
  const colors = { mint: '#53e0bb', blue: '#69b6ff', gold: '#ffd27a', white: '#f4f7fc', muted: '#9daec4', panel: '#101d30', stroke: '#2c415c', bg: '#07111f' };
  function rounded(c, x, y, w, h, r, fill, stroke) {
    c.beginPath(); c.roundRect(x, y, w, h, r);
    if (fill) { c.fillStyle = fill; c.fill(); }
    if (stroke) { c.strokeStyle = stroke; c.lineWidth = 1; c.stroke(); }
  }
  function text(c, value, x, y, size = 14, color = colors.white, weight = 500, align = 'left') {
    c.fillStyle = color; c.font = weight + ' ' + size + 'px "Segoe UI", Arial, sans-serif'; c.textAlign = align; c.fillText(value, x, y); c.textAlign = 'left';
  }
  function wrap(c, value, x, y, maxWidth, lineHeight, maxLines, font, color) {
    c.font = font; c.fillStyle = color;
    let line = ''; let count = 0;
    for (const word of value.split(/\s+/)) {
      const next = line ? line + ' ' + word : word;
      if (c.measureText(next).width > maxWidth && line) {
        c.fillText(line, x, y + count * lineHeight); count += 1; line = word;
        if (count >= maxLines) break;
      } else line = next;
    }
    if (line && count < maxLines) c.fillText(line, x, y + count * lineHeight);
  }
  function background(c) {
    const g = c.createLinearGradient(0, 0, W, H); g.addColorStop(0, '#07111f'); g.addColorStop(.55, '#0b1829'); g.addColorStop(1, '#111d30');
    c.fillStyle = g; c.fillRect(0, 0, W, H);
    c.save(); c.globalAlpha = .07; c.strokeStyle = '#85a0be';
    for (let x = 0; x <= W; x += 80) { c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke(); }
    for (let y = 0; y <= H; y += 80) { c.beginPath(); c.moveTo(0, y); c.lineTo(W, y); c.stroke(); }
    c.restore();
    const glow = c.createRadialGradient(1100, 70, 10, 1100, 70, 480); glow.addColorStop(0, 'rgba(55,140,171,.17)'); glow.addColorStop(1, 'rgba(55,140,171,0)');
    c.fillStyle = glow; c.fillRect(620, 0, 660, 500);
  }
  function topbar(c, section) {
    rounded(c, 52, 34, 42, 42, 12, '#102b34', colors.mint);
    text(c, 'S', 73, 62, 21, colors.mint, 800, 'center');
    text(c, 'SENTINEL  /  VEHICLE INTELLIGENCE', 110, 52, 13, colors.white, 700);
    text(c, 'HACKATHON PROTOTYPE · LOCAL DEMONSTRATION', 110, 70, 9, colors.muted, 550);
    rounded(c, 866, 39, 360, 30, 15, '#102138', '#2b4665');
    text(c, section.toUpperCase(), 1046, 59, 10, colors.mint, 700, 'center');
  }
  function heading(c, title, subtitle) {
    text(c, title, 72, 153, 34, colors.white, 700);
    text(c, subtitle, 74, 182, 14, colors.muted, 450);
  }
  function card(c, x, y, w, h, tag, title, body, color = colors.mint) {
    rounded(c, x, y, w, h, 14, colors.panel, colors.stroke);
    c.fillStyle = color; c.fillRect(x + 18, y + 19, 4, 24);
    text(c, tag.toUpperCase(), x + 34, y + 36, 10, color, 750);
    text(c, title, x + 20, y + 79, 18, colors.white, 650);
    wrap(c, body, x + 20, y + 108, w - 40, 19, 4, '450 12px "Segoe UI", Arial, sans-serif', colors.muted);
  }
  function arrow(c, x1, y1, x2, y2, color, progress) {
    const x = x1 + (x2 - x1) * progress, y = y1 + (y2 - y1) * progress;
    c.strokeStyle = color; c.lineWidth = 2; c.beginPath(); c.moveTo(x1, y1); c.lineTo(x, y); c.stroke();
    if (progress > .5) { c.fillStyle = color; c.beginPath(); c.moveTo(x, y); c.lineTo(x - 9, y - 5); c.lineTo(x - 9, y + 5); c.closePath(); c.fill(); }
  }
  function caption(c, scene, index, count, elapsed, total) {
    rounded(c, 52, 579, 1176, 109, 10, 'rgba(5,13,24,.96)', '#2c415c');
    text(c, 'CAPTION', 76, 606, 10, colors.mint, 800);
    wrap(c, scene.caption, 76, 635, 1120, 22, 2, '550 15px "Segoe UI", Arial, sans-serif', colors.white);
    text(c, String(index + 1).padStart(2, '0') + ' / ' + String(count).padStart(2, '0'), 1200, 607, 10, colors.muted, 650, 'right');
    c.fillStyle = '#32465e'; c.fillRect(52, 704, 1176, 3);
    c.fillStyle = colors.mint; c.fillRect(52, 704, 1176 * Math.max(0, Math.min(1, elapsed / total)), 3);
  }
  function chip(c, label, x, y, color = colors.mint) {
    rounded(c, x, y, 225, 26, 13, '#122239', color);
    text(c, label, x + 112, y + 18, 9, color, 750, 'center');
  }
  function drawScreen(c, image, scene, p) {
    background(c);
    if (image) {
      c.save(); c.shadowColor = 'rgba(0,0,0,.55)'; c.shadowBlur = 26;
      rounded(c, 29, 17, 1222, 687, 12, '#0c1523', '#39536e');
      c.restore();
      c.save(); c.beginPath(); c.roundRect(36, 24, 1208, 673, 9); c.clip();
      c.drawImage(image, 36, 24, 1208, 673);
      c.restore();
      rounded(c, 52, 37, 300, 27, 13, 'rgba(7,17,31,.92)', '#39536e');
      text(c, 'REAL LOCAL UI CAPTURE · ' + scene.shot.toUpperCase(), 202, 55, 9, colors.mint, 750, 'center');
      if (scene.shot === 'engineering') {
        rounded(c, 827, 80, 380, 38, 8, 'rgba(54,37,17,.93)', '#a17a39');
        text(c, 'NO NEW LIVE FRAME IN THIS CAPTURE', 1017, 104, 11, colors.gold, 750, 'center');
      }
    } else background(c);
  }
  function drawHero(c, p) {
    background(c);
    text(c, 'A REGIONAL-TO-CENTRAL PROTOTYPE', 76, 175, 12, colors.mint, 800);
    text(c, 'See the workflow.', 73, 250, 48, colors.white, 750);
    text(c, 'Inspect the evidence.', 74, 307, 39, colors.white, 650);
    wrap(c, 'Operator decisions, vehicle detection and the OCR evidence chain—shown with careful labels for what is live, synthetic and still unproven.', 78, 354, 540, 25, 4, '450 17px "Segoe UI", Arial, sans-serif', colors.muted);
    rounded(c, 78, 477, 381, 42, 21, '#102e36', '#2f766a');
    text(c, 'CAPTIONED HACKATHON WALKTHROUGH', 269, 503, 11, '#d6fff3', 750, 'center');
    const cx = 920, cy = 358;
    for (let i = 1; i <= 3; i += 1) { c.beginPath(); c.strokeStyle = 'rgba(83,224,187,' + (.47 - i * .1) + ')'; c.arc(cx, cy, i * 70 + Math.sin(p * 6.28) * 4, 0, Math.PI * 2); c.stroke(); }
    const cam = (x, y, label, color) => { rounded(c, x, y, 84, 63, 10, '#12243a', color); c.strokeStyle = color; c.lineWidth = 2; c.beginPath(); c.arc(x + 42, y + 29, 12, 0, Math.PI * 2); c.stroke(); text(c, label, x + 42, y + 55, 10, colors.white, 700, 'center'); };
    cam(cx - 204, cy - 80, 'CAM17', colors.blue); cam(cx + 118, cy - 80, 'CAM01', colors.mint);
    rounded(c, cx - 72, cy + 38, 144, 43, 12, '#16333c', colors.mint); text(c, 'EDGE ANALYSIS', cx, cy + 65, 11, colors.mint, 750, 'center');
    arrow(c, cx - 104, cy - 16, cx - 58, cy + 42, colors.blue, Math.min(1, p * 2));
    arrow(c, cx + 104, cy - 16, cx + 58, cy + 42, colors.mint, Math.min(1, p * 2));
    rounded(c, cx - 126, cy + 108, 252, 38, 10, '#112339', '#33516e'); text(c, 'CENTRAL OPERATOR DASHBOARD', cx, cy + 133, 10, colors.white, 650, 'center');
  }
  function drawScope(c, p) {
    background(c); heading(c, 'A focused regional prototype', 'What this build demonstrates—and what it does not claim.');
    card(c, 76, 238, 345, 245, 'CURRENT DEMO', 'Two selected feeds', 'CAM17 and CAM01 are the configured test pair. One shared engineering session analyses one selected camera at a time.', colors.mint);
    card(c, 468, 238, 345, 245, 'FUTURE DESIGN TARGET', 'Regional federation', 'The architecture is intended to inform a much larger deployment. No 80,000-camera load or sustained capacity was measured here.', colors.blue);
    card(c, 860, 238, 345, 245, 'TRUST BOUNDARY', 'Decision support', 'A model suggestion is not a verified identity. Synthetic events and live model hypotheses stay visibly distinct.', colors.gold);
    rounded(c, 77, 511, 1127, 43, 8, '#102137', '#2d425d');
    text(c, 'PROTOTYPE ≠ PRODUCTION SCALE', 102, 539, 10, colors.gold, 800);
    text(c, 'Capacity and recognition accuracy require separate measured validation.', 382, 539, 13, colors.white, 550);
  }
  function drawPreview(c, p) {
    background(c); heading(c, 'Preview access is separate from analysis', 'The operator can inspect an authorised feed without expanding AI workload.');
    card(c, 77, 242, 330, 222, '01 · SEARCH', 'Find an authorised camera', 'Search the machine-readable camera catalogue by ID, name or place name.', colors.blue);
    card(c, 475, 242, 330, 222, '02 · PREVIEW', 'Open on demand', 'A temporary local relay serves a browser preview. Source URLs and credentials stay server-side.', colors.mint);
    card(c, 872, 242, 330, 222, '03 · CLEANUP', 'Close or time out', 'Closing stops the temporary preview; idle cleanup keeps resources bounded. Recording stays off.', colors.gold);
    arrow(c, 413, 350, 462, 350, colors.blue, Math.min(1, p * 3));
    arrow(c, 811, 350, 859, 350, colors.mint, Math.min(1, p * 3));
    rounded(c, 79, 496, 1123, 48, 9, '#10283a', '#2c5e69');
    text(c, 'A search or preview does not add the camera to the configured AI feeds.', 640, 526, 15, '#d8fff1', 650, 'center');
  }
  function drawActions(c, p) {
    background(c); heading(c, 'A person reviews every alert', 'No alert is an automatic accusation or enforcement action.');
    const rows = [
      ['NEW', 'Review source, time and camera', colors.blue],
      ['ACKNOWLEDGED', 'Record that an operator has reviewed it', colors.mint],
      ['RESOLVED / FALSE POSITIVE', 'Close with an auditable decision', colors.gold],
    ];
    rows.forEach((row, i) => {
      const y = 236 + i * 90;
      rounded(c, 85, y, 692, 70, 12, colors.panel, colors.stroke);
      rounded(c, 105, y + 21, 198, 27, 14, '#142940', row[2]);
      text(c, row[0], 204, y + 39, 9, row[2], 800, 'center');
      text(c, row[1], 330, y + 43, 14, colors.white, 600);
      if (i < 2) arrow(c, 440, y + 73, 440, y + 88, row[2], Math.min(1, Math.max(0, p * 3 - i)));
    });
    card(c, 824, 237, 368, 252, 'OPERATOR ACTIONS', 'Acknowledge · false positive · report', 'The detail pane carries event provenance and offers preview and incident-report actions. Historical example data in the queue is synthetic.', colors.gold);
    text(c, 'AUDITABLE HUMAN REVIEW', 640, 543, 12, colors.mint, 750, 'center');
  }
  function drawArchitecture(c, p) {
    background(c); heading(c, 'Regional analysis feeds central command', 'A small, durable event path joins local inference to operator action.');
    card(c, 78, 237, 320, 260, 'REGIONAL EDGE', 'Decode · detect · track', 'Selected streams are sampled by video time. OCR evidence stays bounded in the local worker.', colors.mint);
    card(c, 481, 237, 320, 260, 'RELIABLE DELIVERY', 'SQLite outbox · versioned API', 'Metadata waits safely during an outage. Unique event IDs make retries duplicate-safe.', colors.blue);
    card(c, 884, 237, 320, 260, 'CENTRAL COMMAND', 'Watchlist · alerts · reports', 'The central service stores operational records and pushes status to the operator view.', colors.gold);
    arrow(c, 405, 361, 470, 361, colors.mint, Math.min(1, p * 3));
    arrow(c, 808, 361, 873, 361, colors.blue, Math.min(1, p * 3));
    text(c, 'No media recording · no face recognition · no statewide scale claim', 640, 542, 12, colors.muted, 550, 'center');
  }
  function drawSampling(c, p) {
    background(c); heading(c, 'Decode continuously; analyse selectively', 'The pipeline follows stream timestamps instead of trusting a nominal frame rate.');
    rounded(c, 90, 252, 1100, 116, 14, colors.panel, colors.stroke);
    text(c, 'LIVE DECODE', 114, 281, 10, colors.blue, 800);
    for (let i = 0; i < 12; i += 1) {
      const x = 120 + i * 83;
      rounded(c, x, 300, 48, 38, 7, i % 3 === 0 ? '#164138' : '#17263a', i % 3 === 0 ? colors.mint : '#354b65');
      text(c, 'f' + String(i + 1).padStart(2, '0'), x + 24, 324, 9, i % 3 === 0 ? colors.mint : colors.muted, 650, 'center');
    }
    rounded(c, 90, 409, 1100, 92, 14, '#101d30', '#304863');
    text(c, 'SAMPLED FOR INFERENCE', 114, 438, 10, colors.gold, 800);
    [0, 3, 6, 9].forEach((n, i) => {
      const x = 120 + n * 83;
      rounded(c, x, 455, 48, 29, 6, '#42341a', colors.gold);
      text(c, 'PTS', x + 24, 474, 8, colors.gold, 750, 'center');
      if (i < 3) arrow(c, x + 53, 470, x + 234, 470, colors.gold, Math.min(1, Math.max(0, p * 4 - i)));
    });
    text(c, 'A detection sample is not a unique vehicle; one vehicle may appear in many sampled frames.', 640, 540, 12, colors.muted, 500, 'center');
  }
  function drawProduction(c, p) {
    background(c); heading(c, 'The production-worker OCR path', 'This is the baseline pipeline, not the separate experimental engineering profile.');
    const xs = [72, 314, 556, 798, 1040];
    const items = [
      ['01 · YOLOv8n', 'Vehicle detection', colors.blue],
      ['02 · ByteTrack', 'Vehicle track', colors.mint],
      ['03 · YOLOv9-tiny', 'Plate proposal', colors.gold],
      ['04 · FastALPR', 'CCT-S-v2 OCR', colors.blue],
      ['05 · Temporal', 'Evidence + gate', colors.mint],
    ];
    items.forEach((item, i) => {
      rounded(c, xs[i], 267, 174, 166, 13, '#101d30', colors.stroke);
      rounded(c, xs[i] + 17, 288, 140, 30, 8, '#142a40', item[2]);
      text(c, item[0], xs[i] + 87, 308, 9, item[2], 750, 'center');
      wrap(c, item[1], xs[i] + 15, 354, 144, 22, 2, '600 15px "Segoe UI", Arial, sans-serif', colors.white);
      if (i < 4) arrow(c, xs[i] + 177, 352, xs[i + 1] - 7, 352, item[2], Math.min(1, p * 4));
    });
    rounded(c, 79, 475, 1121, 57, 10, '#152339', '#31465f');
    text(c, 'ONE VEHICLE TRACK → DISTINCT PLATE OBSERVATIONS → A CANDIDATE OR AN ABSTENTION', 640, 509, 13, colors.white, 650, 'center');
  }
  function drawPixels(c, p) {
    background(c); heading(c, 'Localise first, preserve the original crop', 'Display enlargement is not additional camera detail.');
    rounded(c, 81, 231, 688, 310, 15, '#0c192a', '#304763');
    text(c, 'ILLUSTRATIVE FLOW · NOT A CAMERA FRAME', 103, 257, 9, colors.gold, 800);
    rounded(c, 124, 287, 563, 205, 12, '#16283c', '#3b536d');
    c.fillStyle = '#223b52'; c.fillRect(125, 385, 561, 106);
    c.strokeStyle = '#60778d'; c.lineWidth = 2;
    for (let i = 0; i < 5; i += 1) { c.beginPath(); c.moveTo(140 + i * 115, 298); c.lineTo(190 + i * 98, 404); c.stroke(); }
    rounded(c, 266, 344, 270, 120, 30, '#1a3a47', colors.mint);
    rounded(c, 319, 412, 96, 24, 3, '#384754', colors.gold);
    c.strokeStyle = colors.gold; c.lineWidth = 2; c.strokeRect(310, 403, 115, 42);
    text(c, 'NATIVE PLATE REGION', 368, 483, 9, colors.gold, 750, 'center');
    card(c, 813, 242, 386, 131, 'OCR INPUT', 'Original crop', 'Copied before drawing overlays or lossy display encoding.', colors.mint);
    card(c, 813, 395, 386, 131, 'IMPORTANT LIMIT', 'No new photons', 'Blurred, tiny or missing characters cannot be recovered with certainty by resizing.', colors.gold);
  }
  function drawOcr(c, p) {
    background(c); heading(c, 'OCR returns evidence, not certainty', 'A frame contributes alternatives and character-level scores to a track.');
    const bars = [
      ['CHARACTER POSITION 1', .88, .53, .31], ['CHARACTER POSITION 2', .61, .59, .22],
      ['CHARACTER POSITION 3', .74, .40, .19], ['CHARACTER POSITION 4', .48, .46, .25],
    ];
    bars.forEach((item, i) => {
      const y = 238 + i * 61;
      text(c, item[0], 96, y + 8, 9, colors.muted, 700);
      const vals = item.slice(1);
      vals.forEach((v, j) => {
        const x = 332 + j * 250;
        rounded(c, x, y - 4, 210, 23, 5, '#17263a', '#2f435c');
        rounded(c, x + 1, y - 3, 208 * v, 21, 4, j === 0 ? '#1c5949' : '#27364a', null);
      text(c, 'rank ' + (j + 1) + ' · illustrative', x + 10, y + 13, 9, j === 0 ? colors.mint : colors.muted, 550);
      });
    });
    rounded(c, 94, 503, 522, 62, 9, '#111e31', colors.stroke);
    text(c, 'RANKING SCORE ≠ CORRECTNESS PROBABILITY', 113, 529, 10, colors.gold, 750);
    text(c, 'Calibration needs labelled, representative validation.', 113, 549, 10, colors.muted, 500);
    rounded(c, 659, 503, 526, 62, 9, '#111e31', colors.stroke);
    text(c, 'CANDIDATES MAY REMAIN INCOMPLETE', 678, 529, 10, colors.blue, 750);
    text(c, 'Alternatives and abstentions remain visible for review.', 678, 549, 10, colors.muted, 500);
  }
  function drawFusion(c, p) {
    background(c); heading(c, 'Combine distinct observations over time', 'Temporal fusion can reduce one-frame ambiguity—but cannot create information.');
    const xs = [125, 367, 609, 851];
    xs.forEach((x, i) => {
      rounded(c, x, 258, 185, 114, 13, '#101d30', i === 3 ? colors.mint : colors.stroke);
      text(c, 'FRAME ' + String.fromCharCode(65 + i), x + 20, 286, 10, i === 3 ? colors.mint : colors.blue, 750);
      for (let j = 0; j < 6; j += 1) {
        const h = 22 + ((j * 19 + i * 13) % 49);
        rounded(c, x + 20 + j * 23, 347 - h, 14, h, 4, j === i ? colors.mint : '#31465d', null);
      }
      if (i < 3) arrow(c, x + 190, 315, x + 234, 315, colors.blue, Math.min(1, p * 4));
    });
    rounded(c, 183, 420, 916, 82, 13, '#14263b', '#34506b');
    text(c, 'TRACK ID + VIDEO PTS', 215, 450, 10, colors.blue, 800);
    text(c, 'align distinct crops', 440, 468, 13, colors.white, 550);
    arrow(c, 646, 460, 737, 460, colors.mint, Math.min(1, p * 2));
    text(c, 'fused candidate set', 770, 468, 13, colors.mint, 650);
    text(c, 'Repeat frames, copies and scene-cut duplicates do not add independent votes.', 640, 547, 12, colors.muted, 500, 'center');
  }
  function drawGates(c, p) {
    background(c); heading(c, 'Uncertain evidence stays uncertain', 'A disciplined decision gate protects downstream operators from invented certainty.');
    const stages = [
      ['INPUT', 'Distinct original-frame OCR', colors.blue], ['FUSION', 'Track-aligned evidence', colors.mint], ['GATE', 'Evidence thresholds', colors.gold],
    ];
    stages.forEach((stage, i) => {
      const x = 99 + i * 385;
      card(c, x, 255, 328, 144, stage[0], stage[1], i === 2 ? 'Model-specific ranking rules; not calibrated confidence.' : 'Use genuine observations and retain their source.', stage[2]);
      if (i < 2) arrow(c, x + 332, 327, x + 378, 327, stage[2], Math.min(1, p * 3));
    });
    rounded(c, 106, 449, 515, 65, 12, '#15273c', '#3a536e');
    text(c, 'PARTIAL · UNCERTAIN · NO READ', 363, 489, 15, colors.gold, 750, 'center');
    rounded(c, 659, 449, 515, 65, 12, '#123129', '#2b6756');
    text(c, 'MODEL-ACCEPTED · STILL UNVERIFIED', 916, 489, 15, colors.mint, 750, 'center');
    text(c, 'Only independently labelled truth can establish correctness.', 640, 548, 12, colors.muted, 500, 'center');
  }
  function drawProfiles(c, p) {
    background(c); heading(c, 'The diagnostic stack is an experiment', 'Similar names do not mean the same deployed path.');
    rounded(c, 83, 233, 534, 285, 14, '#111e31', '#334a65');
    rounded(c, 104, 253, 245, 29, 7, '#123b32', '#2f816d'); text(c, 'PRODUCTION WORKER', 226, 273, 10, colors.mint, 800, 'center');
    ['YOLOv8n → vehicles', 'ByteTrack → vehicle tracks', 'YOLOv9-tiny → plate proposals', 'FastALPR CCT-S-v2 → OCR'].forEach((v, i) => text(c, v, 120, 321 + i * 42, 13, colors.white, 550));
    rounded(c, 663, 233, 534, 285, 14, '#111e31', '#334a65');
    rounded(c, 684, 253, 245, 29, 7, '#302b19', '#96723a'); text(c, 'ENGINEERING VIEWER', 806, 273, 10, colors.gold, 800, 'center');
    ['YOLO26s → vehicles', 'ByteTrack → track histories', 'YOLOv9-tiny → plate proposals', 'Adapted Awiros CTC → OCR'].forEach((v, i) => text(c, v, 700, 321 + i * 42, 13, colors.white, 550));
    text(c, 'Keep diagnostic findings separate until they pass controlled deployment validation.', 640, 552, 12, colors.muted, 500, 'center');
  }
  function drawBenchmark(c, p) {
    background(c); heading(c, 'A small Indian-plate OCR pilot', 'Controlled degradation of one permitted development passage · not a live-camera test.');
    text(c, '21', 110, 290, 51, colors.white, 800); text(c, 'quality conditions', 111, 318, 12, colors.muted, 500);
    text(c, '1', 110, 383, 51, colors.gold, 800); text(c, 'distinct vehicle passage', 111, 411, 12, colors.muted, 500);
    const chartX = 400, chartW = 660, top = 253;
    const rows = [
      ['CCT-S v2 temporal', 0, 0, 21, colors.blue],
      ['Awiros CTC adapted', 11, 0, 10, colors.mint],
    ];
    rows.forEach((row, i) => {
      const y = top + i * 126;
      text(c, row[0], chartX, y + 10, 15, colors.white, 650);
      const parts = [['Correct accept', row[1], colors.mint], ['Wrong accept', row[2], '#eb7777'], ['Abstain', row[3], colors.gold]];
      parts.forEach((part, j) => {
        const yy = y + 28 + j * 27;
        text(c, part[0], chartX, yy + 13, 9, colors.muted, 550);
        rounded(c, chartX + 122, yy, chartW - 220, 18, 4, '#17273c', '#2b3f58');
        if (part[1]) rounded(c, chartX + 123, yy + 1, (chartW - 222) * part[1] / 21, 16, 3, part[2], null);
        text(c, part[1] + ' / 21', chartX + chartW - 65, yy + 13, 10, part[2], 700, 'right');
      });
    });
    rounded(c, 1010, 500, 187, 42, 8, '#33241d', '#815b35');
    text(c, 'NOT GENERAL ACCURACY', 1103, 526, 9, colors.gold, 800, 'center');
    text(c, 'Source: NIVU Indian License Plate Dataset · CC BY 4.0 · controlled copies of one passage', 640, 561, 9, colors.muted, 450, 'center');
  }
  function drawLive(c, p) {
    background(c); heading(c, 'Live camera results remain the hard boundary', 'Recorded 27 September 2026 · selected live feed · one 90-second observation window.');
    const metrics = [
      ['DECODED FRAMES', '251', colors.blue], ['ANALYSED FRAMES', '20', colors.mint], ['PLATE CROPS', '7', colors.gold], ['MODEL-ACCEPTED IDs', '0', '#ef9b85'],
    ];
    metrics.forEach((m, i) => {
      const x = 76 + i * 292;
      rounded(c, x, 272, 260, 161, 15, '#101d30', '#314861');
      text(c, m[0], x + 19, 305, 10, colors.muted, 750);
      text(c, m[1], x + 20, 379, 43, m[2], 800);
      c.fillStyle = m[2]; c.fillRect(x + 20, 401, 52 + p * 100, 4);
    });
    rounded(c, 83, 478, 1114, 65, 11, '#2b201a', '#74583a');
    text(c, 'CROP COUNT ≠ READABLE PLATES', 111, 506, 11, colors.gold, 800);
    text(c, 'No verified live plate read was established by this sample.', 111, 528, 13, colors.white, 550);
    text(c, 'This was a live test, not a labelled recall or accuracy evaluation.', 640, 560, 10, colors.muted, 500, 'center');
  }
  function drawPrivacy(c, p) {
    background(c); heading(c, 'Privacy and resilience are part of the design', 'The system must handle failure without turning it into false evidence.');
    card(c, 80, 237, 347, 252, 'DATA MINIMISATION', 'No retained live imagery', 'Genuine frames, crops and plaintext plates remain transient. Central records use masked/keyed identifiers and metadata.', colors.mint);
    card(c, 466, 237, 347, 252, 'OUTAGE RECOVERY', 'Queue, retry, deduplicate', 'The local SQLite outbox preserves events until the central API returns. Event IDs prevent duplicate records.', colors.blue);
    card(c, 852, 237, 347, 252, 'HUMAN REVIEW', 'Synthetic data is labelled', 'Demo alerts are explicitly marked. Face recognition, recording and automatic enforcement are out of scope.', colors.gold);
    text(c, 'The final video contains no genuine camera frame or readable plate.', 640, 544, 12, colors.muted, 500, 'center');
  }
  function drawFuture(c, p) {
    background(c); heading(c, 'What must be proven next', 'A credible path from prototype to an evidence-backed deployment proposal.');
    const steps = [
      ['01', 'Reconnect live feeds reliably', 'One-camera decode → sustained two-feed test', colors.blue],
      ['02', 'Create representative ground truth', 'Permitted Indian plates across distance, angle and light', colors.mint],
      ['03', 'Measure each failure stage', 'Vehicle recall · plate crop recall · exact-string precision', colors.gold],
    ];
    steps.forEach((s, i) => {
      const x = 82 + i * 388;
      rounded(c, x, 253, 345, 217, 15, colors.panel, colors.stroke);
      text(c, s[0], x + 22, 295, 13, s[3], 800);
      text(c, s[1], x + 22, 342, 18, colors.white, 650);
      wrap(c, s[2], x + 22, 380, 296, 21, 3, '450 13px "Segoe UI", Arial, sans-serif', colors.muted);
    });
    rounded(c, 83, 501, 1111, 53, 10, '#102f35', '#2b675c');
    text(c, 'A working system workflow. Live plate accuracy and deployment scale remain open validation gates.', 638, 533, 14, '#d8fff3', 650, 'center');
  }
  function draw(c, elapsed, timeline, total, images) {
    let index = timeline.findIndex((entry) => elapsed < entry.start + entry.duration);
    if (index < 0) index = timeline.length - 1;
    const scene = timeline[index];
    const p = Math.max(0, Math.min(1, (elapsed - scene.start) / scene.duration));
    if (scene.kind === 'screen') drawScreen(c, images[scene.shot], scene, p);
    else {
      if (scene.kind === 'hero') drawHero(c, p);
      else if (scene.kind === 'scope') drawScope(c, p);
      else if (scene.kind === 'preview') drawPreview(c, p);
      else if (scene.kind === 'actions') drawActions(c, p);
      else if (scene.kind === 'architecture') drawArchitecture(c, p);
      else if (scene.kind === 'sampling') drawSampling(c, p);
      else if (scene.kind === 'production') drawProduction(c, p);
      else if (scene.kind === 'pixels') drawPixels(c, p);
      else if (scene.kind === 'ocr') drawOcr(c, p);
      else if (scene.kind === 'fusion') drawFusion(c, p);
      else if (scene.kind === 'gates') drawGates(c, p);
      else if (scene.kind === 'profiles') drawProfiles(c, p);
      else if (scene.kind === 'benchmark') drawBenchmark(c, p);
      else if (scene.kind === 'live') drawLive(c, p);
      else if (scene.kind === 'privacy') drawPrivacy(c, p);
      else if (scene.kind === 'future') drawFuture(c, p);
      topbar(c, scene.section);
    }
    if (scene.kind !== 'screen') text(c, 'ILLUSTRATIVE DIAGRAM · NOT CAMERA FOOTAGE', 1208, 569, 8, colors.muted, 500, 'right');
    caption(c, scene, index, timeline.length, elapsed, total);
  }
  return draw;
}

async function main() {
  const browser = await chromium.launch({ headless: true, executablePath: edgePath, args: ['--autoplay-policy=no-user-gesture-required'] });
  const receivedChunks = [];
  let receivedBytes = 0;
  let renderedBuffer = null;
  const server = require('node:http').createServer((request, response) => {
    if (request.method === 'GET' && request.url === '/') {
      response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      response.end('<!doctype html><html><head><style>html,body{margin:0;background:#07111f;overflow:hidden}canvas{display:block;width:1280px;height:720px}</style></head><body><canvas id="screen" width="1280" height="720"></canvas></body></html>');
      return;
    }
    if (request.method === 'GET' && request.url === '/watch' && renderedBuffer) {
      response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      response.end('<!doctype html><html><body style="margin:0;background:#07111f"><video id="demo" controls preload="metadata" style="width:1280px;height:720px" src="/video"></video></body></html>');
      return;
    }
    if (request.method === 'GET' && request.url.startsWith('/video') && renderedBuffer) {
      const range = request.headers.range;
      if (range && /^bytes=\d+-\d*$/.test(range)) {
        const start = Number(range.slice(6).split('-')[0]);
        const requestedEnd = range.split('-')[1];
        const end = Math.min(requestedEnd ? Number(requestedEnd) : renderedBuffer.length - 1, renderedBuffer.length - 1);
        response.writeHead(206, { 'Content-Type': 'video/mp4', 'Accept-Ranges': 'bytes',
          'Content-Range': 'bytes ' + start + '-' + end + '/' + renderedBuffer.length, 'Content-Length': end - start + 1 });
        response.end(renderedBuffer.subarray(start, end + 1));
      } else {
        response.writeHead(200, { 'Content-Type': 'video/mp4', 'Accept-Ranges': 'bytes', 'Content-Length': renderedBuffer.length });
        response.end(renderedBuffer);
      }
      return;
    }
    if (request.method === 'POST' && request.url === '/chunk') {
      const parts = [];
      request.on('data', (part) => parts.push(part));
      request.on('end', () => {
        const chunk = Buffer.concat(parts);
        receivedChunks.push(chunk);
        receivedBytes += chunk.length;
        response.writeHead(204);
        response.end();
      });
      return;
    }
    response.writeHead(404);
    response.end();
  });
  try {
    const images = await captureScreenshots(browser);
    if (process.argv.includes('--prepare-only')) {
      console.log('Dashboard capture checks passed: ' + Object.keys(images).join(', ') + '.');
      console.log('No screenshots or video were written.');
      return;
    }
    await new Promise((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '127.0.0.1', resolve);
    });
    const address = server.address();
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
    await page.goto('http://127.0.0.1:' + address.port + '/', { waitUntil: 'domcontentloaded' });
    const painterSource = painterFactory.toString();
    const input = { painterSource, scenes, totalDuration, images, sceneSeconds };
    await page.evaluate(async (arg) => {
      const painter = new Function('return (' + arg.painterSource + ')()')();
      const canvas = document.getElementById('screen');
      const loaded = {};
      await Promise.all(Object.entries(arg.images).map(([name, src]) => new Promise((resolve, reject) => {
        const image = new Image(); image.onload = () => { loaded[name] = image; resolve(); }; image.onerror = reject; image.src = src;
      })));
      window.sentinelDrawFirst = () => painter(canvas.getContext('2d'), 0, arg.scenes, arg.totalDuration, loaded);
      window.sentinelDrawFirst();
      window.sentinelDemo = { painter, canvas, context: canvas.getContext('2d'), scenes: arg.scenes,
        totalDuration: arg.totalDuration, sceneSeconds: arg.sceneSeconds, loaded, recorder: null,
        uploadQueue: Promise.resolve(), chunkCount: 0, byteCount: 0, stream: null,
        start() {
          this.stream = this.canvas.captureStream(24);
          const type = 'video/mp4;codecs=avc1.42E01E';
          if (!MediaRecorder.isTypeSupported(type)) throw new Error('This Edge build cannot record H.264 MP4.');
          this.recorder = new MediaRecorder(this.stream, { mimeType: type, videoBitsPerSecond: 1800000 });
          this.recorder.ondataavailable = (event) => {
            if (!event.data || !event.data.size) return;
            const chunk = event.data;
            this.chunkCount += 1;
            this.byteCount += chunk.size;
            this.uploadQueue = this.uploadQueue.then(async () => {
              const response = await fetch('/chunk', { method: 'POST', body: chunk });
              if (!response.ok) throw new Error('Video chunk transfer failed: ' + response.status);
            });
          };
          const started = performance.now();
          const animate = () => {
            const elapsed = Math.min((performance.now() - started) / 1000, this.totalDuration);
            this.painter(this.context, elapsed, this.scenes, this.totalDuration, this.loaded);
            requestAnimationFrame(animate);
          };
          this.recorder.start(1000);
          requestAnimationFrame(animate);
        },
        async stop() {
          if (!this.recorder || this.recorder.state === 'inactive') throw new Error('The video recorder stopped unexpectedly.');
          const stopped = new Promise((resolve) => this.recorder.addEventListener('stop', resolve, { once: true }));
          this.recorder.requestData();
          this.recorder.stop();
          await stopped;
          await this.uploadQueue;
          this.stream.getTracks().forEach((track) => track.stop());
          return { chunks: this.chunkCount, bytes: this.byteCount };
        },
      };
    }, input);
    if (!smokeRender) await page.screenshot({ path: posterPath });
    await page.evaluate(() => window.sentinelDemo.start());
    for (let index = 0; index < scenes.length; index += 1) {
      await page.waitForTimeout(sceneSeconds * 1000);
      if ((index + 1) % 5 === 0 || index === scenes.length - 1) {
        console.log('Rendered chapter ' + (index + 1) + ' of ' + scenes.length + '.');
      }
    }
    const recording = await page.evaluate(() => window.sentinelDemo.stop());
    if (receivedBytes !== recording.bytes || !receivedChunks.length) throw new Error('The local video encoder did not deliver all recording chunks.');
    renderedBuffer = Buffer.concat(receivedChunks);
    if (renderedBuffer.toString('ascii', 4, 8) !== 'ftyp') throw new Error('The recorded output is not a valid MP4 container.');
    const verifier = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    await verifier.goto('http://127.0.0.1:' + address.port + '/watch', { waitUntil: 'domcontentloaded' });
    const metadata = await verifier.locator('#demo').evaluate((video) => new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('MP4 metadata validation timed out.')), 30000);
      video.addEventListener('loadedmetadata', () => {
        clearTimeout(timer);
        resolve({ duration: video.duration, width: video.videoWidth, height: video.videoHeight });
      }, { once: true });
      video.addEventListener('error', () => {
        clearTimeout(timer);
        reject(new Error('The browser could not read the rendered MP4.'));
      }, { once: true });
      video.load();
    }));
    if (!Number.isFinite(metadata.duration) || Math.abs(metadata.duration - totalDuration) > 2 || metadata.width !== 1280 || metadata.height !== 720) {
      throw new Error('The MP4 duration or video dimensions did not match the storyboard.');
    }
    const vtt = ['WEBVTT', ''];
    scenes.forEach((scene) => {
      vtt.push(clock(scene.start) + ' --> ' + clock(scene.start + scene.duration - 0.25), scene.caption, '');
    });
    if (smokeRender) {
      console.log('Smoke render passed; ' + recording.chunks + ' video chunks, no files written.');
    } else {
      fs.writeFileSync(videoPath, renderedBuffer);
      fs.writeFileSync(captionsPath, vtt.join('\n'), 'utf8');
      console.log('Created ' + scenes.length + ' captioned chapters; duration ' + totalDuration.toFixed(1) + ' seconds.');
      console.log('Video: ' + videoPath + ' (' + (renderedBuffer.length / 1048576).toFixed(1) + ' MiB)');
      console.log('Captions: ' + captionsPath);
      console.log('Poster: ' + posterPath);
    }
    await verifier.close();
    await page.close();
  } finally {
    if (server.listening) await new Promise((resolve) => server.close(resolve));
    await browser.close();
  }
}

main().catch((error) => { console.error(error.stack || error); process.exitCode = 1; });
