// Measures local synthetic alert delivery to the same WebSocket used by the dashboard.
// This does not measure browser rendering or live OCR latency.
import { readFileSync } from 'node:fs';
import { createHmac, randomUUID } from 'node:crypto';
import { performance } from 'node:perf_hooks';

function localEnv(name) {
  if (process.env[name]) return process.env[name];
  const lines = readFileSync(new URL('../.env', import.meta.url), 'utf8').replace(/^\uFEFF/, '').split(/\r?\n/);
  const line = lines.find(item => item.startsWith(`${name}=`));
  if (!line) throw new Error(`${name} is missing from environment and local .env`);
  return line.slice(name.length + 1);
}

const base = (process.argv[2] || 'http://127.0.0.1:8000').replace(/\/$/, '');
const edgeKey = localEnv('EDGE_API_KEY');
const operatorKey = localEnv('OPERATOR_API_KEY');
const hmacKey = localEnv('WATCHLIST_HMAC_KEY');
const plateHmac = createHmac('sha256', hmacKey).update('GJ01AB1234').digest('hex');
const eventId = randomUUID();
const messages = [];
const ws = new WebSocket(base.replace(/^http/, 'ws') + '/v1/ws/alerts',
  ['sentinel-operator', operatorKey]);

const connected = new Promise((resolve, reject) => {
  ws.addEventListener('open', resolve, { once: true });
  ws.addEventListener('error', () => reject(new Error('WebSocket connection failed')), { once: true });
});
ws.addEventListener('message', event => {
  try { messages.push({ at: performance.now(), data: JSON.parse(event.data) }); }
  catch { /* Ignore malformed unrelated traffic. */ }
});

try {
  await connected;
  const started = performance.now();
  const response = await fetch(base + '/v1/sightings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Edge-Key': edgeKey },
    body: JSON.stringify({
      schema_version: '1.0', event_id: eventId, edge_id: 'latency-test', camera_id: 'cam01',
      observed_at: new Date().toISOString(), stream_pts_ms: null, track_id: 'latency-test',
      masked_plate: 'GJ******34', plate_hmac: plateHmac, confidence: 0.99,
      source_mode: 'synthetic', model_version: 'latency-test-1',
    }),
  });
  if (!response.ok) throw new Error(`Sighting POST returned HTTP ${response.status}`);
  const payload = await response.json();
  const alertId = payload.alert?.id;
  if (alertId == null) throw new Error('Synthetic watchlist match did not create an alert');
  let received;
  const deadline = performance.now() + 5000;
  while (performance.now() < deadline) {
    received = messages.find(message => message.data.type === 'alert_created'
      && message.data.alert?.id === alertId);
    if (received) break;
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  if (!received) throw new Error('Alert did not arrive over WebSocket within five seconds');
  const latencyMs = Math.round((received.at - started) * 100) / 100;
  console.log(JSON.stringify({ event_id: eventId, alert_id: alertId,
    websocket_latency_ms: latencyMs, under_3_seconds: latencyMs < 3000 }, null, 2));
  if (latencyMs >= 3000) process.exitCode = 1;
} finally {
  ws.close();
}
