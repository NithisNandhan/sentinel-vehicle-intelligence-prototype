/* Read-only smoke check of the deployed trusted-local dashboard.
 * Does not start inference, alter watchlists/alerts, or save footage/screenshots.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'C:/Users/NITHIS/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const context = await browser.newContext();
    const page = await context.newPage();
    const errors = [];
    let starts = 0;
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => { if (request.url().endsWith('/v1/recognition/session') && request.method() === 'POST') starts++; });
    await page.goto('http://127.0.0.1:5173/');
    await page.getByRole('heading', { name: 'Analyse footage' }).waitFor();
    assert.equal(await page.getByRole('button', { name: /unlock/i }).count(), 0);
    const sessionCookie = (await context.cookies()).find(cookie => cookie.name === 'sentinel_local');
    assert(sessionCookie?.httpOnly && sessionCookie.sameSite === 'Strict');
    const checks = await page.evaluate(async () => {
      const bootstrap = await fetch('/v1/local/session', { method: 'POST' });
      const session = await bootstrap.json();
      const withoutCsrf = await fetch('/v1/watchlist/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{"query":""}' });
      const allowed = await fetch('/v1/watchlist/search', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': session.csrf_token }, body: '{"query":""}' });
      const watchlist = await allowed.json();
      const status = await fetch('/v1/recognition/status');
      const recognition = await status.json();
      const recordings = await fetch('/v1/recognition/recordings');
      const sources = await recordings.json();
      const alerts = await fetch('/v1/alerts');
      const alertRows = await alerts.json();
      return { bootstrap: bootstrap.status, withoutCsrf: withoutCsrf.status, watchlist: allowed.status, entries: watchlist.total, recognition: status.status, workerState: recognition.status, recordings: recordings.status, recordingCount: sources.recordings?.length, alerts: alerts.status, alertCount: Array.isArray(alertRows) ? alertRows.length : null };
    });
    assert.equal(checks.bootstrap, 200);
    assert.equal(checks.withoutCsrf, 403);
    for (const key of ['watchlist', 'recognition', 'recordings', 'alerts']) assert.equal(checks[key], 200, key);
    assert(checks.recordingCount > 0, 'At least one permitted recording must be registered');
    await page.getByRole('button', { name: 'Watchlist', exact: true }).click();
    await page.getByRole('heading', { name: 'Watchlist', exact: true }).waitFor();
    await page.getByRole('button', { name: /^Alerts/ }).click();
    await page.getByRole('heading', { name: 'Alerts', exact: true }).waitFor();
    await page.getByRole('button', { name: 'Analyse', exact: true }).click();
    await page.getByText('Service connected', { exact: true }).waitFor();
    assert.equal(starts, 0, 'Opening or navigating must not start recognition');
    assert.equal(errors.length, 0, errors.join('\n'));
    const old = await context.request.get('http://127.0.0.1:7780/', { maxRedirects: 0 });
    assert.equal(old.status(), 307);
    assert.equal(old.headers().location, 'http://127.0.0.1:5173/');
    const report = { ...checks, httpOnlyCookie: true, websocketConnected: true, legacyRedirect: 307, automaticStarts: starts, browserErrors: errors.length, note: 'Read-only check of deployed services; no recognition-accuracy claim.' };
    const output = path.resolve(process.argv[2] || 'data/ui-validation/unified-20261030');
    fs.mkdirSync(output, { recursive: true });
    fs.writeFileSync(path.join(output, 'local-smoke.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
    await context.close();
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
