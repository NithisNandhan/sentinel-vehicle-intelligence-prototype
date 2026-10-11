// Take real browser screenshots of both interfaces while a recording is analysed.
//
//   node tools/capture_screens.mjs <recording name fragment> [seek seconds] [vehicles to wait for] [readings to wait for]
//
// The shot is taken on a frame where a plate call-out is on screen.
//
// Needs `puppeteer-core` resolvable from the working directory and an installed
// Chrome or Edge. It drives the pages exactly as a person would: unlocks the
// engineering console with the operator key from .env, starts the recording,
// waits until vehicles are listed, then saves PNGs under data/screenshots
// (ignored by Git; the images contain real footage). The key is typed into the
// page and never printed or written to disk.
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const [wanted = '', seekSeconds = '0', waitFor = '6', readings = '1'] = process.argv.slice(2);
const key = readFileSync(path.join(root, '.env'), 'utf8').split(/\r?\n/)
  .find(line => line.startsWith('OPERATOR_API_KEY='))?.split('=')[1]?.trim();
if (!key) throw new Error('OPERATOR_API_KEY is missing from .env');
const browserPath = ['C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe', '/usr/bin/google-chrome', '/usr/bin/chromium']
  .find(candidate => existsSync(candidate));
if (!browserPath) throw new Error('No Chrome or Edge found');
const out = path.join(root, 'data', 'screenshots');
mkdirSync(out, { recursive: true });
const stamp = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
const save = name => path.join(out, `${stamp}-${name}.png`);

const browser = await puppeteer.launch({ executablePath: browserPath, headless: true,
  defaultViewport: { width: 1680, height: 1050 } });
try {
  const consolePage = await browser.newPage();
  await consolePage.goto('http://127.0.0.1:7780/', { waitUntil: 'domcontentloaded' });
  await consolePage.type('#operator-key', key);
  await consolePage.click('#unlock');
  await consolePage.waitForFunction(() => !document.getElementById('inspection').hidden, { timeout: 20000 });
  await consolePage.waitForFunction(text => [...document.getElementById('camera-select').options]
    .some(option => option.textContent.toLowerCase().includes(text)), { timeout: 60000 }, wanted.toLowerCase());
  await consolePage.evaluate(text => {
    const select = document.getElementById('camera-select');
    select.value = [...select.options].find(option => option.textContent.toLowerCase().includes(text)).value;
  }, wanted.toLowerCase());
  await consolePage.click('#open-camera');
  if (Number(seekSeconds) > 0) {
    await consolePage.waitForFunction(() => !document.getElementById('playback-seek').disabled, { timeout: 60000 });
    await consolePage.$eval('#seek-seconds', (input, value) => { input.value = value; }, seekSeconds);
    await consolePage.click('#playback-seek');
  }
  // Wait for real analysed frames and for vehicles to be listed as they pass.
  await consolePage.waitForFunction((count, needed) => !document.getElementById('frame').hidden
    && document.getElementById('passed-list').children.length >= count
    && [...document.querySelectorAll('#passed-list .platechip:not(.none)')].length >= needed
    && typeof latestSnapshot === 'object' && latestSnapshot && latestSnapshot.plate_in_frame === true,
    { timeout: 420000, polling: 250 }, Number(waitFor), Number(readings));
  await consolePage.screenshot({ path: save('console-full'), fullPage: true });
  const grid = await consolePage.$('.workgrid');
  await grid.screenshot({ path: save('console-video-and-list') });
  const frame = await consolePage.$('.viewport');
  await frame.screenshot({ path: save('console-video') });

  const dashboard = await browser.newPage();
  await dashboard.goto('http://127.0.0.1:5173/', { waitUntil: 'domcontentloaded' });
  await dashboard.waitForSelector('.vehicles-seen li', { timeout: 60000 });
  // When the recorded-evidence view is switched on, wait for its analysed frame too.
  await dashboard.waitForSelector('.recognition-evidence figure img', { timeout: 15000 }).catch(() => {});
  await new Promise(resolve => setTimeout(resolve, 1500));
  const panel = await dashboard.$('.recognition-panel');
  await panel.screenshot({ path: save('dashboard-recognition-panel') });
  await dashboard.screenshot({ path: save('dashboard-full'), fullPage: true });
  const state = await consolePage.evaluate(() => ({
    session: document.getElementById('session-state').textContent,
    position: document.getElementById('video-position').textContent,
    listed: document.getElementById('passed-list').children.length,
  }));
  console.log(JSON.stringify({ saved: out, stamp, ...state }));
} finally {
  await browser.close();
}
