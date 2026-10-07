const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const os = require('node:os');
const { chromium } = require('playwright');

const videoPath = path.resolve(process.argv[2]);
const captionsPath = path.resolve(process.argv[3]);
if (!fs.existsSync(videoPath) || !fs.existsSync(captionsPath)) throw new Error('Video or caption file is missing.');
const videoStat = fs.statSync(videoPath);

const server = http.createServer((request, response) => {
  if (request.url === '/') {
    response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    response.end('<!doctype html><html><body style="margin:0;background:#08111f"><video id="demo" controls preload="metadata" style="width:1280px;height:720px" src="/demo.webm"><track kind="captions" src="/demo.vtt" default></video></body></html>');
    return;
  }
  if (request.url.startsWith('/demo.vtt')) {
    response.writeHead(200, { 'Content-Type': 'text/vtt; charset=utf-8' });
    fs.createReadStream(captionsPath).pipe(response);
    return;
  }
  if (request.url.startsWith('/demo.webm')) {
    const range = request.headers.range;
    if (range && /^bytes=\d+-\d*$/.test(range)) {
      const start = Number(range.slice(6).split('-')[0]);
      const requestedEnd = range.split('-')[1];
      const end = Math.min(requestedEnd ? Number(requestedEnd) : videoStat.size - 1, videoStat.size - 1);
      response.writeHead(206, {
        'Content-Type': 'video/webm',
        'Accept-Ranges': 'bytes',
        'Content-Range': 'bytes ' + start + '-' + end + '/' + videoStat.size,
        'Content-Length': end - start + 1,
      });
      fs.createReadStream(videoPath, { start, end }).pipe(response);
    } else {
      response.writeHead(200, { 'Content-Type': 'video/webm', 'Accept-Ranges': 'bytes', 'Content-Length': videoStat.size });
      fs.createReadStream(videoPath).pipe(response);
    }
    return;
  }
  response.writeHead(404);
  response.end();
});

async function main() {
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  const browser = await chromium.launch({
    headless: true,
    executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    await page.goto('http://127.0.0.1:' + address.port + '/', { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => document.getElementById('demo').readyState >= 1, null, { timeout: 30000 });
    const metadata = await page.locator('#demo').evaluate((video) => ({
      duration: video.duration,
      width: video.videoWidth,
      height: video.videoHeight,
      formatReady: video.readyState,
    }));
    if (!Number.isFinite(metadata.duration) || metadata.duration < 90 || metadata.width !== 1280 || metadata.height !== 720) {
      throw new Error('Video metadata did not match the expected 720p, 90-second demo: ' + JSON.stringify(metadata));
    }
    const checks = [];
    for (const second of [0, 24, 48, 72, Math.floor(metadata.duration - 2)]) {
      const result = await page.locator('#demo').evaluate(async (video, time) => {
        if (time > 0) {
          await new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('Timed out seeking to ' + time)), 15000);
            video.addEventListener('seeked', () => { clearTimeout(timer); resolve(); }, { once: true });
            video.currentTime = time;
          });
        }
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        return {
          at: video.currentTime,
          readyState: video.readyState,
          framePng: time === 48 ? canvas.toDataURL('image/png').split(',')[1] : null,
        };
      }, second);
      if (result.framePng) {
        const previewPath = path.join(os.tmpdir(), 'sentinel-demo-midpoint-review.png');
        fs.writeFileSync(previewPath, Buffer.from(result.framePng, 'base64'));
        delete result.framePng;
      }
      checks.push(result);
    }
    const captionCount = await page.locator('#demo track').evaluate(async (track) => {
      track.track.mode = 'hidden';
      await new Promise((resolve) => {
        if (track.readyState === 2) resolve();
        else track.addEventListener('load', resolve, { once: true });
      });
      return track.track.cues ? track.track.cues.length : 0;
    });
    if (captionCount !== 10) throw new Error('Expected 10 WebVTT caption cues, found ' + captionCount);
    console.log(JSON.stringify({ videoBytes: videoStat.size, ...metadata, sampledPositions: checks, captionCues: captionCount }, null, 2));
  } finally {
    await browser.close();
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
}

main().catch((error) => {
  console.error(error.stack || error);
  server.close();
  process.exitCode = 1;
});
