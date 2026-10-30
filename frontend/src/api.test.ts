import { beforeEach, describe, expect, it, vi } from 'vitest';
import { cameraPreview, formatTime, getAlerts, getHourlyReports, injectDemo, searchCatalogue, startCataloguePreview,
  getRecognitionStatus, getRecognitionCameras, setRecognitionSession, heartbeatRecognition, getRecognitionRecordings,
  controlRecognition, ensureLocalSession, resetLocalSession, addWatchlist, patchWatchlist, searchWatchlist, getDiagnostics, getRecognitionEvidence } from './api';
import { imageData } from './Inspector';

const values = new Map<string, string>();
const localSession = { csrf_token: 'test-request-token', expires_at: '2037-01-01T00:00:00Z' };
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
function mockApi(value: unknown = {}) {
  const fetcher = vi.fn(async (path: RequestInfo | URL, _init?: RequestInit) => String(path) === '/v1/local/session' ? json(localSession) : json(value));
  vi.stubGlobal('fetch', fetcher);
  return fetcher;
}
const operationCalls = (fetcher: ReturnType<typeof mockApi>) => fetcher.mock.calls.filter(call => String(call[0]) !== '/v1/local/session');

beforeEach(() => {
  vi.restoreAllMocks(); resetLocalSession(); values.clear();
  vi.stubGlobal('sessionStorage', { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => values.set(key, value), removeItem: (key: string) => values.delete(key) });
});

describe('automatic local-session boundary', () => {
  it('bootstraps once without operator secrets and removes obsolete stored keys', async () => {
    values.set('sentinel.operatorKey', 'obsolete-private-value');
    const fetcher = mockApi([]);
    await Promise.all([getAlerts(), getAlerts()]);
    expect(fetcher.mock.calls.filter(call => call[0] === '/v1/local/session')).toHaveLength(1);
    expect(values.size).toBe(0);
    for (const [, init] of operationCalls(fetcher)) {
      expect(init?.credentials).toBe('same-origin');
      expect((init?.headers as Headers).get('X-Operator-Key')).toBeNull();
    }
  });
  it('uses a transient CSRF token for mutations, not for reads', async () => {
    const fetcher = mockApi({});
    await getRecognitionStatus(); await setRecognitionSession('rec-example');
    const [read, write] = operationCalls(fetcher);
    expect((read[1]?.headers as Headers).get('X-CSRF-Token')).toBeNull();
    expect((write[1]?.headers as Headers).get('X-CSRF-Token')).toBe(localSession.csrf_token);
    expect(values.size).toBe(0);
  });
  it('exposes a setup failure instead of prompting for a manual key', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json({}, 403)));
    await expect(ensureLocalSession()).rejects.toThrow('Local access is unavailable');
  });
  it('rejects malformed bootstrap payloads', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json({ csrf_token: '', expires_at: 'invalid' })));
    await expect(ensureLocalSession()).rejects.toThrow('invalid');
  });
  it('re-establishes an expired session once following a pre-operation 401', async () => {
    let requests = 0;
    const fetcher = vi.fn(async (path: RequestInfo | URL) => path === '/v1/local/session' ? json(localSession) : ++requests === 1 ? json({}, 401) : json([]));
    vi.stubGlobal('fetch', fetcher);
    await getAlerts();
    expect(fetcher.mock.calls.filter(call => call[0] === '/v1/local/session')).toHaveLength(2);
    expect(requests).toBe(2);
  });
  it('does not replay a conflicting write', async () => {
    const fetcher = vi.fn(async (path: RequestInfo | URL) => path === '/v1/local/session' ? json(localSession) : json({ detail: 'Entry changed; refresh.' }, 409));
    vi.stubGlobal('fetch', fetcher);
    await expect(patchWatchlist(3, { expected_version: 1, active: false })).rejects.toThrow('Entry changed');
    expect(fetcher.mock.calls.filter(call => call[0] !== '/v1/local/session')).toHaveLength(1);
  });
  it('does not blindly retry an interrupted write', async () => {
    const fetcher = vi.fn(async (path: RequestInfo | URL) => { if (path === '/v1/local/session') return json(localSession); throw new TypeError('network'); });
    vi.stubGlobal('fetch', fetcher);
    await expect(setRecognitionSession('rec-example')).rejects.toThrow('unreachable');
    expect(fetcher.mock.calls.filter(call => call[0] !== '/v1/local/session')).toHaveLength(1);
  });
});

describe('existing API compatibility', () => {
  it('converts numeric alert IDs and preserves recorded traceability', async () => {
    mockApi([{ id: 42, sighting_id: 7, camera_id: 'rec-example', source_mode: 'recorded', stream_pts_ms: 4500, session_id: 's-1', time_basis: 'processing_time' }]);
    const [alert] = await getAlerts();
    expect([alert.id, alert.sighting_id, alert.source_mode, alert.stream_pts_ms, alert.session_id, alert.time_basis]).toEqual(['42', '7', 'recorded', 4500, 's-1', 'processing_time']);
  });
  it('parses central hourly aggregation buckets without exposing that legacy navigation', async () => {
    mockApi({ buckets: [{ alerts: 1 }] }); expect((await getHourlyReports())[0].alerts).toBe(1);
  });
  it('keeps the explicit demo contract, without a normal UI injection control', async () => {
    const fetcher = mockApi({ accepted: true }); await injectDemo();
    expect(operationCalls(fetcher)[0][1]?.body).toBe('{}');
  });
  it('preserves catalogue paging and URL-free preview starts', async () => {
    const fetcher = mockApi({ items: [] }); await searchCatalogue(' Janpath ', 25); await startCataloguePreview('cam03');
    expect(operationCalls(fetcher)[0][0]).toContain('q=Janpath&offset=25&limit=25');
    expect(operationCalls(fetcher)[1][0]).toBe('/v1/catalogue/cam03/preview');
    expect(operationCalls(fetcher)[1][1]?.method).toBe('POST');
  });
  it('observes shared state without implicitly starting inference', async () => {
    const fetcher = mockApi({ status: 'paused', tracks: [] });
    expect((await getRecognitionStatus()).status).toBe('paused');
    expect(operationCalls(fetcher)[0][0]).toBe('/v1/recognition/status');
    expect(operationCalls(fetcher)[0][1]?.method).toBeUndefined();
    expect(operationCalls(fetcher)[0][1]?.cache).toBe('no-store');
  });
  it('retains limited configured camera scope in the legacy adapter', async () => {
    mockApi({ available: true, scope: 'configured_relays', cameras: [{ id: 'cam01' }] });
    expect((await getRecognitionCameras()).scope).toBe('configured_relays');
  });
  it('uses explicit source-only start, heartbeat and stop', async () => {
    const fetcher = mockApi(); await setRecognitionSession('rec-example'); await heartbeatRecognition('rec-example'); await setRecognitionSession(null);
    expect(operationCalls(fetcher).map(call => call[1]?.body)).toEqual(['{"camera_id":"rec-example"}', '{"camera_id":"rec-example"}', '{"camera_id":null}']);
  });
  it('lists recordings independently and serializes supported playback actions', async () => {
    const fetcher = mockApi({ recordings: [{ id: 'rec-example', name: 'Junction' }] });
    expect((await getRecognitionRecordings()).recordings[0].id).toBe('rec-example');
    await controlRecognition('pause'); await controlRecognition('resume'); await controlRecognition('restart', 999); await controlRecognition('seek', -50);
    expect(operationCalls(fetcher).slice(1).map(call => call[1]?.body)).toEqual(['{"action":"pause"}', '{"action":"resume"}', '{"action":"restart"}', '{"action":"seek","position_ms":0}']);
  });
});

describe('new operator contracts', () => {
  it('keeps watchlist identifiers out of URLs', async () => {
    const fetcher = mockApi(); await addWatchlist({ plate: 'GJ01AB1234', reason: 'Fixture' }); await searchWatchlist('GJ01AB1234');
    for (const call of operationCalls(fetcher)) { expect(String(call[0])).not.toContain('GJ01AB1234'); expect(call[1]?.body).toContain('GJ01AB1234'); }
  });
  it('includes expected revision for watchlist edits', async () => {
    const fetcher = mockApi(); await patchWatchlist(9, { expected_version: 4, active: false });
    expect(operationCalls(fetcher)[0][0]).toBe('/v1/watchlist/9');
    expect(operationCalls(fetcher)[0][1]?.body).toBe('{"expected_version":4,"active":false}');
  });
  it('pins diagnostics to a selected session and track', async () => {
    const fetcher = mockApi(); await getDiagnostics('s-1', 'det-2');
    expect(operationCalls(fetcher)[0][0]).toBe('/v1/recognition/diagnostics?session_id=s-1&track_id=det-2');
  });
  it('treats disabled evidence as unavailable but does not conceal service failures', async () => {
    const fetcher = vi.fn(async (path: RequestInfo | URL) => path === '/v1/local/session' ? json(localSession) : json({}, 404));
    vi.stubGlobal('fetch', fetcher); expect(await getRecognitionEvidence('original')).toBeNull();
    expect(fetcher.mock.calls.at(-1)?.[0]).toBe('/v1/recognition/evidence?mode=original');
    fetcher.mockImplementation(async path => path === '/v1/local/session' ? json(localSession) : json({}, 503));
    await expect(getRecognitionEvidence()).rejects.toThrow('503');
  });
});

describe('presentation safety', () => {
  it('shows UTC observations in IST', () => { expect(formatTime('2026-09-23T12:00:00Z')).toContain('17:30'); });
  it('preserves the local media fallback', () => { expect(cameraPreview({ id: 'cam01', name: 'Camera', status: 'online' })).toBe('/media/cam01/index.m3u8'); });
  it('accepts bounded base64 images only, never active content or remote URLs', () => {
    expect(imageData('YWJj')).toBe('data:image/png;base64,YWJj');
    expect(imageData('https://example.org/frame')).toBeUndefined();
    expect(imageData('javascript:alert(1)')).toBeUndefined();
    expect(imageData('A'.repeat(1_500_001))).toBeUndefined();
  });
});
