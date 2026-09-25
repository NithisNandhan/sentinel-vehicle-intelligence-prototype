import { beforeEach, describe, expect, it, vi } from 'vitest';
import { cameraPreview, formatTime, getAlerts, getHourlyReports, injectDemo,
  searchCatalogue, setOperatorKey, startCataloguePreview } from './api';

const values = new Map<string, string>();

beforeEach(() => {
  values.clear();
  vi.stubGlobal('sessionStorage', {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value); },
    removeItem: (key: string) => { values.delete(key); },
  });
  vi.restoreAllMocks();
});

describe('operator API boundary', () => {
  it('uses the operator key and converts numeric alert IDs for UI selection', async () => {
    setOperatorKey('test-operator-key');
    const fetcher = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => [{
      id: 42, sighting_id: 7, camera_id: 'cam01', observed_at: '2026-09-23T12:00:00Z',
      masked_plate: 'GJ••1234', reason: 'Synthetic test', severity: 'high', status: 'new', source_mode: 'synthetic',
    }] });
    vi.stubGlobal('fetch', fetcher);
    const alerts = await getAlerts();
    expect(alerts[0].id).toBe('42');
    expect(alerts[0].sighting_id).toBe('7');
    const headers = (fetcher.mock.calls[0][1] as RequestInit).headers as Headers;
    expect(headers.get('X-Operator-Key')).toBe('test-operator-key');
  });

  it('parses central hourly aggregation buckets', async () => {
    setOperatorKey('test-operator-key');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({
      hours: 24, buckets: [{ window_start: '2026-09-23T12:00:00Z', sightings: 3, alerts: 1 }],
    }) }));
    expect((await getHourlyReports())[0].alerts).toBe(1);
  });

  it('sends an empty JSON object for the demo injector request model', async () => {
    setOperatorKey('test-operator-key');
    const fetcher = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ accepted: true }) });
    vi.stubGlobal('fetch', fetcher);
    await injectDemo();
    expect((fetcher.mock.calls[0][1] as RequestInit).body).toBe('{}');
  });

  it('searches the catalogue with the operator key and pagination', async () => {
    setOperatorKey('test-operator-key');
    const fetcher = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({
      total: 1, offset: 25, limit: 25,
      items: [{ id: 'cam03', name: 'Janpath', location: 'Janpath', codec: 'h264', live: true }],
    }) });
    vi.stubGlobal('fetch', fetcher);
    expect((await searchCatalogue(' Janpath ', 25)).items[0].id).toBe('cam03');
    expect(fetcher.mock.calls[0][0]).toContain('q=Janpath&offset=25&limit=25');
    expect(((fetcher.mock.calls[0][1] as RequestInit).headers as Headers).get('X-Operator-Key'))
      .toBe('test-operator-key');
  });

  it('starts an on-demand preview without sending a source URL', async () => {
    setOperatorKey('test-operator-key');
    const fetcher = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({
      id: 'a'.repeat(32), camera_id: 'cam03', status: 'starting',
      preview_url: `/media/preview-${'a'.repeat(32)}/index.m3u8`,
    }) });
    vi.stubGlobal('fetch', fetcher);
    await startCataloguePreview('cam03');
    expect(fetcher.mock.calls[0][0]).toBe('/v1/catalogue/cam03/preview');
    expect((fetcher.mock.calls[0][1] as RequestInit).method).toBe('POST');
  });
});

describe('presentation helpers', () => {
  it('shows UTC observations in IST', () => {
    expect(formatTime('2026-09-23T12:00:00Z')).toContain('17:30');
  });

  it('uses the local media proxy when the catalogue has no preview URL', () => {
    expect(cameraPreview({ id: 'cam01', name: 'Camera 1', status: 'online' })).toBe('/media/cam01/index.m3u8');
  });
});
