import { afterEach, describe, expect, it, vi } from 'vitest';
import { getRecognitionReplay, resetLocalSession, type RecognitionReplayPage, type RecognitionReplaySample, type RecognitionStatus } from './api';
import { loadReplay, MAX_REPLAY_PAGE_BYTES, ReplayClock, replaySampleAt, replaySessionMatches } from './replay';

const identity = { sessionId: 'session-a', cameraId: 'rec-a' };
const sample = (position: number, reading?: string): RecognitionReplaySample => ({
  video_position_ms: position, frame: 'AAAA',
  vehicles: reading ? [{ key: '0-1', track_id: '1', vehicle_class: 'car', confidence: .8,
    first_ms: 0, last_ms: position, observations: 3, state: 'uncertain', reading, thumb: null, watchlist: null }] : [],
  vehicles_seen: reading ? [{ key: '0-1', track_id: '1', vehicle_class: 'car', confidence: .8,
    first_ms: 0, last_ms: position, observations: 3, state: 'uncertain', masked_plate: 'AB******34', watchlist: null }] : [],
  unique_model_accepted_unverified: 0,
});
const page = (offset: number, count = 4): RecognitionReplayPage => ({
  schema_version: '1.0', source_mode: 'recorded', session_id: identity.sessionId, camera_id: identity.cameraId,
  video_duration_ms: 10_000, available: true, reason: null, sample_count: count,
  offset, next_offset: offset + 1 < count ? offset + 1 : null, samples: [sample(offset * 1000)],
});
const options = () => ({ signal: new AbortController().signal });

afterEach(() => { vi.unstubAllGlobals(); resetLocalSession(); });

describe('source-time replay clock', () => {
  it('keeps cached playback during a failed poll but rejects an authoritative run/source/status change', () => {
    const completed: RecognitionStatus = { schema_version: '1.0', status: 'completed', source_mode: 'recorded',
      session_id: identity.sessionId, camera_id: identity.cameraId, recognition_state: 'completed', tracks: [] };
    expect(replaySessionMatches(identity, completed)).toBe(true);
    // The session hook keeps this successful status when a later poll rejects.
    const failedPoll = { error: 'Recognition service unavailable', status: completed };
    expect(replaySessionMatches(identity, failedPoll.status)).toBe(true);
    for (const change of [{ session_id: 'session-b' }, { camera_id: 'rec-b' },
      { status: 'processing' as const }, { status: 'paused' as const }, { source_mode: 'live' as const }]) {
      expect(replaySessionMatches(identity, { ...completed, ...change })).toBe(false);
    }
    expect(replaySessionMatches(identity, null)).toBe(false);
    expect(replaySessionMatches(null, completed)).toBe(false);
  });
  it('advances at 1× using elapsed clock time, pauses and resumes without a jump', () => {
    let now = 100;
    const clock = new ReplayClock(10_000, () => now);
    clock.play(); now += 2000;
    expect(clock.read()).toBe(2000);
    clock.pause(); now += 5000;
    expect(clock.read()).toBe(2000);
    clock.play(); now += 750;
    expect(clock.read()).toBe(2750);
  });
  it('scrubs within source bounds, stops at EOF and can replay again', () => {
    let now = 0;
    const clock = new ReplayClock(5000, () => now);
    clock.play(); now = 1000;
    clock.seek(3500); now = 1500;
    expect(clock.read()).toBe(4000);
    clock.seek(9999);
    expect(clock.read()).toBe(5000);
    expect(clock.playing).toBe(false);
    clock.play(); now += 500;
    expect(clock.read()).toBe(500);
    clock.seek(-10);
    expect(clock.read()).toBe(0);
    clock.seek(Number.NaN);
    expect(clock.read()).toBe(0);
  });
  it('does not show later frames, vehicles or final readings before their PTS', () => {
    const early = sample(500), later = sample(1500, 'AB12CD1234');
    expect(replaySampleAt([early, later], 0)).toBeNull();
    expect(replaySampleAt([early, later], 1499)).toBe(early);
    expect(replaySampleAt([early, later], 1499)?.vehicles).toEqual([]);
    expect(replaySampleAt([early, later], 1500)).toBe(later);
    expect(replaySampleAt([early, later], 9000)?.vehicles[0].reading).toBe('AB12CD1234');
  });
});

describe('bounded replay page loading', () => {
  it('loads exact ordered samples with at most three parallel requests and progress', async () => {
    let active = 0, peak = 0;
    const progress: number[] = [];
    const result = await loadReplay(async offset => {
      active++; peak = Math.max(peak, active);
      await new Promise(resolve => setTimeout(resolve, offset % 2 ? 3 : 1));
      active--;
      return page(offset, 7);
    }, identity, { ...options(), progress: loaded => progress.push(loaded) });
    expect(peak).toBeLessThanOrEqual(3);
    expect(result.samples.map(row => row.video_position_ms)).toEqual([0, 1000, 2000, 3000, 4000, 5000, 6000]);
    expect(progress).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });
  it('rejects a changed session or source instead of showing cached data', async () => {
    for (const change of [{ session_id: 'session-b' }, { camera_id: 'rec-b' }, { source_mode: 'live' }]) {
      await expect(loadReplay(async offset => ({ ...page(offset), ...change }), identity, options())).rejects.toThrow('session changed');
    }
  });
  it('rejects unsupported counts, pagination changes and metadata changes', async () => {
    await expect(loadReplay(async offset => page(offset, 301), identity, options())).rejects.toThrow('display limits');
    await expect(loadReplay(async offset => ({ ...page(offset), next_offset: 0 }), identity, options())).rejects.toThrow('display limits');
    await expect(loadReplay(async offset => ({ ...page(offset), video_duration_ms: offset ? 9000 : 10_000 }), identity, options())).rejects.toThrow('display limits');
  });
  it('rejects a single oversized page and an aggregate memory overflow', async () => {
    const huge = page(0, 1);
    huge.samples[0].frame = 'A'.repeat(MAX_REPLAY_PAGE_BYTES);
    await expect(loadReplay(async () => huge, identity, options())).rejects.toThrow('memory limit');
    const onePageBytes = new TextEncoder().encode(JSON.stringify(page(0))).byteLength;
    await expect(loadReplay(async offset => page(offset), identity,
      { ...options(), maxBytes: onePageBytes + 5 })).rejects.toThrow('memory limit');
  });
  it('does not accept out-of-order frames or metadata from future source positions', async () => {
    await expect(loadReplay(async offset => ({ ...page(offset), samples: [sample(0)] }), identity, options())).rejects.toThrow('increasing source-time');
    const future = sample(500, 'AB12CD1234');
    future.vehicles[0].last_ms = 1500;
    await expect(loadReplay(async offset => ({ ...page(offset, 1), samples: [future] }), identity, options())).rejects.toThrow('display limits');
  });
  it('cancels outstanding loading without retrying', async () => {
    const controller = new AbortController();
    const calls: number[] = [];
    await expect(loadReplay(async offset => {
      calls.push(offset);
      controller.abort();
      return page(offset);
    }, identity, { signal: controller.signal })).rejects.toThrow('cancelled');
    expect(calls).toEqual([0]);
  });
  it('aborts all pending parallel pages when replay loading is cancelled', async () => {
    const controller = new AbortController();
    const pending: AbortSignal[] = [];
    const result = loadReplay(async (offset, signal) => {
      if (offset === 0) return page(0);
      pending.push(signal);
      return await new Promise((_, reject) => signal.addEventListener('abort',
        () => reject(new Error('Replay loading was cancelled.')), { once: true }));
    }, identity, { signal: controller.signal });
    await new Promise(resolve => setTimeout(resolve, 1));
    expect(pending).toHaveLength(3);
    controller.abort();
    await expect(result).rejects.toThrow('cancelled');
    expect(pending.every(signal => signal.aborted)).toBe(true);
  });
  it('surfaces one unavailable/error response without a retry loop', async () => {
    const fetcher = vi.fn(async () => { throw new Error('Replay buffer exhausted.'); });
    await expect(loadReplay(fetcher, identity, options())).rejects.toThrow('buffer exhausted');
    expect(fetcher).toHaveBeenCalledTimes(1);
    await expect(loadReplay(async offset => ({ ...page(offset), available: false, reason: 'This analysis has no replay buffer.' }),
      identity, options())).rejects.toThrow('no replay buffer');
  });
});

describe('protected replay API', () => {
  it('uses read-only one-sample pages, cancellation and the selected frame mode', async () => {
    const signal = new AbortController().signal;
    const fetcher = vi.fn(async (path: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify(
      path === '/v1/local/session' ? { csrf_token: 'transient', expires_at: new Date(Date.now() + 60_000).toISOString() } : page(2)),
      { status: 200, headers: { 'Content-Type': 'application/json' } }));
    vi.stubGlobal('fetch', fetcher);
    await getRecognitionReplay('session-a', 2, 'original', signal);
    const request = fetcher.mock.calls.find(call => String(call[0]).startsWith('/v1/recognition/replay'));
    expect(request?.[0]).toBe('/v1/recognition/replay?session_id=session-a&offset=2&limit=1&mode=original');
    expect(request?.[1]?.signal).toBe(signal);
    expect(request?.[1]?.method).toBeUndefined();
    // A GET replay never calls the session/control mutation endpoints.
    expect(fetcher.mock.calls.filter(call => String(call[0]).includes('/recognition/control'))).toHaveLength(0);
  });
  it('does not retry a not-complete conflict', async () => {
    const fetcher = vi.fn(async (path: RequestInfo | URL) => new Response(JSON.stringify(
      path === '/v1/local/session' ? { csrf_token: 'transient', expires_at: new Date(Date.now() + 60_000).toISOString() }
        : { detail: 'The analysis has not completed.' }),
      { status: path === '/v1/local/session' ? 200 : 409, headers: { 'Content-Type': 'application/json' } }));
    vi.stubGlobal('fetch', fetcher);
    await expect(getRecognitionReplay('session-a')).rejects.toThrow('not completed');
    expect(fetcher.mock.calls.filter(call => String(call[0]).includes('/recognition/replay'))).toHaveLength(1);
  });
});
