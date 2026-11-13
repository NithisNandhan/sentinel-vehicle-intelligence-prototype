/** Bounded recorded-analysis loading and source-PTS playback; no AI operations. */
import type { RecognitionReplayPage, RecognitionReplaySample, RecognitionStatus } from './api';

export const MAX_REPLAY_SAMPLES = 300;
export const MAX_REPLAY_BYTES = 64 * 1024 * 1024;
export const MAX_REPLAY_PAGE_BYTES = 1024 * 1024;

export interface ReplayData {
  sessionId: string;
  cameraId: string;
  durationMs: number;
  samples: RecognitionReplaySample[];
}

export interface ReplayIdentity { sessionId: string; cameraId: string }
/** A failed poll does not replace the last successful status. Cached playback
 *  remains valid until a successful response confirms that its run changed. */
export function replaySessionMatches(identity: ReplayIdentity | null,
  lastSuccessfulStatus: RecognitionStatus | null): boolean {
  return !!(identity && lastSuccessfulStatus?.status === 'completed'
    && lastSuccessfulStatus.source_mode === 'recorded'
    && identity.sessionId === lastSuccessfulStatus.session_id
    && identity.cameraId === lastSuccessfulStatus.camera_id);
}
type PageFetcher = (offset: number, signal: AbortSignal) => Promise<unknown>;
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const integer = (value: unknown): value is number => finite(value) && Number.isInteger(value) && value >= 0;

function validSample(sample: unknown, duration: number): sample is RecognitionReplaySample {
  if (!sample || typeof sample !== 'object') return false;
  const row = sample as RecognitionReplaySample;
  if (!finite(row.video_position_ms) || row.video_position_ms < 0 || row.video_position_ms > duration
      || !(row.frame === null || typeof row.frame === 'string')
      || !integer(row.unique_model_accepted_unverified)) return false;
  for (const list of [row.vehicles, row.vehicles_seen]) {
    if (!Array.isArray(list) || list.length > 200) return false;
    for (const vehicle of list) {
      if (!vehicle || typeof vehicle !== 'object' || typeof vehicle.track_id !== 'string'
          || vehicle.track_id.length > 96 || typeof vehicle.vehicle_class !== 'string'
          || !integer(vehicle.observations) || typeof vehicle.state !== 'string') return false;
      // A saved sample must not include vehicles/updates from a future position.
      for (const time of [vehicle.first_ms, vehicle.last_ms]) {
        if (time !== null && time !== undefined && (!finite(time) || time < 0 || time > row.video_position_ms)) return false;
      }
    }
  }
  return true;
}

function checkedPage(value: unknown, identity: ReplayIdentity, offset: number,
  expected?: { count: number; duration: number }): { page: RecognitionReplayPage; bytes: number } {
  if (!value || typeof value !== 'object') throw new Error('The replay response is invalid.');
  const page = value as RecognitionReplayPage;
  if (page.schema_version !== '1.0' || page.source_mode !== 'recorded'
      || page.session_id !== identity.sessionId || page.camera_id !== identity.cameraId) {
    throw new Error('The analysis session changed. Exit replay and refresh.');
  }
  if (!page.available) throw new Error(typeof page.reason === 'string' ? page.reason : 'Replay is unavailable for this analysis run.');
  if (!integer(page.sample_count) || page.sample_count < 1 || page.sample_count > MAX_REPLAY_SAMPLES
      || !finite(page.video_duration_ms) || page.video_duration_ms < 0 || page.offset !== offset
      || !Array.isArray(page.samples) || page.samples.length !== 1
      || page.next_offset !== (offset + 1 < page.sample_count ? offset + 1 : null)
      || !validSample(page.samples[0], page.video_duration_ms)
      || (expected && (page.sample_count !== expected.count || page.video_duration_ms !== expected.duration))) {
    throw new Error('The replay is incomplete or exceeds the display limits.');
  }
  const bytes = new TextEncoder().encode(JSON.stringify(value)).byteLength;
  if (bytes > MAX_REPLAY_PAGE_BYTES) throw new Error('A replay sample exceeds the display memory limit.');
  return { page, bytes };
}

export async function loadReplay(fetchPage: PageFetcher, identity: ReplayIdentity,
  options: { signal: AbortSignal; progress?: (loaded: number, total: number) => void; maxBytes?: number }): Promise<ReplayData> {
  const maximum = options.maxBytes ?? MAX_REPLAY_BYTES;
  if (!integer(maximum) || maximum < 1 || maximum > MAX_REPLAY_BYTES) throw new Error('Invalid replay memory limit.');
  const controller = new AbortController();
  const cancel = () => controller.abort();
  if (options.signal.aborted) cancel();
  else options.signal.addEventListener('abort', cancel, { once: true });
  const assertActive = () => { if (controller.signal.aborted) throw new Error('Replay loading was cancelled.'); };
  try {
    assertActive();
    const first = checkedPage(await fetchPage(0, controller.signal), identity, 0);
    assertActive();
    let bytes = first.bytes;
    if (bytes > maximum) throw new Error('Replay exceeds the display memory limit.');
    const count = first.page.sample_count;
    const samples: RecognitionReplaySample[] = new Array(count);
    samples[0] = first.page.samples[0];
    let next = 1, loaded = 1;
    let failure: unknown, failed = false;
    options.progress?.(loaded, count);
    const worker = async () => {
      try {
        while (next < count) {
          assertActive();
          const offset = next++;
          const { page, bytes: pageBytes } = checkedPage(await fetchPage(offset, controller.signal), identity, offset,
            { count, duration: first.page.video_duration_ms });
          assertActive();
          bytes += pageBytes;
          if (bytes > maximum) throw new Error('Replay exceeds the display memory limit.');
          samples[offset] = page.samples[0];
          options.progress?.(++loaded, count);
        }
      } catch (error) { if (!failed) { failure = error; failed = true; } controller.abort(); }
    };
    await Promise.all(Array.from({ length: Math.min(3, count - 1) }, worker));
    if (failed) throw failure;
    assertActive();
    if (samples.some((sample, index) => index > 0 && sample.video_position_ms <= samples[index - 1].video_position_ms)) {
      throw new Error('The replay samples are not in increasing source-time order.');
    }
    return { ...identity, durationMs: first.page.video_duration_ms, samples };
  } finally {
    controller.abort();
    options.signal.removeEventListener('abort', cancel);
  }
}

/** Latest sample at or before the source clock; never return future readings. */
export function replaySampleAt(samples: readonly RecognitionReplaySample[], positionMs: number): RecognitionReplaySample | null {
  if (!finite(positionMs)) return null;
  let low = 0, high = samples.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (samples[middle].video_position_ms <= positionMs) low = middle + 1;
    else high = middle;
  }
  return low ? samples[low - 1] : null;
}

/** 1× source-time clock, independent of inference speed or reported frame rate. */
export class ReplayClock {
  private anchor = 0;
  private position = 0;
  playing = false;
  constructor(readonly durationMs: number, private now = () => performance.now()) {
    if (!finite(durationMs) || durationMs < 0) throw new Error('Invalid replay duration.');
  }
  read(): number {
    const value = Math.min(this.durationMs, Math.max(0, this.position + (this.playing ? this.now() - this.anchor : 0)));
    if (value >= this.durationMs) { this.position = value; this.playing = false; }
    return value;
  }
  play(): void {
    this.position = this.read();
    if (this.position >= this.durationMs) this.position = 0;
    this.anchor = this.now();
    this.playing = this.durationMs > 0;
  }
  pause(): void { this.position = this.read(); this.playing = false; }
  seek(positionMs: number): void {
    if (!finite(positionMs)) return;
    this.position = Math.min(this.durationMs, Math.max(0, positionMs));
    this.anchor = this.now();
    if (this.position >= this.durationMs) this.playing = false;
  }
}
