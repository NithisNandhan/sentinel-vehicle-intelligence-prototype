import { describe, expect, it } from 'vitest';
import type { RecognitionEvidence, RecognitionStatus } from './api';
import { imageData } from './Inspector';
import {
  failureMessage, formatVideoTime, framePlaceholderTitle, retainEvidence, processingHeading, recognitionCount, recognitionIsFresh, recognitionIsRunning,
  recognitionLabel, recognitionTone, sessionLabel, sourceDescription, sourceLabel, staleLimit,
  vehicleEvidence, vehicleIsInspectable, vehicleReadingLabel,
} from './recognition';

const status: RecognitionStatus = {
  schema_version: '1.0', status: 'live', camera_id: 'cam17', source_mode: 'live',
  recognition_state: 'uncertain', frame_age_seconds: 0.4, tracks: [],
};
const recording: RecognitionStatus = {
  schema_version: '1.0', status: 'processing', camera_id: 'rec-0123456789ab', source_mode: 'recorded',
  recognition_state: 'reading', frame_age_seconds: 2, stale_after_seconds: 30, tracks: [],
  session_id: 's-1', recording_name: 'Junction A', video_position_ms: 1500, video_duration_ms: 6000,
};

describe('operator recognition presentation', () => {
  it('never describes a model acceptance as a verified plate', () => {
    expect(recognitionLabel('model_accepted')).toBe('Model-accepted · unverified');
    expect(recognitionTone('model_accepted')).toBe('warn');
    expect(recognitionTone('uncertain')).toBe('warn');
  });
  it('keeps missing or unknown states unavailable instead of confirmed', () => {
    expect(recognitionLabel('unreadable')).toBe('Unreadable');
    expect(recognitionLabel('unknown')).toBe('Unavailable');
  });
  it('only shows current tracks when the live frame age is known and at most five seconds', () => {
    expect(recognitionIsFresh(status)).toBe(true);
    expect(recognitionIsFresh({ ...status, frame_age_seconds: 5 })).toBe(true);
    for (const age of [undefined, null, -1, 5.01, Number.NaN, Infinity]) {
      expect(recognitionIsFresh({ ...status, frame_age_seconds: age })).toBe(false);
    }
    expect(recognitionIsFresh({ ...status, status: 'connecting' })).toBe(false);
    expect(recognitionIsFresh(null)).toBe(false);
  });
  it('calls an idle session live only while the opt-in neutral label is off', () => {
    const idle: RecognitionStatus = { ...status, status: 'paused', camera_id: null };
    expect(processingHeading(idle, false)).toBe('SHARED LIVE PROCESSING');
    expect(processingHeading(idle, true)).toBe('SHARED PROCESSING · NO SOURCE SELECTED');
    expect(processingHeading(status, true)).toBe('SHARED LIVE PROCESSING');
    expect(processingHeading(recording, true)).toBe('SHARED RECORDED PROCESSING');
    expect(processingHeading(recording, false)).toBe('SHARED RECORDED PROCESSING');
    expect(processingHeading(null)).toBe('SHARED LIVE PROCESSING');  // Off unless VITE_NEUTRAL_IDLE_LABEL=true.
  });

  it('never presents recorded footage as a live source', () => {
    expect(sourceLabel('recorded')).toBe('RECORDED');
    expect(sourceLabel('synthetic')).toBe('SIMULATED');
    expect(sourceLabel('live')).toBe('LIVE SOURCE');
    expect(sourceDescription('recorded')).toBe('Recorded footage');
    expect(sessionLabel({ ...recording, status: 'processing' })).toBe('Analysing recording');
    expect(sessionLabel({ ...recording, status: 'completed' })).toBe('Recording finished');
    expect(sessionLabel({ ...recording, status: 'paused' })).toBe('Recording paused');
    expect(sessionLabel({ ...status, status: 'paused', camera_id: null })).toBe('Paused');
    expect(sessionLabel(status)).toBe('Live analysis');
    expect(recognitionLabel('completed')).toBe('Completed');
  });
  it('treats a recording as current while analysing, paused or finished, within its own limit', () => {
    for (const state of ['processing', 'paused', 'completed'] as const) {
      expect(recognitionIsFresh({ ...recording, status: state })).toBe(true);
    }
    expect(recognitionIsFresh({ ...recording, frame_age_seconds: 12 })).toBe(true);   // slow CPU, still current
    expect(recognitionIsFresh({ ...recording, frame_age_seconds: 30.1 })).toBe(false);
    expect(recognitionIsFresh({ ...recording, status: 'connecting' })).toBe(false);
    expect(recognitionIsFresh({ ...recording, status: 'unavailable' })).toBe(false);
    expect(staleLimit(recording)).toBe(30);
    expect(staleLimit({ ...recording, stale_after_seconds: 99999 })).toBe(5);
    // A camera can never borrow the longer recorded limit or the recorded states.
    expect(staleLimit({ ...status, stale_after_seconds: 30 })).toBe(5);
    expect(recognitionIsFresh({ ...status, stale_after_seconds: 30, frame_age_seconds: 12 })).toBe(false);
    expect(recognitionIsFresh({ ...status, status: 'processing' })).toBe(false);
  });
  it('keeps a held recording alive but not a stopped or failed session', () => {
    for (const state of ['connecting', 'processing', 'paused', 'completed'] as const) {
      expect(recognitionIsRunning({ ...recording, status: state })).toBe(true);
    }
    expect(recognitionIsRunning({ ...recording, status: 'unavailable' })).toBe(false);
    expect(recognitionIsRunning({ ...status, status: 'paused', camera_id: null })).toBe(false);
    expect(recognitionIsRunning(status)).toBe(true);
    expect(recognitionIsRunning(null)).toBe(false);
  });
  it('shows video time without inventing a position', () => {
    expect(formatVideoTime(0)).toBe('0:00');
    expect(formatVideoTime(65_400)).toBe('1:05');
    expect(formatVideoTime(3_723_000)).toBe('1:02:03');
    for (const value of [undefined, null, -1, Number.NaN, Infinity]) expect(formatVideoTime(value)).toBe('—');
  });
  it('explains failure codes in plain words and never echoes unknown text as fact', () => {
    expect(failureMessage('ChangedRecording')).toContain('permission receipt');
    expect(failureMessage('MissingPts')).toContain('timestamps');
    expect(failureMessage('SomethingNew')).toBe('Analysis stopped (SomethingNew).');
    expect(failureMessage(undefined)).toBe('');
  });
  it('keeps unavailable counts distinct from measured zero', () => {
    expect(recognitionCount(0)).toBe('0');
    expect(recognitionCount(undefined)).toBe('—');
    expect(recognitionCount(-1)).toBe('—');
    expect(recognitionCount(Number.NaN)).toBe('—');
  });
});

describe('frame retention while evidence polls fail', () => {
  const evidence = (sessionId: string, frame: string | null = 'AAAA'): RecognitionEvidence => ({
    source_mode: 'recorded', session_id: sessionId, video_position_ms: 1200, frame, vehicles: [],
  } as RecognitionEvidence);
  const processing: RecognitionStatus = { ...recording, status: 'processing', session_id: 'session-a' };

  it('keeps the displayed frame when a later poll fails or returns nothing for the same session', () => {
    const shown = retainEvidence(null, evidence('session-a'), 'session-a');
    expect(imageData(shown?.frame, 'jpeg')).toBeDefined();
    for (const gap of [undefined, null]) {
      const kept = retainEvidence(shown, gap, 'session-a');
      expect(kept).toBe(shown);
      expect(imageData(kept?.frame, 'jpeg')).toBe(imageData(shown?.frame, 'jpeg'));
    }
  });

  it('replaces the frame when a newer poll succeeds', () => {
    const next = evidence('session-a', 'BBBB');
    expect(retainEvidence(evidence('session-a'), next, 'session-a')).toBe(next);
  });

  it('never claims the frame is unavailable while a recording is being analysed', () => {
    for (const status of [processing, { ...processing, status: 'connecting' as const }, null]) {
      expect(framePlaceholderTitle(status as RecognitionStatus | null)).not.toMatch(/unavailable/i);
    }
    expect(framePlaceholderTitle(processing)).toBe('Waiting for analysis evidence');
  });

  it('clears a stale frame when the session changes', () => {
    const old = evidence('session-a');
    expect(retainEvidence(old, undefined, 'session-b')).toBeNull();
    expect(retainEvidence(old, null, 'session-b')).toBeNull();
    expect(retainEvidence(old, evidence('session-a', 'CCCC'), 'session-b')).toBeNull();
    const fresh = evidence('session-b');
    expect(retainEvidence(old, fresh, 'session-b')).toBe(fresh);
  });
});

describe('vehicle rows before and after a plate reading', () => {
  const row: NonNullable<RecognitionStatus['vehicles_seen']>[number] = {
    key: '0-12', track_id: '12', vehicle_class: 'car', confidence: .8,
    first_ms: 100, last_ms: 100, observations: 0, state: 'reading',
    masked_plate: null, watchlist: null,
  };
  const visual: RecognitionEvidence['vehicles'][number] = {
    ...row, reading: 'GJ01AB1234', thumb: null,
  };

  it('gives a newly tracked vehicle a visible placeholder immediately without waiting for OCR', () => {
    expect(vehicleEvidence(row, null)).toBeUndefined();
    expect(vehicleReadingLabel(row, undefined, 'processing')).toBe('Reading plate…');
    expect(row.observations).toBe(0);
    expect(row.masked_plate).toBeNull();
  });

  it('updates the same row when a reading arrives and keeps it visible after EOF', () => {
    const evidence: RecognitionEvidence = {
      source_mode: 'recorded', session_id: 's-1', video_position_ms: 500, frame: null, vehicles: [visual],
    };
    const matched = vehicleEvidence(row, evidence);
    expect(matched).toBe(visual);
    expect(vehicleReadingLabel(row, matched, 'processing')).toBe('GJ01AB1234');
    expect(vehicleReadingLabel(row, matched, 'completed')).toBe('GJ01AB1234');
    expect(row.key).toBe('0-12');
    expect(row.state).toBe('reading'); // Displaying text never changes acceptance.
  });

  it('keeps no-text vehicles at EOF with a clear missing-reading label', () => {
    expect(vehicleReadingLabel(row, undefined, 'completed')).toBe('No plate reading');
    expect(vehicleReadingLabel(row, undefined, 'paused')).toBe('Plate reading paused');
  });

  it('uses the existing masked reading if full recorded evidence is unavailable', () => {
    expect(vehicleReadingLabel({ ...row, masked_plate: 'GJ******34' }, undefined, 'completed')).toBe('GJ******34');
    expect(recognitionLabel('model_accepted')).toBe('Model-accepted · unverified');
  });

  it('uses the scene-aware row key so a reused track ID cannot borrow another scene reading', () => {
    const evidence: RecognitionEvidence = {
      source_mode: 'recorded', session_id: 's-1', video_position_ms: 500, frame: null,
      vehicles: [{ ...visual, key: '1-12', reading: 'OTHER' }, visual],
    };
    expect(vehicleEvidence(row, evidence)).toBe(visual);
    expect(vehicleEvidence({ ...row, key: '2-12' }, evidence)).toBeUndefined();
    expect(vehicleReadingLabel(row, vehicleEvidence({ ...row, key: '2-12' }, evidence), 'processing')).toBe('Reading plate…');
    const legacy = { ...visual, key: undefined };
    expect(vehicleEvidence(row, { ...evidence, vehicles: [legacy] })).toBe(legacy);
  });

  it('only opens the newest scene row for a track-ID-only Inspector', () => {
    const newest = { ...row, key: '1-12' };
    const otherTrack = { ...row, key: '0-13', track_id: '13' };
    const rows = [newest, otherTrack, row];
    expect(vehicleIsInspectable(newest, rows)).toBe(true);
    expect(vehicleIsInspectable(row, rows)).toBe(false);
    expect(vehicleIsInspectable(otherTrack, rows)).toBe(true);
    expect(vehicleIsInspectable(row, [])).toBe(false);
  });

  it('keeps legacy unique rows inspectable and disables older unkeyed duplicates', () => {
    const newest = { ...row, key: undefined };
    const older = { ...row, key: undefined };
    expect(vehicleIsInspectable(newest, [newest])).toBe(true);
    expect(vehicleIsInspectable(newest, [newest, older])).toBe(true);
    expect(vehicleIsInspectable(older, [newest, older])).toBe(false);
    expect(vehicleIsInspectable(row, [newest, row])).toBe(false);
  });
});
