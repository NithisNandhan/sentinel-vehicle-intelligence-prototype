import { describe, expect, it } from 'vitest';
import type { RecognitionStatus } from './api';
import { recognitionCount, recognitionIsFresh, recognitionLabel, recognitionTone } from './recognition';

const status: RecognitionStatus = {
  schema_version: '1.0', status: 'live', camera_id: 'cam17', source_mode: 'live',
  recognition_state: 'uncertain', frame_age_seconds: 0.4, tracks: [],
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
  it('keeps unavailable counts distinct from measured zero', () => {
    expect(recognitionCount(0)).toBe('0');
    expect(recognitionCount(undefined)).toBe('—');
    expect(recognitionCount(-1)).toBe('—');
    expect(recognitionCount(Number.NaN)).toBe('—');
  });
});
