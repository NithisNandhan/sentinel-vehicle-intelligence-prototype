import { beforeEach, describe, expect, it, vi } from 'vitest';
import { confirmAgrees, decideReview, getReviewEvidence, getReviewQueue, normalisePlate, patternCells } from './review';
import { resetLocalSession } from './api';

const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
const session = { csrf_token: 'csrf', expires_at: '2037-01-01T00:00:00Z' };

beforeEach(() => { vi.restoreAllMocks(); resetLocalSession(); });

describe('review helpers (synthetic plates)', () => {
  const item = { masked_plate: 'MH******57', unknown_positions: [5], plate_length: 10 };
  it('shows only the central mask without evidence, and unknown positions explicitly', () => {
    const cells = patternCells(item);
    expect(cells.map(cell => cell.state)).toEqual(['known', 'known', 'hidden', 'hidden', 'hidden', 'unknown', 'hidden', 'hidden', 'known', 'known']);
    expect(patternCells(item, 'MH49B?5557').map(cell => cell.character).join('')).toBe('MH49B?5557');
  });
  it('confirm must agree with every known position; otherwise correct is required', () => {
    expect(confirmAgrees('mh49-bf5557', item, 'MH49B?5557')).toBe(true);
    expect(confirmAgrees('MH49BF5587', item, 'MH49B?5557')).toBe(false);
    expect(confirmAgrees('MH49BF557', item)).toBe(false);
    expect(confirmAgrees('KA49BF5557', item)).toBe(false);
    expect(normalisePlate(' mh 49 ')).toBe('MH49');
  });
});

describe('reviewer access', () => {
  it('sends the reviewer token and CSRF token, never stores the token', async () => {
    const fetcher = vi.fn(async (path: RequestInfo | URL, init?: RequestInit) => String(path) === '/v1/local/session' ? json(session) : json({ reviewer: 'asha', pending: 0, items: [], result: {}, alert: null, _init: init }));
    vi.stubGlobal('fetch', fetcher);
    await getReviewQueue('secret-token');
    await decideReview('secret-token', 4, 'confirm', 'MH49BF5557');
    const calls = fetcher.mock.calls.filter(call => String(call[0]) !== '/v1/local/session');
    expect(String(calls[0][0])).toBe('/v1/review/queue?status=pending');
    const headers = new Headers(calls[1][1]?.headers);
    expect(headers.get('X-Reviewer-Token')).toBe('secret-token');
    expect(headers.get('X-CSRF-Token')).toBe('csrf');
    expect(JSON.parse(String(calls[1][1]?.body))).toEqual({ action: 'confirm', plate: 'MH49BF5557' });
  });
  it('maps rejected tokens and missing evidence', async () => {
    vi.stubGlobal('fetch', vi.fn(async (path: RequestInfo | URL) => String(path) === '/v1/local/session' ? json(session) : json({ detail: 'Reviewer access required' }, 403)));
    await expect(getReviewQueue('wrong')).rejects.toThrow('Reviewer access was rejected');
    await expect(getReviewQueue('')).rejects.toThrow('Enter your reviewer token');
    vi.stubGlobal('fetch', vi.fn(async (path: RequestInfo | URL) => String(path) === '/v1/local/session' ? json(session) : json({ detail: 'expired' }, 404)));
    expect(await getReviewEvidence('token', 3)).toBeNull();
  });
});
