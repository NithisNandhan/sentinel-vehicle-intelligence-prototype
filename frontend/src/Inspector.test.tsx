import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { WithheldProposals } from './Inspector';

describe('withheld OCR proposals', () => {
  it('shows raw text and native evidence separately with an explicit exclusion notice', () => {
    const html = renderToStaticMarkup(<WithheldProposals proposals={[{
      pts_ms: 500, width: 100, height: 30, image: 'eA==', text: 'MH47',
      reason: 'Insufficient text evidence: no character reaches 0.6', used_for_fusion: false,
      characters: [{ character: 'M', score: .45, alternatives: [{ character: 'N', score: .4 }] }],
    }]} />);
    expect(html).toContain('Withheld proposals');
    expect(html).toContain('MH47');
    expect(html).toContain('data:image/png;base64,eA==');
    expect(html).toContain('no observations or reconstruction votes');
    expect(html).toContain('may still be a real plate');
    expect(html).toContain('0.450');
    expect(html).toContain('no character reaches 0.6');
  });

  it('handles missing preview and raw text without presenting a confirmed reading', () => {
    const html = renderToStaticMarkup(<WithheldProposals proposals={[{
      pts_ms: 500, width: 100, height: 30, image: null, text: '', used_for_fusion: false,
    }]} />);
    expect(html).toContain('No raw OCR text');
    expect(html).toContain('Native preview unavailable');
    expect(html).not.toContain('<img');
    expect(renderToStaticMarkup(<WithheldProposals proposals={[]} />)).toBe('');
  });
});
