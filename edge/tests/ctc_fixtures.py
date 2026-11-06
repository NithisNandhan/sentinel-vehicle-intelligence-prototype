"""SYNTHETIC CTC posteriors for unit and integration tests. Not real OCR output.

Each character is written as one emission row followed by a blank row, so
repeated characters stay separate. `spec` items are a character, or a tuple
(character, peak) or (character, peak, rival) where the remaining mass goes to
`rival` instead of the blank.
"""
TOKENS = ('', *'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', ' ')


def rows_for(spec, tokens=TOKENS, *, floor=1e-4):
    rows = []
    for item in spec:
        character, peak, rival = (item, .97, None) if isinstance(item, str) else (*item, None)[:3]
        row = [floor] * len(tokens)
        row[tokens.index(character)] = peak
        rest = 1. - peak - floor * (len(tokens) - 2)
        row[tokens.index(rival) if rival else 0] += max(0., rest)
        total = sum(row)
        rows.append([value / total for value in row])
        blank = [floor] * len(tokens)
        blank[0] = 1. - floor * (len(tokens) - 1)
        rows.append(blank)
    return rows


def reader_with(frames, tokens=TOKENS, **options):
    from sentinel_edge.ctc_fusion import CTCPlateReader
    reader = CTCPlateReader(tokens, **options)
    for index, spec in enumerate(frames):
        assert reader.add(rows_for(spec, tokens), timestamp_ms=index * 400, source_id=f'frame-{index}')
    return reader
