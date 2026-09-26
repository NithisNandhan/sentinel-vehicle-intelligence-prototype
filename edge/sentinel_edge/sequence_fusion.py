"""Insertion/deletion-aware fusion. Missing characters are not confirming votes."""
from __future__ import annotations

import math
from collections import deque

from .temporal import TemporalPlateReader


def token_alignment(reference, observed):
    """Global edit alignment; maps reference positions to observed positions/None."""
    n, m = len(reference), len(observed)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        costs[i][0] = i
    for j in range(m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs[i][j] = min(costs[i - 1][j - 1] + (reference[i - 1] != observed[j - 1]),
                              costs[i - 1][j] + 1, costs[i][j - 1] + 1)
    result, i, j = [None] * n, n, m
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != observed[j - 1]):
            result[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return result, costs[n][m]


class SequencePlateReader:
    def __init__(self, alphabet, pad_char, *, slots=13, window=8):
        if '?' not in alphabet or not 8 <= slots <= 26 or not 3 <= window <= 12:
            raise ValueError('Sequence fusion requires explicit unknown token and bounded capacity')
        self.alphabet, self.pad_char, self.slots = alphabet, pad_char, slots
        self.samples = deque(maxlen=window)
        self._reader = None

    @property
    def observation_count(self):
        return len(self.samples)

    def reset(self):
        self.samples.clear()
        self._reader = None

    def add(self, matrix, *, timestamp_ms, quality=1.):
        # Validate without prematurely requiring equal per-frame sequence lengths.
        validator = TemporalPlateReader(self.alphabet, self.pad_char)
        validator.add(matrix, timestamp_ms=timestamp_ms, quality=quality)
        if len(matrix) > self.slots or not isinstance(timestamp_ms, int) or timestamp_ms < 0:
            raise ValueError('Invalid sequence observation')
        if self.samples and timestamp_ms - self.samples[-1][0] < 150:
            return False
        normalized = validator.samples[0][2]
        indices = [max(range(len(row)), key=row.__getitem__) for row in normalized]
        text = ''.join(self.alphabet[i] for i in indices).split(self.pad_char, 1)[0]
        if len(text) >= self.slots:
            return False  # Capacity includes an observed EOS, never invent it.
        self.samples.append((timestamp_ms, max(.1, min(quality, 1.)), normalized, text))
        self._reader = None
        return True

    def _aligned_reader(self):
        if self._reader is not None:
            return self._reader
        reader = TemporalPlateReader(self.alphabet, self.pad_char, window=8)
        if not self.samples:
            return reader
        # Medoid minimises edit disagreement. Tie favours the better-supported
        # length, then higher visual score; no target labels or registry involved.
        texts = [s[3] for s in self.samples]
        def key(sample):
            _, weight, matrix, text = sample
            score = sum(max(row) for row in matrix[:len(text)]) / max(1, len(text))
            return (sum(token_alignment(text, other)[1] for other in texts),
                    -sum(len(other) == len(text) for other in texts), -score * weight)
        reference = min(self.samples, key=key)[3]
        if not reference:
            self._reader = reader
            return reader
        unknown = [0.] * len(self.alphabet)
        unknown[self.alphabet.index('?')] = 1.
        for pts, quality, matrix, text in self.samples:
            mapping, edits = token_alignment(reference, text)
            if edits > max(2, math.ceil(len(reference) * .3)):
                continue
            aligned = [matrix[j] if j is not None else unknown[:] for j in mapping]
            # EOS must retain its actual probability. A length mismatch cannot
            # count as repeated confirmation of where the text ends.
            end = matrix[len(text)] if len(text) == len(reference) and len(text) < len(matrix) else unknown[:]
            aligned.append(end)
            for position in range(len(aligned), self.slots):
                source = len(text) + position - len(reference)
                aligned.append(matrix[source] if len(text) == len(reference) and source < len(matrix) else unknown[:])
            reader.add(aligned, timestamp_ms=pts, quality=quality)
        self._reader = reader
        return reader

    def candidates(self):
        return self._aligned_reader().candidates()

    def accepted(self, threshold):
        return self._aligned_reader().accepted(threshold)

    def partial(self, threshold=.6):
        reader = self._aligned_reader()
        if not reader.samples:
            return ''
        total = sum(weight for _, weight, _ in reader.samples)
        result = []
        for position in range(self.slots):
            row = [sum(weight * matrix[position][i] for _, weight, matrix in reader.samples) / total
                   for i in range(len(self.alphabet))]
            index = max(range(len(row)), key=row.__getitem__)
            char = self.alphabet[index]
            if char == self.pad_char and row[index] >= .75:
                break
            support = sum(matrix[position][index] >= threshold for _, _, matrix in reader.samples)
            result.append(char if char.isascii() and char.isalnum() and row[index] >= threshold and support >= 2 else '?')
        return ''.join(result)
