"""Bounded CTC sequence decoding/rescoring across distinct original frames.

CTC time positions are not character slots. Candidate sequence likelihoods are
computed per frame before combining; frames/views are never concatenated or
counted twice. Scores are uncalibrated, not correctness probabilities.
"""
from __future__ import annotations

import math
from collections import deque

from .temporal import PlateCandidate, indian_plate_format

NEG = float('-inf')


def logsum(*values):
    if len(values) == 2:
        a, b = values
        if a == NEG:
            return b
        if b == NEG:
            return a
        return max(a, b) + math.log1p(math.exp(-abs(a - b)))
    peak = max(values, default=NEG)
    return peak + math.log(sum(math.exp(v - peak) for v in values)) if peak != NEG else NEG


def validate_ctc(rows, tokens):
    if not tokens or tokens[0] != '' or len(set(tokens)) != len(tokens):
        raise ValueError('CTC requires blank at index zero and distinct tokens')
    if not 1 <= len(rows) <= 320:
        raise ValueError('CTC time dimension exceeds bounds')
    for row in rows:
        if (len(row) != len(tokens) or any(not math.isfinite(p) or p < 0 for p in row)
                or abs(sum(row) - 1) > 1e-4):
            raise ValueError('Invalid CTC posterior')


def greedy_text(rows, tokens):
    result, previous = [], None
    for row in rows:
        index = max(range(len(row)), key=row.__getitem__)
        if index != 0 and index != previous:
            result.append(tokens[index])
        previous = index
    return ''.join(result).replace(' ', '').upper()


def prefix_candidates(rows, tokens, *, beam_width=100, max_length=25):
    """Raw-token prefix beam: repeated characters need a blank or another token.

    Upper/lowercase merge only after CTC collapse, so `Aa` is not silently
    shortened to `A`. No grammar-based probability renormalisation occurs.
    """
    validate_ctc(rows, tokens)
    if not 1 <= beam_width <= 200 or not 1 <= max_length <= 25:
        raise ValueError('Invalid CTC beam bounds')
    # Scale forward probabilities each step to prevent underflow. Retain the
    # exact log scale; this is numerical stabilisation, not discarded mass or
    # a claim of greater confidence. Candidate rescoring is still exact CTC.
    beam = {(): (1., 0.)}
    log_scale = 0.
    for row in rows:
        choices = sorted(range(1, len(row)), key=row.__getitem__, reverse=True)[:8]
        expanded = {}
        def add(prefix, blank=0., nonblank=0.):
            if not blank and not nonblank:
                return
            b, n = expanded.get(prefix, (0., 0.))
            expanded[prefix] = (b + blank, n + nonblank)
        for prefix, (blank, nonblank) in beam.items():
            if row[0]:
                add(prefix, blank=(blank + nonblank) * row[0])
            for token in choices:
                if row[token] <= 0:
                    continue
                probability = row[token]
                repeat = bool(prefix and prefix[-1] == token)
                if repeat:
                    add(prefix, nonblank=nonblank * probability)
                if len(prefix) < max_length:
                    add(prefix + (token,), nonblank=(blank if repeat else blank + nonblank) * probability)
        selected = sorted(expanded.items(), key=lambda item: sum(item[1]), reverse=True)[:beam_width]
        if not selected:
            return []
        scale = sum(selected[0][1])
        log_scale += math.log(scale)
        beam = {prefix: (blank / scale, nonblank / scale) for prefix, (blank, nonblank) in selected}
    sequences = {}
    for prefix, scores in beam.items():
        text = ''.join(tokens[token] for token in prefix).replace(' ', '').upper()
        if text and text.isascii() and text.isalnum():
            sequences[text] = logsum(sequences.get(text, NEG), math.log(sum(scores)) + log_scale)
    return sorted(sequences.items(), key=lambda item: (-item[1], item[0]))[:200]


def sequence_score(rows, tokens, text):
    """Exact CTC likelihood for case-insensitive text, including space paths.

    Track the previous *raw* symbol as well as output length. This preserves
    lowercase/uppercase repeated-character semantics and all blank/space paths.
    Viterbi character peaks are diagnostics, not marginal character confidence.
    """
    if not text or len(text) > 25 or not text.isascii() or not text.isalnum() or text != text.upper():
        return NEG, ()
    by_character = {char: tuple(i for i, token in enumerate(tokens)
                               if i and token != ' ' and token.upper() == char) for char in set(text)}
    if any(not choices for choices in by_character.values()):
        return NEG, ()
    spaces = tuple(i for i, token in enumerate(tokens) if token == ' ')
    # State: (emitted length, last raw token), value: summed log-likelihood,
    # best-path log-likelihood, best-path character peak probabilities.
    states = {(0, 0): (0., 0., ())}
    for row in rows:
        nxt = {}
        def add(key, log_probability, best_probability, peaks):
            current = nxt.get(key)
            if current is None:
                nxt[key] = (log_probability, best_probability, peaks)
            else:
                best = (best_probability, peaks) if best_probability > current[1] else (current[1], current[2])
                nxt[key] = (logsum(current[0], log_probability), *best)
        for (length, previous), (probability, best, peaks) in states.items():
            for token in (0, *spaces):
                if row[token]:
                    value = math.log(row[token])
                    add((length, token), probability + value, best + value, peaks)
            if previous and previous not in spaces and row[previous] and length:
                value = math.log(row[previous])
                updated = peaks[:-1] + (max(peaks[-1], row[previous]),)
                add((length, previous), probability + value, best + value, updated)
            if length < len(text):
                for token in by_character[text[length]]:
                    if token == previous or not row[token]:
                        continue
                    value = math.log(row[token])
                    add((length + 1, token), probability + value, best + value, peaks + (row[token],))
        states = nxt
    final = [value for (length, _), value in states.items() if length == len(text)]
    return (logsum(*(v[0] for v in final)), max(final, key=lambda v: v[1])[2]) if final else (NEG, ())


class CTCPlateReader:
    def __init__(self, tokens, *, window=8, beam_width=100):
        if not 3 <= window <= 12 or not 1 <= beam_width <= 200:
            raise ValueError('Invalid CTC temporal bounds')
        self.tokens, self.beam_width = tuple(tokens), beam_width
        self.samples = deque(maxlen=window)
        self._cache = None
        self._peaks = {}
        self._proposals = ()

    @property
    def observation_count(self):
        return len(self.samples)

    def add(self, rows, *, timestamp_ms, source_id, quality=1.0):
        validate_ctc(rows, self.tokens)
        if (not isinstance(timestamp_ms, int) or isinstance(timestamp_ms, bool) or timestamp_ms < 0
                or not isinstance(source_id, str) or not source_id
                or not isinstance(quality, (int, float)) or isinstance(quality, bool)
                or not math.isfinite(quality) or not .1 <= quality <= 1.):
            raise ValueError('Distinct source identity and timestamp required')
        if any(sample[1] == source_id for sample in self.samples):
            return False
        if self.samples and timestamp_ms - self.samples[-1][0] < 150:
            return False
        candidates = prefix_candidates(rows, self.tokens, beam_width=self.beam_width)
        self.samples.append((timestamp_ms, source_id, rows, candidates, quality))
        self._cache = None
        return True

    def propose(self, candidates):
        """Derived-view suggestions only: never add observations or scores.

        All suggestions are rescored against the original-frame posteriors.
        Reserve at most 32 search slots, keeping the total search bounded.
        """
        values = tuple(dict.fromkeys(text for text in candidates
                                    if isinstance(text, str) and indian_plate_format(text)))[:32]
        if values != self._proposals:
            self._proposals = values
            self._cache = None

    def candidates(self):
        if self._cache is not None:
            return self._cache
        # Union of independently generated frame hypotheses. A candidate is
        # rescored against every frame, not only frames that happened to emit it.
        proposals = {}
        for _, _, _, candidates, _ in self.samples:
            for text, score in candidates:
                if indian_plate_format(text):
                    proposals[text] = max(proposals.get(text, NEG), score)
        originals = sorted(proposals, key=lambda text: -proposals[text])
        proposals = list(dict.fromkeys((*self._proposals, *originals)))[:200]
        result, self._peaks = [], {}
        for text in proposals:
            likelihoods, peaks = [], []
            weights = []
            for _, _, rows, _, quality in self.samples:
                score, character_peaks = sequence_score(rows, self.tokens, text)
                likelihoods.append(score)
                peaks.append(character_peaks)
                weights.append(quality)
            if not likelihoods or not all(math.isfinite(v) for v in likelihoods):
                continue
            # Relative image-quality weights, not extra observations or a
            # calibrated confidence boost. Default weights retain equal fusion.
            denominator = sum(weights)
            character_scores = [sum(weight * values[i] for weight, values in zip(weights, peaks))
                                / denominator for i in range(len(text))]
            result.append(PlateCandidate(text, sum(character_scores) / len(text), min(character_scores),
                                         sum(weight * value for weight, value in zip(weights, likelihoods))
                                         / denominator, len(self.samples),
                                         indian_plate_format(text)))
            self._peaks[text] = peaks
        self._cache = sorted(result, key=lambda candidate: (-candidate.log_score, candidate.plate))[:200]
        return self._cache

    def accepted(self, threshold=.85):
        candidates = self.candidates()
        if self.observation_count < 3 or not candidates:
            return None
        best = candidates[0]
        if best.score < threshold or best.minimum_character_score < .75:
            return None
        if len(candidates) > 1 and best.log_score - candidates[1].log_score < .5:
            return None
        if any(sum(values[i] >= .6 for values in self._peaks[best.plate]) < 2 for i in range(len(best.plate))):
            return None
        return best
