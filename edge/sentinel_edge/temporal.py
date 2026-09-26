"""Bounded fusion of pretrained OCR probabilities; scores are NOT calibrated.

Only probability tensors/text are retained in RAM. No images or plaintext
readings are written or logged. Format filtering never renormalizes scores.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass


def indian_plate_format(text: str) -> str | None:
    """Broad common-format hints, not an exhaustive legal registration validator."""
    if re.fullmatch(r"\d{2}BH\d{4}[A-Z]{1,2}", text):
        return "bharat"
    if 7 <= len(text) <= 12 and re.fullmatch(r"[A-Z]{2}\d{1,3}[A-Z]{0,3}\d{1,4}", text):
        return "state"
    return None


@dataclass(frozen=True, repr=False)
class PlateCandidate:
    plate: str
    score: float
    minimum_character_score: float
    log_score: float
    observations: int
    format_name: str


class TemporalPlateReader:
    def __init__(self, alphabet: str, pad_char: str, *, window: int = 12,
                 beam_width: int = 200, min_observations: int = 3):
        if len(set(alphabet)) != len(alphabet) or pad_char not in alphabet:
            raise ValueError("Invalid OCR alphabet")
        if not 1 <= min_observations <= window <= 40 or not 1 <= beam_width <= 200:
            raise ValueError("Invalid temporal evidence bounds")
        self.alphabet = alphabet
        self.pad_char = pad_char
        self.beam_width = beam_width
        self.min_observations = min_observations
        self.samples = deque(maxlen=window)
        self._cache: list[PlateCandidate] | None = None

    @property
    def observation_count(self) -> int:
        return len(self.samples)

    def reset(self) -> None:
        self.samples.clear()
        self._cache = None

    def add(self, matrix, *, timestamp_ms: int, quality: float = 1.0) -> bool:
        if self.samples and timestamp_ms - self.samples[-1][0] < 150:
            return False
        if not matrix or (self.samples and len(matrix) != len(self.samples[-1][2])):
            raise ValueError("Inconsistent OCR slots")
        normalized = []
        for row in matrix:
            values = [float(value) for value in row]
            if len(values) != len(self.alphabet) or any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("Invalid OCR probabilities")
            total = sum(values)
            if total <= 0:
                raise ValueError("Empty OCR probability distribution")
            normalized.append([v / total for v in values])
        if not math.isfinite(quality) or quality <= 0:
            raise ValueError("Invalid observation quality")
        self.samples.append((timestamp_ms, max(0.1, min(quality, 1.0)), normalized))
        self._cache = None
        return True

    def candidates(self) -> list[PlateCandidate]:
        if self._cache is not None:
            return self._cache
        if not self.samples:
            return []
        total_weight = sum(sample[1] for sample in self.samples)
        slots = len(self.samples[0][2])
        fused = [[sum(weight * matrix[position][token] for _, weight, matrix in self.samples) / total_weight
                  for token in range(len(self.alphabet))] for position in range(slots)]
        # Fixed-position CCT outputs, not CTC tokens. Padding can only trail text.
        beam = [("", 0.0, ())]
        pad_index = self.alphabet.index(self.pad_char)
        for row in fused:
            choices = sorted(range(len(row)), key=row.__getitem__, reverse=True)[:4]
            if pad_index not in choices:
                choices.append(pad_index)
            expanded = []
            for text, log_score, selected in beam:
                for token in choices:
                    character = self.alphabet[token]
                    if self.pad_char in text and character != self.pad_char:
                        continue
                    if character != self.pad_char and not ("A" <= character <= "Z" or "0" <= character <= "9"):
                        continue
                    probability = row[token]
                    if probability <= 0:
                        continue
                    expanded.append((text + character, log_score + math.log(probability), selected + (probability,)))
            beam = sorted(expanded, key=lambda item: item[1], reverse=True)[:self.beam_width]
        candidates = []
        for text, log_score, selected in beam:
            plate = text.rstrip(self.pad_char)
            format_name = indian_plate_format(plate)
            if format_name is None:
                continue
            character_scores = selected[:len(plate)]
            candidates.append(PlateCandidate(plate, sum(character_scores) / len(plate),
                                             min(character_scores), log_score,
                                             len(self.samples), format_name))
        self._cache = candidates
        return candidates

    def accepted(self, threshold: float) -> PlateCandidate | None:
        if len(self.samples) < self.min_observations:
            return None
        candidates = self.candidates()
        if not candidates:
            return None
        best = candidates[0]
        if best.score < threshold or best.minimum_character_score < 0.75:
            return None
        if len(candidates) > 1 and best.log_score - candidates[1].log_score < 0.5:
            return None
        # Every character must have support in at least two separated readings.
        for position, character in enumerate(best.plate):
            token = self.alphabet.index(character)
            if sum(matrix[position][token] >= 0.60 for _, _, matrix in self.samples) < 2:
                return None
        return best
