"""Opt-in Indian registration format check (SENTINEL_PLATE_FORMAT_VALIDATOR, off by default).

Stricter than `temporal.indian_plate_format`, which stays the default broad hint:

- state series: state code from STATE_CODES, district 01-99 (Delhi also 1-9), 0-3 series
  letters, number 1-4 digits and not all zeros;
- Bharat series: year 21 or later, `BH`, four digits, one or two letters.

A reading that already fits is never changed. One that does not fit may have its
look-alike characters swapped (O/0, I/1, B/8, S/5, Z/2, G/6), but only when exactly one
valid plate needs the fewest swaps; that is, when the letter/digit position decides it.
Anything else is `invalid` (no valid plate at all) or `ambiguous` (several equally small
fixes): the reading is dropped, never guessed. This is a format check, not a registry:
a valid result can still be the wrong vehicle, and rare formats (diplomatic, military,
temporary, trade) are deliberately not covered.
"""
from __future__ import annotations

import functools
import itertools
import re
from dataclasses import dataclass

from .temporal import indian_plate_format

# Current RTO state/UT codes plus legacy codes still seen on older plates:
# DN (Dadra & Nagar Haveli, merged into DD in 2020), OR (Odisha before OD, 2012),
# UA (Uttarakhand before UK, 2007), TS (Telangana before TG, 2024).
STATE_CODES = frozenset((
    'AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR '
    'PB PY RJ SK TG TN TR TS UA UK UP WB').split())

# Each pair is a letter and the digit OCR most often confuses it with.
CONFUSABLE = {'O': '0', '0': 'O', 'I': '1', '1': 'I', 'B': '8', '8': 'B',
              'S': '5', '5': 'S', 'Z': '2', '2': 'Z', 'G': '6', '6': 'G'}
MAX_SWAPPABLE = 12      # 2**12 = 4,096 variants at most; plates are at most 12 characters.

_STATE = re.compile(r'([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{1,4})')
_BHARAT = re.compile(r'(\d{2})BH(\d{4})([A-Z]{1,2})')


@dataclass(frozen=True)
class FormatResult:
    status: str             # 'valid', 'corrected', 'invalid' or 'ambiguous'
    plate: str | None       # The plate to use; None unless valid or corrected.
    swaps: int = 0          # Characters changed (0 unless corrected).

    @property
    def usable(self):
        return self.plate is not None


def is_valid_plate(text: str) -> bool:
    """True when `text` (capitals and digits, no spaces) is a plausible Indian plate."""
    if not isinstance(text, str) or indian_plate_format(text) is None:
        return False
    state = _STATE.fullmatch(text)
    if state:
        code, district, _, number = state.groups()
        if code not in STATE_CODES or int(district) == 0 or int(number) == 0:
            return False
        return len(district) == 2 or code == 'DL'
    bharat = _BHARAT.fullmatch(text)
    return bool(bharat) and int(bharat.group(1)) >= 21 and int(bharat.group(2)) > 0


@functools.lru_cache(maxsize=8192)
def check_plate(text: str) -> FormatResult:
    """Validate a reading, fixing look-alike characters only when the position decides it."""
    text = (text or '').upper()
    if is_valid_plate(text):
        return FormatResult('valid', text)
    positions = [i for i, character in enumerate(text) if character in CONFUSABLE]
    if not text.isascii() or not text.isalnum() or not positions or len(positions) > MAX_SWAPPABLE:
        return FormatResult('invalid', None)
    best, found = None, []
    for count in range(1, len(positions) + 1):
        for chosen in itertools.combinations(positions, count):
            characters = list(text)
            for i in chosen:
                characters[i] = CONFUSABLE[characters[i]]
            candidate = ''.join(characters)
            if is_valid_plate(candidate):
                found.append(candidate)
        if found:
            best = count
            break
    if not found:
        return FormatResult('invalid', None)
    if len(found) > 1:
        return FormatResult('ambiguous', None)
    return FormatResult('corrected', found[0], best)


# ---------------------------------------------------------------------------
# Strict policy and partial patterns (result states, docs/FEATURE_DECISIONS.md)
#
# The automatic acceptance path never applies the look-alike fix above: a reading
# that would need a swap is reported as `lookalike` and goes to review instead.
# Patterns use '?' for an unknown position. Format rules may *narrow* which
# completions are possible (for bounded watchlist candidate search), but never
# choose a character that the camera did not support.
# ---------------------------------------------------------------------------

UNKNOWN = '?'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DIGITS = '0123456789'


def strict_status(text: str) -> str:
    """'valid', 'lookalike' (only a look-alike swap would fit: review, never fix),
    'ambiguous' (several swaps fit) or 'invalid'. Never changes the text."""
    result = check_plate(text)
    if result.status == 'valid':
        return 'valid'
    if result.status == 'corrected':
        return 'lookalike'
    return result.status


class TooManyCompletions(ValueError):
    pass


def _templates(length):
    from .temporal import _PLATE_PATTERNS
    return [pattern for pattern in _PLATE_PATTERNS if len(pattern) == length]


def completions(pattern: str, *, limit: int = 20000):
    """Every format-valid plate that agrees with `pattern` at its known positions.

    Raises TooManyCompletions above `limit` so callers cannot drift into an
    unbounded search. The known characters are never altered.
    """
    import itertools as _itertools
    if (not isinstance(pattern, str) or not pattern or not pattern.isascii()
            or any(not (c.isalnum() and c == c.upper()) and c != UNKNOWN for c in pattern)):
        raise ValueError('Pattern is capitals, digits and ? only')
    if not 1 <= limit <= 200000:
        raise ValueError('Bounded completion limit required')
    unknown = [i for i, c in enumerate(pattern) if c == UNKNOWN]
    if not unknown:
        return [pattern] if is_valid_plate(pattern) else []
    found = {}
    for template in _templates(len(pattern)):
        if any(c != UNKNOWN and not _fits(c, t) for c, t in zip(pattern, template)):
            continue
        choices = [LETTERS if template[i] == 'L' else DIGITS if template[i] == 'D' else template[i] for i in unknown]
        total = 1
        for option in choices:
            total *= len(option)
        if total > limit * 4:
            raise TooManyCompletions('Too many unknown positions to search safely')
        for fill in _itertools.product(*choices):
            characters = list(pattern)
            for i, c in zip(unknown, fill):
                characters[i] = c
            text = ''.join(characters)
            if text not in found and is_valid_plate(text):
                found[text] = None
                if len(found) > limit:
                    raise TooManyCompletions('Too many format-valid completions')
    return list(found)


def _fits(character, token):
    return ('A' <= character <= 'Z' if token == 'L' else
            '0' <= character <= '9' if token == 'D' else character == token)


def pattern_compatible(pattern: str) -> bool:
    """True when at least one valid Indian plate agrees with the known positions."""
    try:
        return bool(completions(pattern, limit=200000))
    except TooManyCompletions:
        return True
