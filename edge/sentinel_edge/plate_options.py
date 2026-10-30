"""Opt-in plate-pipeline settings. Every option is off unless its variable is `true`.

None of them changes the 3-frame fusion rule or any acceptance threshold.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PlateOptions:
    # Drop plate proposals that look like blank surfaces before they become fusion evidence.
    reject_non_plates: bool = False
    # Read two-line plates row by row instead of squeezing both rows into one line.
    two_line: bool = False

    @classmethod
    def from_env(cls, environ=None):
        environ = os.environ if environ is None else environ
        flag = lambda name: environ.get(name, '').strip().lower() == 'true'
        return cls(reject_non_plates=flag('SENTINEL_REJECT_NON_PLATES'),
                   two_line=flag('SENTINEL_TWO_LINE_PLATES'))
