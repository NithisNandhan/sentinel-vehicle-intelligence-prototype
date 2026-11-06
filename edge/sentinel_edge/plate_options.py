"""Plate-pipeline settings and the two named operational profiles.

`SENTINEL_PROFILE` selects a named bundle:

- `baseline` (default): the current approved behaviour. Every option below is off.
- `candidate`: the integrated Pipeline B behaviour on CPU: result states,
  plate-level tracking, near-duplicate guard and adaptive sampling.
- `candidate-gpu`: `candidate` plus multi-source plate proposals (native tiles and
  the text-region detector), which only fit the per-frame budget on a GPU.

An individual variable set to `true` or `false` overrides its profile default, for
diagnosis and rollback. Nothing here changes the 3-frame fusion rule or any
acceptance threshold; result states only add blockers to the existing gate.
"""
import os
from dataclasses import dataclass, fields

PROFILES = ('baseline', 'candidate', 'candidate-gpu')

_CANDIDATE = dict(result_states=True, plate_level_tracking=True, near_duplicate_guard=True,
                  adaptive_sampling=True)
PROFILE_DEFAULTS = {
    'baseline': {},
    'candidate': _CANDIDATE,
    'candidate-gpu': dict(_CANDIDATE, multi_source_plates=True),
}

ENVIRONMENT = {
    'reject_non_plates': 'SENTINEL_REJECT_NON_PLATES',
    'two_line': 'SENTINEL_TWO_LINE_PLATES',
    'format_validator': 'SENTINEL_PLATE_FORMAT_VALIDATOR',
    'position_vote': 'SENTINEL_POSITION_VOTE',
    'result_states': 'SENTINEL_RESULT_STATES',
    'plate_level_tracking': 'SENTINEL_PLATE_LEVEL_TRACKING',
    'near_duplicate_guard': 'SENTINEL_NEAR_DUPLICATE_GUARD',
    'adaptive_sampling': 'SENTINEL_ADAPTIVE_SAMPLING',
    'multi_source_plates': 'SENTINEL_MULTI_SOURCE_PLATES',
    'secondary_veto': 'SENTINEL_SECONDARY_VETO',
    'partial_watchlist': 'SENTINEL_PARTIAL_WATCHLIST',
}


@dataclass(frozen=True)
class PlateOptions:
    # Drop plate proposals that look like blank surfaces before they become fusion evidence.
    reject_non_plates: bool = False
    # Read two-line plates row by row instead of squeezing both rows into one line.
    two_line: bool = False
    # Admit only readings that fit the Indian registration format (plate_format.py),
    # fixing look-alike characters only where the letter/digit position decides it.
    # Such readings are marked format_adjusted and can never be CONFIRMED.
    format_validator: bool = False
    # Add the per-position, confidence-weighted vote of the frames' readings as a candidate.
    position_vote: bool = False
    # Emit one confirmed / partial / unreadable result per vehicle passage (result_state.py).
    result_states: bool = False
    # Carry a vehicle's plate evidence across a tracker-ID change when exactly one
    # recently lost track is geometrically consistent with the new one.
    plate_level_tracking: bool = False
    # Never count a nearly identical crop (a parked or stalled vehicle) as a new frame.
    near_duplicate_guard: bool = False
    # Sample every 250 ms only while a visible vehicle still lacks enough evidence.
    adaptive_sampling: bool = False
    # Pool plate proposals from the vehicle-crop detector, native tiles and text regions.
    multi_source_plates: bool = False
    # Veto-only secondary OCR (disabled until evaluation supports enabling it).
    secondary_veto: bool = False
    # Edge-side partial watchlist leads (review only, never a confirmed alert).
    partial_watchlist: bool = False
    profile: str = 'baseline'

    @classmethod
    def from_env(cls, environ=None):
        environ = os.environ if environ is None else environ
        profile = environ.get('SENTINEL_PROFILE', 'baseline').strip().lower() or 'baseline'
        if profile not in PROFILES:
            raise ValueError('SENTINEL_PROFILE must be one of: ' + ', '.join(PROFILES))
        values = dict(PROFILE_DEFAULTS[profile])
        for name, variable in ENVIRONMENT.items():
            raw = environ.get(variable, '').strip().lower()
            # Only exact `true`/`false` override; any other value keeps the profile
            # default (backward compatible with the earlier `== 'true'` rule).
            if raw in ('true', 'false'):
                values[name] = raw == 'true'
        return cls(profile=profile, **values)

    def reader_options(self):
        """Keyword arguments for CTCPlateReader."""
        return dict(validate_format=self.format_validator, position_vote=self.position_vote)

    def manifest(self):
        """Every effective setting, for evaluation reports and the candidate-profile record."""
        return {field.name: getattr(self, field.name) for field in fields(self)}
