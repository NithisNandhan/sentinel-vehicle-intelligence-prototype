"""Bounded geometric continuity heuristics; these do not prove plate identity."""
from dataclasses import dataclass
import math
from numbers import Real


def _finite(value):
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class PlateContinuityPolicy:
    max_gap_ms: int = 1000
    max_scale_change: float = 2.5
    max_relative_jump: float = 1.5

    def __post_init__(self):
        if (not isinstance(self.max_gap_ms, int) or isinstance(self.max_gap_ms, bool)
                or self.max_gap_ms < 0 or not _finite(self.max_scale_change)
                or self.max_scale_change < 1 or not _finite(self.max_relative_jump)
                or self.max_relative_jump < 0):
            raise ValueError('Invalid plate continuity bounds')


DEFAULT_POLICY = PlateContinuityPolicy()


def _geometry(box):
    try:
        if len(box) != 4 or not all(_finite(value) for value in box):
            return None
        x1, y1, x2, y2 = map(float, box)
    except (TypeError, ValueError, OverflowError):
        return None
    width, height = x2-x1, y2-y1
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        return None
    return width, height, x1+width/2, y1+height/2


def continuity_reason(previous_box, previous_pts_ms, current_box, pts_ms, policy=DEFAULT_POLICY):
    """Return a text-free reason to split, or None for a plausible continuation.

    Boxes are global native-pixel xyxy coordinates, timestamps are nonnegative
    integer video milliseconds. Both previous arguments may be None for a first
    observation. Scale bounds apply to each dimension in both directions. Gap
    and geometry boundaries are inclusive; timestamps must advance strictly.
    """
    if not isinstance(policy, PlateContinuityPolicy):
        raise ValueError('PlateContinuityPolicy required')
    if not isinstance(pts_ms, int) or isinstance(pts_ms, bool) or pts_ms < 0:
        return 'invalid_timestamp'
    current = _geometry(current_box)
    if current is None:
        return 'invalid_box'
    if previous_box is None and previous_pts_ms is None:
        return None
    if not isinstance(previous_pts_ms, int) or isinstance(previous_pts_ms, bool) or previous_pts_ms < 0:
        return 'invalid_timestamp'
    previous = _geometry(previous_box)
    if previous is None:
        return 'invalid_box'
    if pts_ms <= previous_pts_ms:
        return 'non_forward_timestamp'
    if pts_ms-previous_pts_ms > policy.max_gap_ms:
        return 'gap'
    pw, ph, px, py = previous
    cw, ch, cx, cy = current
    if max(pw/cw, cw/pw, ph/ch, ch/ph) > policy.max_scale_change:
        return 'scale_change'
    diagonal = max(math.hypot(pw, ph), math.hypot(cw, ch))
    if math.hypot(cx-px, cy-py)/diagonal > policy.max_relative_jump:
        return 'relative_jump'
    return None
