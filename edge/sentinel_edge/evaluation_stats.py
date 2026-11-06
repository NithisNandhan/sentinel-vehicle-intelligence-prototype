"""Small-sample statistics for evaluation reports. No dependencies.

Passage-level intervals assume independent passages. Passages from the same
recording are correlated (same camera, light, model state), so the true
uncertainty is wider than these intervals; reports must say so.
"""
from __future__ import annotations

import math

Z95 = 1.959963984540054


def wilson(successes: int, total: int, z: float = Z95):
    """Two-sided Wilson score interval, or (None, None) when total is 0."""
    if total <= 0:
        return None, None
    if not 0 <= successes <= total:
        raise ValueError('successes must lie in [0, total]')
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0., centre - half), min(1., centre + half)


def rule_of_three(total: int):
    """Approximate one-sided 95% upper bound on an error rate after zero errors in `total`."""
    return None if total <= 0 else min(1., 3 / total)


def exact_zero_upper(total: int, confidence: float = .95):
    """Exact (Clopper-Pearson) one-sided upper bound after zero events in `total`."""
    return None if total <= 0 else 1 - (1 - confidence) ** (1 / total)


def proportion(successes: int, total: int) -> dict:
    low, high = wilson(successes, total)
    return {'count': successes, 'of': total, 'rate': round(successes / total, 4) if total else None,
            'wilson_95': [round(low, 4), round(high, 4)] if total else None}


def error_bound(errors: int, total: int) -> dict:
    """Error rate with its upper bound; the rule of three when nothing went wrong."""
    result = proportion(errors, total)
    if total and errors == 0:
        result.update(rule_of_three_upper_95=round(rule_of_three(total), 4),
                      exact_upper_95=round(exact_zero_upper(total), 4),
                      samples_needed_for_0_1_percent_bound=3000)
    return result
