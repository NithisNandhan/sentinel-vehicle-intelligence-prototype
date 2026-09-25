"""Plate transformations shared by live and synthetic modes."""

from __future__ import annotations

import hashlib
import hmac
import re


def normalize_plate(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def mask_plate(value: str) -> str:
    normalized = normalize_plate(value)
    if not 5 <= len(normalized) <= 20:
        raise ValueError("Plate must contain 5 to 20 letters or digits")
    return normalized[:2] + "*" * (len(normalized) - 4) + normalized[-2:]


def plate_hmac(value: str, key: str) -> str:
    normalized = normalize_plate(value)
    if not 5 <= len(normalized) <= 20:
        raise ValueError("Plate must contain 5 to 20 letters or digits")
    if not key:
        raise ValueError("WATCHLIST_HMAC_KEY is required")
    return hmac.new(key.encode("utf-8"), normalized.encode("ascii"), hashlib.sha256).hexdigest()

