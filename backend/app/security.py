import hashlib
import hmac
import re

from fastapi import Header, HTTPException

from .config import get_settings


def normalize_plate(plate: str) -> str:
    result = re.sub(r"[^A-Z0-9]", "", plate.upper())
    if len(result) < 5 or len(result) > 20:
        raise ValueError("plate must contain 5 to 20 letters or digits")
    return result


def plate_token(plate: str) -> str:
    key = get_settings().watchlist_hmac_key.encode("utf-8")
    return hmac.new(key, normalize_plate(plate).encode("ascii"), hashlib.sha256).hexdigest()


def check_edge_key(x_edge_key: str | None = Header(default=None)) -> None:
    expected = get_settings().edge_api_key
    if not x_edge_key or not hmac.compare_digest(x_edge_key, expected):
        raise HTTPException(status_code=401, detail="invalid edge key")


def check_operator_key(x_operator_key: str | None = Header(default=None)) -> None:
    expected = get_settings().operator_api_key
    if not x_operator_key or not hmac.compare_digest(x_operator_key, expected):
        raise HTTPException(status_code=401, detail="invalid operator key")

