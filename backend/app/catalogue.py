"""Authenticated internal proxy for catalogue search and preview control."""

import json
import urllib.error
import urllib.request

from fastapi import HTTPException

from .config import get_settings


def gateway_request(method: str, path: str, payload: dict | None = None) -> dict:
    settings = get_settings()
    if not settings.catalogue_gateway_url:
        raise HTTPException(status_code=503, detail="Sentinel catalogue unavailable")
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        settings.catalogue_gateway_url.rstrip("/") + path,
        data=body,
        method=method,
        headers={"X-Edge-Key": settings.edge_api_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        messages = {404: "Camera or preview not found",
                    409: "Camera is offline or its stream is unavailable", 422: "Invalid catalogue request",
                    429: "Two previews are already active", 503: "Sentinel catalogue unavailable"}
        status = error.code if error.code in messages else 503
        raise HTTPException(status_code=status, detail=messages[status]) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise HTTPException(status_code=503, detail="Sentinel catalogue unavailable") from None
