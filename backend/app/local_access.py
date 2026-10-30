"""No-prompt sessions for the explicit trusted local-browser profile."""
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, Response

from .security import COOKIE_NAME, SESSION_SECONDS, create_local_session, local_request_allowed
from .unified_contract import LocalSessionOut

router = APIRouter(prefix="/v1/local", tags=["local-access"])


@router.post("/session", response_model=LocalSessionOut)
def bootstrap(request: Request, response: Response):
    if not local_request_allowed(request, require_origin=True):
        raise HTTPException(status_code=403, detail="Local dashboard access is not configured for this origin")
    token, csrf, expiry = create_local_session()
    response.set_cookie(COOKIE_NAME, token, max_age=SESSION_SECONDS, httponly=True,
                        secure=request.url.scheme == "https", samesite="strict", path="/v1")
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return LocalSessionOut(csrf_token=csrf, expires_at=datetime.fromtimestamp(expiry, timezone.utc))
