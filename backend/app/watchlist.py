"""Operator watchlist management; raw identifiers are never persisted or returned."""
import json
from datetime import timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .db import get_db
from .models import WatchlistAudit, WatchlistEntry, WatchlistRevision
from .security import check_operator_key, normalize_plate, plate_token
from .unified_contract import WatchlistCreate, WatchlistItem, WatchlistPage, WatchlistPatch, WatchlistSearch

router = APIRouter(prefix="/v1/watchlist", tags=["watchlist"], dependencies=[Depends(check_operator_key)])


def revision(session: Session) -> int:
    row = session.get(WatchlistRevision, 1)
    if row is None:
        version = session.scalar(select(func.max(WatchlistEntry.version))) or 1
        session.add(WatchlistRevision(id=1, version=version))
        session.flush()
        return version
    return row.version


def advance(session: Session) -> int:
    revision(session)
    return session.execute(update(WatchlistRevision).where(WatchlistRevision.id == 1)
                           .values(version=WatchlistRevision.version + 1).returning(WatchlistRevision.version)).scalar_one()


def public(entry: WatchlistEntry) -> WatchlistItem:
    stamp = entry.created_at
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return WatchlistItem(id=entry.id, masked_plate=entry.masked_plate, reason=entry.reason,
                         severity=entry.severity, active=entry.active, version=entry.version, created_at=stamp)


def page(session: Session, offset: int, limit: int, query: str = "") -> WatchlistPage:
    statement = select(WatchlistEntry)
    if query:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        filters = [WatchlistEntry.reason.ilike("%" + escaped + "%", escape="\\"),
                   WatchlistEntry.masked_plate.ilike("%" + escaped + "%", escape="\\")]
        try:
            filters.append(WatchlistEntry.plate_hmac == plate_token(query))
        except ValueError:
            pass
        statement = statement.where(or_(*filters))
    total = session.scalar(select(func.count()).select_from(statement.subquery())) or 0
    items = session.scalars(statement.order_by(WatchlistEntry.id.desc()).offset(offset).limit(limit)).all()
    return WatchlistPage(items=[public(item) for item in items], total=total, offset=offset,
                         limit=limit, version=revision(session))


@router.get("", response_model=WatchlistPage)
def list_entries(response: Response, offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100), session: Session = Depends(get_db)):
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return page(session, offset, limit)


@router.post("/search", response_model=WatchlistPage)
def search_entries(body: WatchlistSearch, response: Response, session: Session = Depends(get_db)):
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return page(session, body.offset, body.limit, body.query.strip())


@router.post("", response_model=WatchlistItem, status_code=201)
def create_entry(body: WatchlistCreate, response: Response, session: Session = Depends(get_db)):
    try:
        plate = normalize_plate(body.plate)
    except ValueError:
        raise HTTPException(status_code=422, detail="Plate must contain 5 to 20 letters or digits") from None
    reason = body.reason.strip()
    if not reason:
        raise HTTPException(status_code=422, detail="A watchlist reason is required")
    token = plate_token(plate)
    if session.scalar(select(WatchlistEntry.id).where(WatchlistEntry.plate_hmac == token)) is not None:
        raise HTTPException(status_code=409, detail="This watchlist identifier already exists")
    try:
        version = advance(session)
        entry = WatchlistEntry(plate_hmac=token, masked_plate=plate[:2] + "*" * (len(plate) - 4) + plate[-2:],
                              reason=reason, severity=body.severity, active=True, version=version)
        session.add(entry)
        session.flush()
        session.add(WatchlistAudit(entry_id=entry.id, revision=version, action="create",
                                  changes=json.dumps({"reason": reason, "severity": body.severity, "active": True})))
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(status_code=409, detail="Watchlist changed; refresh and retry") from None
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return public(entry)


@router.patch("/{entry_id}", response_model=WatchlistItem)
def change_entry(entry_id: int, body: WatchlistPatch, response: Response, session: Session = Depends(get_db)):
    entry = session.get(WatchlistEntry, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Watchlist entry not found")
    if entry.version != body.expected_version:
        raise HTTPException(status_code=409, detail="Watchlist entry changed; refresh before editing")
    changes = {key: value for key, value in body.model_dump(exclude_unset=True).items() if key != "expected_version"}
    if any(value is None for value in changes.values()):
        raise HTTPException(status_code=422, detail="Watchlist changes cannot be null")
    if "reason" in changes:
        changes["reason"] = changes["reason"].strip()
        if not changes["reason"]:
            raise HTTPException(status_code=422, detail="A watchlist reason is required")
    changes = {key: value for key, value in changes.items() if getattr(entry, key) != value}
    if changes:
        version = advance(session)
        result = session.execute(update(WatchlistEntry).where(WatchlistEntry.id == entry_id,
                                WatchlistEntry.version == body.expected_version).values(**changes, version=version))
        if result.rowcount != 1:
            session.rollback()
            raise HTTPException(status_code=409, detail="Watchlist entry changed; refresh before editing")
        action = "update" if "active" not in changes else ("reactivate" if changes["active"] else "deactivate")
        session.add(WatchlistAudit(entry_id=entry_id, revision=version, action=action, changes=json.dumps(changes)))
        session.commit()
        session.refresh(entry)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return public(entry)
