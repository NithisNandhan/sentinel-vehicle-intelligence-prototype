# Unified dashboard implementation

Approved design: `docs/superpowers/specs/2026-10-30-sentinel-unified-dashboard-design.md`, including Paper teal and no manual unlock.

The writing-plans skill is not installed in the available catalog. This task-local plan follows the approved design directly.

1. Integrator establishes shared DTOs in `backend/app/unified_contract.py`; generate JSON Schema and browser types using `tools/generate_unified_contract.py`.
2. Backend owner implements local sessions/CSRF and existing-key compatibility, watchlist API/migration/audits/revisions, diagnostic gateway and focused tests. Backend files only.
3. Worker owner implements bounded, recorded-only `/internal/diagnostics?session_id=...&track_id=...`, aligned evidence original/overlay selection, legacy browser redirects and tests. Edge files only; no inference/model/acceptance changes.
4. Frontend owner implements the Paper teal React shell, Analyse/Alerts/Watchlist and Inspector. Reuse existing status/session/evidence contracts and generated DTOs. Remove browser-held operator secrets and the separate model-dashboard link. Frontend files only, except generated types which remain integrator-owned.
5. Integrator configures loopback deployment and no-secret same-origin proxying, updates launcher/documentation and checks generated contract parity.
6. Run relevant backend/edge/frontend tests, build/type checks, isolated database migration and cross-process smoke tests. Preserve all existing data and unrelated edits.
7. Rebuild affected cached services only, check real local startup without changing models, and inspect synthetic browser fixtures at the approved viewport sizes. Store rendered evidence and run the frontend-design evidence checker.
8. Commit only this task's changes after verification; follow the user's prior instruction to push meaningful implementations, without publishing credentials or private diagnostic data. Update changed indexed code after a successful push, without a full semantic rebuild.

Endpoint coordination: browser POST `/v1/local/session` returns LocalSessionOut and an HTTP-only cookie; unsafe cookie requests require `X-CSRF-Token`. Browser websocket `/v1/ws/alerts` uses the local cookie without service-key subprotocols. New watchlist routes follow the design; PATCH uses `expected_version`. Diagnostics use DiagnosticsSnapshot from the shared DTOs; worker output must match it. Missing evidence is 404, stale session 409, unauthorised 401/403, unavailable worker 503. Local sessions and service credentials remain separate.
