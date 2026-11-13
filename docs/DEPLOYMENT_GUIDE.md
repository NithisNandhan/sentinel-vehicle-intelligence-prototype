# Deployment guide

This guide records the earlier candidate-profile implementation and its test
boundaries. For current source selection, restoration and recorded-video setup,
start with [the current handoff](CURRENT_HANDOFF_20261113.md).

Nothing here was run against Docker in the build container (no daemon). Commands are the repository's existing ones plus
the new settings; the Python/Node test commands were run.

## Prerequisites

Docker Desktop (WSL2 on Windows) for the full stack; Python ≥3.11 and Node 22 for tests. GPU optional: the stack runs on
CPU; `candidate-gpu` is intended for an NVIDIA GPU (`compose.recorded-gpu.yaml`) and is **unmeasured**.
Model weights and recordings are not in Git (see `TEAMMATE_HANDOFF.md`).

## Launch

```bash
# one-time: keys in .env (never overwrite an existing .env)
powershell -File scripts/init-env.ps1
docker compose --profile offline up --build          # API, database, dashboard, demo simulator
powershell -File scripts/start-recorded-analysis.ps1 # recorded analysis (CPU)
```

Backend alone (development): `cd backend && pip install -r requirements-dev.txt && alembic upgrade head && uvicorn app.main:app`.
Frontend: `cd frontend && npm ci && npm run dev`. Open <http://127.0.0.1:5173>. Never use `docker compose down -v`.

## Database migration

```bash
cd backend && alembic upgrade head      # 0003 -> 0004_result_states (additive; creates recognition_results, review_decisions)
alembic downgrade 0003_watchlist_management   # drops only those two tables
```
Verified on PostgreSQL 16 here (upgrade, repeat upgrade, downgrade, upgrade). `0001` creates PostGIS and could not be run locally.

## Profiles

| Want | Set in `.env` |
| --- | --- |
| Today's behaviour (default) | `SENTINEL_PROFILE=baseline` |
| Candidate on CPU | `SENTINEL_PROFILE=candidate` |
| Candidate on GPU (+ native tiles, text detector) | `SENTINEL_PROFILE=candidate-gpu`; optional `SENTINEL_TEXT_DETECTOR_ASSETS=<dir>` |
| Roll back | `SENTINEL_PROFILE=baseline`, then `docker compose --profile recorded up -d --force-recreate recorded-analysis` |
| Roll back one feature | e.g. `SENTINEL_PLATE_LEVEL_TRACKING=false` (flags override the profile) |

The recorded-analysis container mounts `./edge` read-only, so a recreate loads the code on disk. Database rows already
written stay; `baseline` simply stops emitting 2.0 events. Rollback verified at unit level (flag/profile tests), not by a
Docker drill.

## Review set-up

1. Choose a reviewer token (long, random). Compute its digest: `printf %s "$TOKEN" | sha256sum`.
2. `.env`: `REVIEWER_TOKENS=asha:<digest>` (comma separate more). Never store the token itself.
3. Optional evidence: `REVIEW_EVIDENCE_ENABLED=true` (API) and `SENTINEL_REVIEW_EVIDENCE=true` (edge). Live-camera imagery
   needs the additional `SENTINEL_REVIEW_EVIDENCE_LIVE=true`; leave it off unless policy allows.
4. Optional partial leads: `SENTINEL_PARTIAL_WATCHLIST=true` (requires a candidate profile; read the HMAC risk first).
5. Revoke a device: `REVOKED_EDGE_IDS=region-demo-1-shared`; demand identities with `REQUIRE_EDGE_ID=true`.
6. Open the dashboard → **Review**, enter the token (kept only in tab memory).

## Evaluate

```bash
python tools/validate_annotations.py --labels data/labels/<labels>.json --manifest data/evaluation/<name>-manifest.json
# Same invocation as docs/RECORDED_VIDEO.md; set SENTINEL_PROFILE=baseline or candidate in the environment for each run.
docker compose --profile recorded run --rm --no-deps -v "${PWD}/data:/workspace/data" recorded-analysis \
   python /workspace/tools/run_ocr_research.py /workspace/tools/evaluate_recorded.py \
   --labels /workspace/data/labels/<labels>.json --split development --recordings-dir /recordings --device cpu \
   --output /workspace/data/evaluation/dev-candidate-001.json
python tools/shadow_compare.py --baseline data/evaluation/dev-baseline-001.json --candidate data/evaluation/dev-candidate-001.json
python tools/seeded_watchlist_replay.py --output data/evaluation/replay.json [--partial-matching]   # needs backend deps
```
Run the baseline and candidate reports with `SENTINEL_PROFILE` set accordingly on the *same* labels. Sealed splits need
`--final` and are logged in the ledger.

## Tests

```bash
cd edge && pip install -e .[test] && python -m pytest -q          # 2 known failures, see TEST_REPORT.md
cd backend && pip install -r requirements-dev.txt && python -m pytest -q
cd frontend && npm ci && npx vitest run && npx tsc --noEmit
PYTHONPATH=edge:edge/tests python -m pytest -q tools               # replay needs backend deps in the same interpreter
```

## Operations notes

- Monitoring that exists: `dropped_sampled_frames`, collector drops and decode-queue peak in the engineering snapshot;
  live worker health heartbeats; outbox depth. **Recorded sessions send no camera-health reports** (known gap).
- New counters: `delivery.result_metrics` (suppressed fragments, delivered per state, partial-lead outcomes),
  `TrackedOCR.metrics` (`confirmation_blocked_*`, `track_reassociations`, `near_duplicate_crops`, `passages_*`).
  They are not yet exported to an external monitor.
- Dead letters: `Outbox.dead_letter_count()`; after fixing the cause `Outbox.requeue_dead_letters()`.
- Retention runs at API start and via `POST /v1/admin/retention/run`; schedule it (cron) for long-running services.
- Licences: see `RELEASE_READINESS.md` (Ultralytics AGPL-3.0, SVTRv2/OpenOCR, dataset terms) before any distribution.
