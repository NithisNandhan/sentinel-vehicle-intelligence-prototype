# Unified dashboard validation

Validated on 30 October 2026. This records the interface overhaul, not a new
recognition-accuracy benchmark.

## Delivered

- One Paper teal React interface at `http://127.0.0.1:5173/` with Analyse,
  Alerts and Watchlist. The old worker browser address redirects here.
- One shared, explicitly started recorded-analysis session across navigation.
  Original/overlay frame selection and a contextual recognition Inspector.
- Bounded, recorded-only diagnostic evidence: native crops, reconstruction,
  per-frame characters, alternative candidates, gate reasons and runtime details.
- Automatic trusted-local sessions; no operator unlock or browser-held service
  key. HTTP-only SameSite Strict cookies, exact local Host/Origin checks and
  CSRF verification protect browser mutations. Existing service-key access remains.
- Server-side keyed watchlist identifiers, masked displays, audited edits,
  optimistic version checks and reversible deactivation. Legacy entries retain
  their hidden identifier rather than inventing a display value.
- Central alert acknowledgement, resolution, false-positive actions and history.
  Uncertain model suggestions remain separate from central watchlist alerts.
- Removed the unused map component/dependency. Updated vulnerable frontend test
  tooling and compatible dependencies; the resulting npm audit reports zero
  known vulnerabilities at this validation point.

## Evidence

- Backend: 103 tests passed. Three existing framework deprecation warnings remain.
- Frontend: production build/type check passed; 31 tests passed with Vitest 5.0.5.
- Worker: 87 focused diagnostics/session/recognition tests passed; one was skipped
  because the isolated worker runtime lacks central API dependencies. Tests use
  the cached research runtime; see the commands below.
- Browser: 15 workflow checks passed with labelled synthetic API/media fixtures.
  Viewports: 1440×900, 1366×768, 768×1024, 390×844 and 320×844. Checked navigation,
  no automatic inference, Inspector close/focus/polling, watchlist changes and
  version conflicts, alert decisions, pause/resume, and loading/empty/error/
  partial-service/access-failure states. Desktop and narrow-screen renders were
  visually inspected. This is focused interaction testing, not a full accessibility
  certification.
- The installed frontend-design evidence checker passed. Local artefacts are in
  `data/ui-validation/unified-20261030/`; private runtime data and screenshots are
  not added to GitHub by this implementation.
- Real deployed-service read-only smoke: automatic bootstrap 200, watchlist 200,
  recognition status 200, recording catalogue 200, alerts 200, WebSocket connected,
  missing-CSRF rejection 403 and legacy redirect 307. Two permitted recordings
  are registered. No analysis was started by opening or navigating the UI.
- PostgreSQL migration reached `0003_watchlist_management`. Existing counts were
  preserved: 16 sightings, 15 alerts and 22 alert-audit rows. A private local
  database backup was made before deployment. Synthetic mutation testing used
  isolated fixtures/databases, not the existing operational records.

## Reproduce

Use the existing local dependencies/model cache and Node 22.12+ on the supported
22 LTS line (or Node 24 LTS) for the updated frontend test runner. The dashboard
container uses Node 22. Browser scripts require an
installed Chrome and Playwright; set `PLAYWRIGHT_MODULE` to your local package
path when the supplied Windows-runtime fallback does not exist.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:PYTHONPATH='backend'
& backend/.venv/Scripts/python.exe -m pytest backend/tests -q -p no:cacheprovider
& backend/.venv/Scripts/python.exe tools/generate_unified_contract.py --check

Push-Location frontend
npm ci
npm run build
npm test
npm audit
# For the isolated browser fixture check, leave this running in another terminal:
npm run dev -- --host 127.0.0.1 --port 5174
Pop-Location
node tools/verify_unified_dashboard.cjs
# Against the deployed stack; read-only, with no saved footage:
node tools/smoke_local_dashboard.cjs

docker exec sentinel-prototype-recorded-analysis-1 python /workspace/tools/run_ocr_research.py -m pytest /workspace/edge/tests/test_unified_diagnostics.py /workspace/edge/tests/test_shared_recognition.py /workspace/edge/tests/test_recorded_session.py /workspace/edge/tests/test_live_detection_viewer.py /workspace/edge/tests/test_tracked_ocr.py /workspace/edge/tests/test_gpu_detectors.py -q -p no:cacheprovider
```

## Limits

Weights, sampling, reconstruction algorithms and recognition acceptance thresholds
were not changed. This work establishes no new OCR accuracy, every-vehicle recall,
real-time throughput or live-camera result. Evidence is opt-in and restricted to
permitted recordings; live imagery/full hypotheses remain blocked. The runtime
was left paused for explicit user initiation. This is a trusted local-computer
profile, not a public or multi-user deployment. Previously identified recognition,
rejected-event queue and historical handoff-archive issues remain separate tasks.
