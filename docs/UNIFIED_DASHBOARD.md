# Unified local dashboard

The Paper teal interface has three sections: Analyse, Alerts and Watchlist.
There is one browser address, `http://127.0.0.1:5173/`, and no manual operator
unlock. The local computer/browser is trusted; this is not individual-user
authentication or a public deployment profile.

## Start

Start Docker Desktop. From the project directory:

```powershell
# Optional: explicitly allow transient images/full OCR for permitted recordings.
$env:DASHBOARD_RECORDED_EVIDENCE = 'true'
.\scripts\start-recorded-analysis.ps1 -Device gpu -SkipBuild
# Use -Device cpu on a machine without an NVIDIA GPU.
```

Omit `-SkipBuild` when images need rebuilding. Preserve the restored model
volumes and `.env`; service keys remain server-side. Do not use a VITE_* variable
to pass an operator key into JavaScript. The old `compose.autounlock.yaml` name
is retained only as a compatibility alias for the local session profile.

The Docker-published UI, API and worker ports must stay on loopback. The Vite
API proxy preserves the browser's local Host. `LOCAL_DASHBOARD_MODE=true`
uses automatic local sessions with HTTP-only same-site cookies, strict origins
and CSRF-protected mutations. Direct keyed API/script compatibility remains.
Do not expose this profile through a LAN/public tunnel.

## Use

1. Analyse: choose an already registered, permitted recording and explicitly
   start processing. Selection or opening the page does not start inference.
2. Pause/resume/stop using the existing shared-session controls. Navigating to
   another section or closing the Inspector does not create a second worker.
3. Select a vehicle track to inspect native crops, reconstruction, per-frame
   character evidence, candidate alternatives, gate reasons and device timings.
4. Watchlist: add/search masked entries, edit reason/severity, or deactivate and
   reactivate entries. Deactivation preserves alert history. The keyed identifier
   is generated on the server; full plaintext input is not retained.
5. Alerts: review central matches and acknowledge, resolve or mark false positive.
   Model-local possible matches are not presented as confirmed central alerts.

Images, full candidates and the Inspector are available only for permitted
recordings when recorded evidence is enabled. Genuine live camera imagery is
not opened by this UI change. Scores are uncalibrated model scores, not measured
accuracy. Reconstruction is derived evidence, not an extra observation.

## Validation and limits

The implementation plan and approved design are under `docs/plans/` and
`docs/superpowers/specs/2026-10-30-sentinel-unified-dashboard-design.md`.
The canonical new DTOs are `backend/app/unified_contract.py`; regenerate browser
types and JSON Schema with `python tools/generate_unified_contract.py` and check
parity with `--check`.

The UI overhaul does not change weights, model acceptance, sampling or recognition
quality. Existing accuracy gaps, the rejected-event queue limitation and older
handoff archive issue remain separate work. Historical two-dashboard instructions
and videos describe the previous interface, not the current product scope.
