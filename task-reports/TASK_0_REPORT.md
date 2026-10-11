# Task 0 report: prepare the checkout and a test baseline

Closed 10 October 2026. **Score: 9 / 10**

## What was there before

- Checkout at commit `b3b05e1`, two documentation commits behind `origin/main`.
- `scripts/init-env.ps1` had been edited locally from `-Encoding utf8NoBOM` to
  `-Encoding UTF8` so it would run under Windows PowerShell 5.1. That wrote a
  byte-order mark at the start of `.env`.
- `.venv311` existed but held only `pip` and `setuptools`.
- No frontend packages installed. Docker Desktop installed but stopped.
- No test had been run on this machine.

## What I changed

| File or place | Before | After |
| --- | --- | --- |
| Git checkout | `b3b05e1` | `27935a3` (fast-forward, local edit kept) |
| `scripts/init-env.ps1` | `Set-Content … -Encoding UTF8` (writes a mark in 5.1) | Writes the bytes directly as UTF-8 with no mark; works in PowerShell 5.1 and 7 |
| `.env` | Started with bytes `EF BB BF` | Mark removed; all generated keys kept unchanged |
| `.venv311` | Empty | Edge test extras and backend dev requirements installed |
| `frontend/node_modules` | Absent | Installed from the lockfile (`npm ci`) |
| Docker Desktop | Stopped | Running (engine 29.6.1, 7.2 GB RAM for containers) |
| `backend/tests/test_recognition.py` | One parametrised test used a 1 MiB test ID | Short IDs added; the test body is unchanged |
| `BUILD_INSTRUCTIONS.md` | Did not exist | The plan, rules and audit log |

## Audit result

| Suite | Result |
| --- | --- |
| Edge | 287 passed, 1 failed (Windows-only: a test hard-codes a Linux `/workspace` path) |
| Backend | 53 passed |
| Handoff tools | 14 passed in a Linux container; 11 of 14 on Windows (SQLite file locks during cleanup) |
| Frontend | 15 passed; production build passed |

These match the counts recorded in the handoff (287, 53, 14, 15).

## What still has to change

- Install GitHub CLI (`gh`) or plan a browser download of the release assets.
- Confirm this GitHub account can read the private release on
  `NithisNandhan/sentinel-vehicle-intelligence-prototype`. Not checked yet.
- Free disk is 35 GB on D: and 40 GB on C:, below the guide's 60 GB allowance.
  A selective restore is needed (Task 8).
- Optional: make `test_camera_suitability` and the three handoff tests
  portable to Windows. They pass on Linux, where the project runs them.

## Score

| Part | Points | Reason |
| --- | --- | --- |
| Built | 4 / 4 | Everything in scope is done |
| Tested | 3 / 3 | All four suites run; every failure explained and shown to be Windows-only |
| Proven | 2 / 3 | Docker and the fixed script verified for real; release access and disk space are still unconfirmed |
