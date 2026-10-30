# Sentinel unified dashboard — Paper teal

Date: 30 October 2026

Status: visual direction approved; written specification awaiting review. No application implementation is included in this document.

## 1. Decisions and purpose

The user selected **Paper teal** after reviewing the colour preview. This specification carries forward the proposed focused workspace: one dashboard, three navigation sections, and a contextual Inspector replacing the separate model dashboard.

The product processes downloaded CCTV recordings, recognises vehicle plates across frames, compares accepted readings with a watchlist, and presents operational alerts. Drawing vehicle or plate boxes is not the completion criterion. UI work does not establish recognition accuracy.

The primary user is the local operator; the same authorised user may inspect model evidence. The Inspector is a progressive-disclosure control, not a separate security role.

### In scope

- One browser entry point at `http://127.0.0.1:5173/`, one operator unlock, and one shared recognition session.
- Analyse, Alerts and Watchlist sections.
- Recording selection and existing start, pause, resume and stop controls.
- Aligned analysed frames, recent vehicle-track results and an optional Inspector.
- Protected watchlist management, including the backend support this requires.
- Migration of engineering capabilities into the main application, with old browser entry points redirected.
- Responsive, accessible UI and regression tests for existing behaviour.

### Not in scope

- Maps, routes, GIS, regional statistics, camera-catalogue search or hourly-report navigation.
- Live Sentinel access, face recognition, registry integration or regional-scale deployment claims.
- New model weights, training, changed acceptance thresholds, automatic acceptance of uncertain candidates or recognition-accuracy claims.
- Arbitrary video uploads, scrubbing/seeking, bulk watchlist import, individual user accounts or a second theme.
- Unrelated removal of backend capabilities, stored history or Docker volumes.

Existing source registration remains the way to add a permitted recording. The redesign must not show an upload control without an implemented upload contract.

## 2. Visual system

| Token | Value | Use |
| --- | --- | --- |
| Background | `#F6F5F1` | Warm neutral workspace |
| Surface | `#FFFDF9` | Navigation, tables and Inspector |
| Main text | `#242925` | Labels and primary content |
| Secondary text | `#626B65` | Timestamps and supporting metadata |
| Border | `#DEDFD8` | Thin structural dividers |
| Accent | `#0F766E` | Primary action and active navigation |
| On accent | `#FFFFFF` | Primary button text |
| Successful operation | `#166534` | Healthy/successful system actions |
| Uncertain | `#92400E` | Uncertain readings and attention |
| Watchlist alert/error | `#B91C1C` | Central matches and actionable failures |

- Use CSS custom properties for product tokens, retaining React, TypeScript, Vite and Lucide.
- Use the system font, including Segoe UI on Windows; use system monospace for plate text and numeric inspection values. No external font dependency.
- Default body text is 14px, secondary labels at least 12px, and page headings approximately 24px. Use restrained weights and spacing based on an 8px rhythm.
- Use small 8px corner radii and thin dividers. Shadows are reserved for overlays.
- Remove glow effects, gradients, decorative metrics, oversized introductions and nested card chrome.
- Labels accompany status colours. Green must not imply independently verified recognition.
- Motion is limited to brief state changes and respects reduced-motion preferences.

## 3. Application shell and navigation

A compact top bar contains Sentinel, Analyse, Alerts, Watchlist, connection status and Lock/Unlock. A small settings menu replaces secondary navigation. There is no full-height sidebar.

Analyse is the initial section. Alerts shows a count of active central alerts. Switching sections does not create, reset or stop the shared inference session.

The application owns session state and heartbeat handling above individual page components. Opening the Inspector does not start another worker or another analysis session.

Connection status and processing state are separate: the API can be connected while analysis is paused, completed, loading or unavailable. CPU/GPU labels come from actual runtime metadata; mixed devices are not labelled entirely GPU.

## 4. Analyse workspace

### Toolbar

- Select an existing authorised recording, identified by its human-readable name and recorded-source label.
- Use the currently supported Start, Pause, Resume and Stop actions with state-appropriate disabling and visible pending feedback.
- Merely selecting or previewing a recording does not start recognition.
- Switching an active recording requires an explicit stop-and-switch confirmation; cancellation preserves the session.
- A source/session conflict is shown as a conflict, not silently replaced.

### Primary content

On desktop, the analysed frame receives approximately two-thirds of the workspace and the results list receives the remainder. On narrower screens, the frame appears above results.

Show an original/overlay display control. Every displayed overlay belongs to the same analysed frame, session and video timestamp. Do not put old boxes over newer playback. The frame is an analysed sample, not a promise of real-time video.

The progress indicator is read-only. Show video position and duration separately from processing time. Do not invent a recording's wall-clock capture time when it is unknown.

### Results

Use compact rows containing vehicle-track ID/class, available plate text, reading state, central watchlist result and video time. Selecting a row opens the Inspector.

Preserve the following distinctions:

- Reading: evidence is accumulating.
- Uncertain: a provisional candidate exists but has not passed acceptance.
- Accepted by model: the existing model rules passed; independent verification is not implied.
- Unreadable: no usable complete reading was obtained.
- No plate yet: a vehicle track exists without suitable plate evidence.
- Evidence unavailable: the relevant in-memory evidence has expired; this is not the same as unreadable.
- Watchlist match: an alert from the central accepted-event workflow, not a locally guessed candidate match.

Scores are called model scores, never accuracy percentages. An acknowledgment is not a declaration that OCR is correct. No UI action bypasses the acceptance gate.

Counts must reflect their definitions: vehicle tracks, model-accepted readings, uncertain readings and central matches. Track counts are not ground-truth counts of every distinct vehicle. The bounded recent-results list must disclose its retention limit; it must not imply that every historical crop remains available.

## 5. Inspector

The Inspector is closed by default and opens within the same application. It is a right-side sheet on desktop and a full-screen sheet on narrow screens, with labelled Close and normal keyboard focus handling.

It contains the selected track's:

1. Native plate crop, original dimensions, timestamp and zoom.
2. Checked reconstruction, where available, labelled derived evidence rather than an independent frame.
3. Distinct source-frame OCR observations and per-character alternatives.
4. Ranked hypotheses: a short initial list and expansion up to the existing retained-candidate bound, not every imaginable registration.
5. Observation count, acceptance/rejection reasons and reconstruction conflicts.
6. Actual model identities, devices and measured stage durations under an expandable technical section.

Show absent evidence and failed reconstruction explicitly. Enlarging a crop does not add pixels or establish new detail. Repeated/resized/reconstructed copies do not increase the independent-observation count.

Diagnostic metadata and images load on demand. Closing the Inspector stops its additional polling/image requests, but not analysis. Selected track and session IDs accompany requests so source changes cannot mix evidence.

## 6. Alerts

Use a compact queue with filters for New, Acknowledged, Resolved and False positive. Selecting an alert opens details within the same page/sheet pattern.

Details include source, video time where available, processing time, plate identifier subject to the display policy, watchlist reason/severity and existing action audit history. Reuse the supported acknowledge, resolve and false-positive actions.

Pending actions disable duplicate submission; failure preserves the current detail and offers retry. Status changes remain server-authoritative and survive restart.

Normal operational UI contains no synthetic-alert injection button. Test/demo fixtures remain separate and explicitly labelled. Maps and route reconstruction are not included in alert details.

## 7. Watchlist management

The existing worker snapshot is not an operator-management API. Implement protected operator endpoints, rather than exposing the edge key or using browser-side database access.

The page lists masked entries, reason, severity and active state; provides search, Add, Edit details and Deactivate/Reactivate. Plate text is entered only when adding or performing an exact membership lookup. Plate changes require deactivating the old entry and adding the new identifier, preserving history.

### Proposed operator contracts

- `GET /v1/watchlist`: paginated entries; default 25, maximum 100 per response. Return ID, masked plate if available, reason, severity, active state, entry version and creation time. Do not return plaintext or the HMAC.
- `POST /v1/watchlist/search`: search text in the JSON body, supporting masked/reason search and normalised exact-plate lookup without putting raw plates into URLs.
- `POST /v1/watchlist`: plate, nonempty reason up to 240 characters, and severity from low/medium/high/critical; default high. Use existing plate normalisation and validation. A duplicate returns a clear conflict rather than silently replacing/reactivating an entry.
- `PATCH /v1/watchlist/{id}`: change reason, severity or active state with the expected entry version. A stale edit returns conflict; refresh before retrying.

Deactivate is reversible and never hard-deletes existing sightings, alerts or audits. It requires confirmation. The UI does not offer retrospective rematching of old sightings.

### Data and security

- Derive the keyed plate identifier on the server using the existing secret; never send that secret to the browser.
- Retain the existing HMAC-based storage policy, plus a masked display value for new entries. Existing entries whose mask cannot be reconstructed show their entry ID and a legacy-hidden label; do not manufacture a mask.
- Add a monotonic watchlist revision and mutation audit using an Alembic migration. The revision advances transactionally on actual changes, including deactivation. Do not derive it only from currently active entries, which could make it decrease.
- Audit actions and changed non-sensitive metadata; do not retain plaintext input in request logs or audits. The actor is the local shared-key operator, not a claimed individually identified account.
- Every management request requires the operator key. Reuse current origin/access restrictions; perform no mutations through GET.
- Preserve the edge snapshot contract and matching behaviour. Validate snapshot compatibility after mutations.

## 8. Single frontend and protected evidence boundary

The React application remains the sole browser UI on port 5173. FastAPI remains the central API. The recognition worker may continue on its existing local service port; one dashboard does not mean one process.

Add an authenticated, allowlisted central diagnostic gateway for the Inspector. The browser never receives the edge service key or worker viewer token. Do not iframe the old engineering dashboard or proxy arbitrary worker URLs.

Use bounded track metadata and on-demand image responses rather than repeatedly forwarding the full engineering snapshot with every crop. Validate identifiers and session identity, enforce response-size/candidate limits and use no-store responses. Keep the existing lightweight status contract separate from diagnostics.

Full plate hypotheses and images are transient, protected inspection data for permitted recorded footage and remain subject to the recorded-evidence setting. If that setting is disabled, show masked results and a clear evidence-unavailable state; do not bypass the flag. No such exposure is added to the genuine live-feed path. Existing persistent accepted-event storage remains masked/keyed.

Do not put credentials, plates or diagnostic payloads into URLs, localStorage, analytics or automatic exports. Lock clears sensitive UI state and stops protected requests. Preserve existing worker lease behaviour without silently terminating another authorised consumer's session.

Redirect legacy worker browser entry points, including the separate engineering page, to the unified application without placing keys in the redirect. Keep required internal worker endpoints. Update launcher messages and documentation to advertise only the unified UI.

## 9. Responsive, accessibility and content-state contract

Target checks: 1440x900 and 1366x768 desktop; 768x1024 tablet; 390x844 mobile, with a 320px minimum-width sanity check.

- Stack analysis content when the two columns cannot remain readable. Preserve full plate readability and sensible wrapping of long source names/reasons.
- Use full-screen detail sheets on narrow screens, large touch targets, semantic controls, visible focus and keyboard-operable navigation.
- Modal/sheet focus returns to its trigger. Escape closes dismissible sheets. Critical status information is not hover-only or colour-only.
- Use restrained live announcements; do not announce every inferred frame.

| State | Visible feedback | Safe action |
| --- | --- | --- |
| Loading | Stable layout; explicit model/service loading state | Wait, cancel where supported |
| Empty | Distinguish no registered recordings, no analysis, no readings and no matches | Select/register a source or start processing |
| Error | Name the unavailable stage without revealing URLs or credentials | Retry the relevant action |
| Partial | Keep available data but mark missing/aged evidence and service disconnection | Inspect available data or reconnect |
| Success | Show completed playback or saved operator/watchlist action | Continue reviewing; reversible deactivation where relevant |
| Permission | Locked state clears images and full text | Unlock using the existing operator key |

## 10. Boundaries and verification

Application areas affected: the React shell/styles and recognition view; new Inspector and Watchlist components; frontend API types; central diagnostic/watchlist routes and models/migrations; worker internal evidence/browser-entry handlers; launch instructions and tests.

Preserve existing user edits and data. Remove unused map dependencies only after their frontend consumers are removed and the bundle/tests confirm no remaining use. Do not delete backend map/history features or recordings as part of the overhaul.

Verification must cover:

1. Frontend build/type checking and relevant unit tests.
2. Backend validation, operator-key rejection, duplicate/conflicting watchlist edits, revision monotonicity, reversible deactivation and legacy-entry display.
3. Diagnostic allowlists, bounds, no-store behaviour, source/session mismatch, disabled recorded-evidence settings and absence of leaked service keys or genuine live imagery.
4. One shared session across sections; no inference on preview, no inference restart on navigation/Inspector changes, and no stale evidence after source changes or locking.
5. Alert actions and watchlist changes persisting across restart, using isolated test data.
6. Browser verification at the target widths, including all listed content states, keyboard/focus behaviour and long content. Capture only clearly synthetic UI fixtures for rendered evidence.
7. Existing CPU/GPU startup and recognition interfaces remaining compatible. Model scores/outputs are not tuned to make the new UI look successful.

Record the frontend design contract and rendered evidence and run the installed frontend-design evidence checker. A static source inspection does not establish rendered correctness.

Known recognition-quality gaps, rejected-event queue handling and the handoff archive's token-directory omission remain separately tracked work. This redesign does not claim to resolve them.

## 11. Approval and next step

Paper teal is approved. This document records the proposed unified implementation boundary. Review and approve this written specification before implementation planning or application changes. No further visual direction needs to be invented unless the user requests a revision.
