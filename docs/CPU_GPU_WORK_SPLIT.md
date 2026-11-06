# CPU and GPU work allocation

> Historical planning document dated 9 October 2026. Statements below describe
> that starting point, not current completion. Recorded-video ingestion and
> the unified dashboard have since been implemented. Use the current README,
> `RECORDED_VIDEO.md` and measured evidence for present behaviour.

Prepared 9 October 2026 for the Ryzen 7 teammate and the project owner working with Codex.

The teammate owns the work that can be developed and verified on CPU. The owner and Codex own the model experiments, training, GPU integration and final performance measurements. This is a task schedule, not a record of completed work. No tasks have been sent to another person or added to a calendar.

The schedule uses completion milestones because working hours and a deadline have not been supplied. Both owners start Phase 1 together; later phases begin when their listed inputs are ready. A delayed GPU experiment should not block dataset review, recorded-input development or interface tests.

## Current starting point

- Source and release assets are documented in [TEAMMATE_HANDOFF.md](TEAMMATE_HANDOFF.md). A repository clone alone excludes the large model and data archives.
- The synthetic central workflow already works without an NVIDIA GPU. A complete clean-machine restoration remains an acceptance task.
- The shared engineering session uses general YOLO26s, a parent YOLOv9 plate detector, adapted Awiros OCR, temporal evidence and checked ECC reconstruction. Its previously validated execution used CPU detectors and GPU OCR.
- The independent production worker still uses YOLOv8n/FastALPR. Keep its defaults distinct from the engineering session.
- Recorded-video input in the integrated shared session remains to be implemented according to the 8 October status report. Existing live launchers are not recorded-video launchers.
- The 28/31 exact OCR result belongs to a small still-image diagnostic. Genuine live recognition and independent multi-vehicle video accuracy remain unverified.

## Schedule

| Phase | Teammate on Ryzen 7 | Owner and Codex on NVIDIA | Exit milestone |
|---|---|---|---|
| 1 Restore and agree inputs | C1 clean CPU restore; C2 footage inventory and evaluation protocol | G1 restore/check GPU models; agree protocol and interfaces | Synthetic workflow works on teammate machine; source, frame and result contracts agreed |
| 2 Build input and freeze labels | C3 recorded source contract; C4 video adapter; finish C2 independent labels and split | G2 run detector/OCR baseline on development footage using the agreed source interface | Video decoder/control tests pass; development/validation/test manifests fixed before tuning |
| 3 Integrate controls and diagnose | C5 operator and engineering controls; C6 CPU crop/track/layout diagnostics | G3 run original versus reconstruction/restoration/OCR ablations | Both interfaces observe one recorded session; stage failures have measured causes |
| 4 Improve the measured bottleneck | C7 evaluation tools and CPU logic changes justified by results; C8 delivery/recovery tests | G4 targeted training if needed; export and runtime parity | Candidate change has validation evidence, versioned assets and no unresolved export regression |
| 5 Verify recognition and runtime | C9 independent label review, report scoring and operator workflow verification | G5 GPU held-out run, endurance, latency, memory and overload measurements | Correct and wrong acceptances, abstentions and resource costs are reported on the frozen test set |
| 6 Package the demonstration | C10 documentation, presentation, CPU instructions and demo capture | G6 review model claims, final integration and release reproducibility | Demonstration is explicitly recorded footage; setup and reports are reproducible |

Phases 2 and 3 use development footage for iteration. Neither owner tunes models, thresholds or preprocessing on the final held-out test. If the final test reveals a failure, record it; subsequent tuning needs a new independent test set.

## Teammate task list

All tasks below start as **scheduled**, not completed.

### C1 Clean CPU setup and restoration

Priority: first. Dependencies: repository access and selected release archives.

Install the ordinary development tools, restore into fresh destinations, generate local keys, run the synthetic demo and relevant existing tests. Preserve archive hashes and saved metadata. Start with central services; model archives and CPU model execution are a separate step. Full training archives are unnecessary for backend/frontend development.

Add or document a CPU development route that does not request NVIDIA devices. Check dependency/provider selection and fail visibly if an experimental model cannot load on CPU. Do not run the current GPU engineering launcher unchanged and assume fallback works. Record exact Ryzen model, RAM, OS, package versions and available disk space.

Done when the teammate can independently build the source, use the operator workflow and repeat a bounded CPU diagnostic with its dependencies stated. GPU speed parity is not part of this task.

### C2 Footage, labels and evaluation protocol

Priority: first. Dependencies: none beyond permitted source access; owner reviews labels/protocol.

Inventory permitted original-resolution Indian CCTV recordings and retain licence/source receipts. Assign video and passage IDs; independently transcribe readable plates, mark unreadable passages, and annotate vehicle and plate boxes. Include differing layouts, distance, motion, lighting and visibility where available. Preserve uncertain labels rather than converting model output into truth.

Separate development, validation and test by vehicle/video/scene, with registration separation where feasible. Detect duplicates and near-duplicate passages across splits. Agree precision, coverage and latency goals with the owner before the final test, and state the actual number of independently labelled vehicles. No minimum sample count or generalisation claim is inferred from a small pilot.

Done when a versioned manifest, labels, review notes and frozen split are available to both machines. Only development labels guide model debugging; final test labels are used for final scoring.

### C3 Recorded-source contracts and metadata

Priority: Phase 2. Dependencies: C1 and the Phase 1 interface agreement.

Extend the existing source contracts to represent recorded footage explicitly across API schemas, event storage and both interfaces. Keep recording ID, video presentation time and processing time distinct. Define track reset and replay/session identity so restarting or looping cannot silently create ambiguous duplicate events. Test serialization and old live/synthetic behavior.

Done when recorded source metadata survives round-trip through the application and cannot be mistaken for live footage. Full plate candidates and images retain the current separation between engineering and operator views.

### C4 Recorded-video decoding and session controls

Priority: Phase 2. Dependencies: C3 and sample licensed footage.

Implement local-video PyAV input into the existing shared recognition session. Preserve native image pixels and original video PTS. Support select/start/pause/resume/stop and document seek/replay behavior. Clear tracking/crop evidence at source changes, seeking, EOF/loops and detected discontinuities. Handle missing timestamps explicitly.

Support paced demonstration playback and unpaced evaluation as distinct modes. Decode/control tests can use lightweight fixtures and an injected test recognizer; label those as plumbing tests. Use a CPU recognizer for bounded functional checks when available. The owner supplies the GPU runtime for performance and model validation.

Done when the same recorded sequence reaches the session correctly, controls behave predictably and failures stop or recover without leaking earlier vehicle evidence.

### C5 Operator and engineering interfaces

Priority: Phase 3. Dependencies: C3/C4.

Add recorded-source selection and controls, video time, processing state, explicit recorded labels and useful stage counters. Both views must observe one session. Keep uncertain candidates in the engineering console and only accepted metadata in the central workflow. Preserve key handling, private evidence routes and source URL redaction.

Done when start/pause/switch/stop work from either interface, stale rows disappear, and preview alone does not start recognition. Fixture-based UI tests do not establish OCR accuracy.

### C6 CPU image, crop, tracking and layout diagnostics

Priority: Phase 3. Dependencies: C2/C4; run on development footage.

Inspect native crop extraction, candidate ROI selection, duplicated observations, track mixing, timestamp gaps and alignment rejection reasons. CPU can handle crop geometry, ECC alignment, layout/row splitting, deskew, quality measures and temporal scoring. Compare against marked crops to separate localization failures from recognition failures.

Implement a targeted CPU-stage correction only when the diagnostics justify it. Do not add every proposed stage simply because it appears in the architecture. Keep model wrappers/checkpoints under the owner’s responsibility.

Done when failures have stage-specific examples, counters and tests, and any correction improves the agreed development comparison. The owner reruns the real-model comparison on GPU.

### C7 Evaluation and result analysis

Priority: Phase 3 onward. Dependencies: C2; GPU execution is supplied by G2/G3/G5.

Prepare manifests, benchmark orchestration, score calculation and report viewers. Score unique vehicle coverage, correct plate localization, usable crop yield, exact whole-plate reads, wrong accepted reads, abstentions and candidate recall. Treat accepted precision as undefined when there are no acceptances. Distinguish inference duration, video time, paced playback and dashboard delivery latency.

Review returned GPU predictions against independent labels. Confidence calibration and acceptance-rule analysis are CPU work, using validation outputs only. Use untouched test outputs for final scoring after model and acceptance settings are frozen. Do not claim a likelihood score is a measured accuracy percentage.

Done when a report can be regenerated from manifests and prediction files with model revisions, settings and hashes recorded. Slow CPU experiments are useful correctness checks but cannot reproduce GPU latency.

### C8 Central delivery, reliability and regression

Priority: Phase 4. Dependencies: C3/C4; accepted-event fixtures before G5 real outputs.

Verify SQLite outbox delivery, API outages, replay idempotency, restart recovery, watchlist matching, alert acknowledgement/resolution, route/report behavior and audit persistence. Exercise corrupted input, EOF, missing PTS, interrupted extraction, stale sessions and absent model dependencies. Use real GPU recognition outputs for the final genuine recorded workflow.

Done when failures are bounded, queued accepted events survive restart and operator decisions persist. Synthetic event injection remains clearly labelled and cannot substitute for a recognized recording.

### C9 Independent verification

Priority: Phase 5. Dependencies: frozen model/settings and G5 outputs.

Independently review labels and the owner’s held-out results, check denominators and trace reported acceptances to the correct video passage. Confirm the complete operator workflow using returned real recognition events. Keep wrong readings and rejected passages in the report.

Done when each reported outcome is auditable. A failed recognition gate remains open even if the dashboard and demo recording work.

### C10 Documentation and demonstration

Priority: Phase 6. Dependencies: C8/C9 and G5/G6.

Update setup instructions, architecture, evidence tables, presentation and source attribution. Capture a clearly labelled recorded-video demonstration. Provide separate CPU functional instructions and GPU experiment instructions. Report completed work and remaining failures without reusing historical claims as current evidence.

Done when a fresh reviewer can restore the relevant assets, run the synthetic workflow, inspect recorded results and understand the exact validation limits.

## Owner and Codex task list

| ID | Scheduled work | Depends on | Completion evidence |
|---|---|---|---|
| G1 | Restore selected detector/OCR checkpoints and isolated GPU runtime; verify hashes and actual inference | Release assets | Model revisions, provider/runtime checks and bounded inference outputs |
| G2 | Run unchanged models on development recordings; compare automatic crops with marked crops | Agreed source interface; C2 development set | Stage-specific baseline with correct/wrong/abstained outcomes |
| G3 | Evaluate parent/adapted OCR, original-only evidence, checked reconstruction and optional learned restoration; evaluate detector profiles if localization fails | G2; C6 diagnostics | Controlled comparisons with identical labels, inputs and declared budgets |
| G4 | Train or adapt only the stage shown to fail; manage checkpoints, GPU dependencies and exported runtime parity | G3; verified training/validation data | Training receipt, selected checkpoint, validation regressions and CPU/GPU/export comparisons |
| G5 | Run frozen configuration on held-out videos; measure actual GPU latency, memory, two-stream load, overload and endurance | C4/C5/C8; frozen model/protocol | Independent recognition results and resource/failure measurements; accepted output through the central workflow |
| G6 | Review changes and claims, complete final integration, preserve release assets and clean GPU rebuild instructions | C9/C10; G5 | Reproducible artifacts, evidence-linked demonstration and explicit remaining gates |

G4 is conditional. Existing checkpoints may be sufficient on better footage; new training is not automatically required. TensorRT/FP16 or other acceleration is considered only if measured runtime requires it, followed by recognition and parity checks.

## Ownership boundaries and exchanges

- Teammate owns source adapters, recorded metadata, backend/frontend controls, setup for CPU, labels, benchmark scoring, delivery/recovery tests and documentation.
- Owner and Codex own model loading/runtime adapters, experimental detector/OCR selection, neural restoration, training scripts/checkpoints and GPU benchmark execution.
- Existing reconstruction/temporal logic can be developed on CPU by the teammate. Notify the owner before changing shared evidence scoring so GPU comparisons use the same revision.
- Agree the frame contract (native BGR array, source ID, PTS, discontinuity/reset signal) and result contract (track, candidates, acceptance/rejection reason, model version) before parallel edits. Reuse existing contracts where possible.
- Work on separate branches. Use task IDs in commits and handoff notes. Integrate source contracts first, then the adapter, then interfaces, then recognition changes. Avoid simultaneous edits to shared session files.
- Each handoff records commit ID, file/data/model hashes, command, environment, output location and remaining failures. Do not put credentials or plaintext genuine live identifiers in handoff notes.

Useful starting files:

| Area | Existing files |
|---|---|
| Source and session | `edge/sentinel_edge/stream.py`, `tracked_live_viewer.py`, `live_detection_viewer.py`, `recognition_contract.py` |
| Recorded metadata and controls | `backend/app/schemas.py`, `models.py`, `recognition.py`, `frontend/src/RecognitionPanel.tsx`, `recognition.ts`, `edge/sentinel_edge/engineering_page.py` |
| Crops and temporal evidence | `crop_collector.py`, `plate_search.py`, `tracked_ocr.py`, `ctc_fusion.py`, `sequence_fusion.py`, `reconstruction.py`, `live_reconstruction.py` under `edge/sentinel_edge` |
| Model/runtime work | `experimental_detectors.py`, `awiros_experiment.py`, `corpus_ocr_adaptation.py`, `hybrid_models.py` under `edge/sentinel_edge`; `tools/run_ocr_research.py` |
| Delivery and restoration | `edge/sentinel_edge/outbox.py`, `shared_delivery.py`, `backend/app/main.py`, `scripts/restore-teammate-handoff.ps1`, `tools/handoff_archives.py` |

## Work after the recognition milestone

All ordinary backend/frontend hardening, bundle reduction, framework maintenance, retention/access auditing, load-test tooling and regional architecture documentation go to the teammate. Actual GPU capacity measurements and deployment validation remain with the owner. These follow the recorded recognition milestone rather than delaying it.

Registry integration, fleet deployment and an 80,000-camera capacity claim are separate future scope. Do not start them as completion requirements for this prototype.

## First handoff to the teammate

> Work from the current private repository and `handoff-2026-10-07` release. Read `docs/TEAMMATE_HANDOFF.md` and this work split. Your Ryzen 7 machine owns C1 through C10 in dependency order. Start with clean CPU restoration and the synthetic demo, then inventory permitted recorded footage. Record CPU/RAM details and setup failures. Agree the recorded frame/result contracts with us before changing shared session interfaces. We supply model runtime integration, training and GPU measurements. Do not retrain models, change acceptance thresholds or present injected events as recognized plates. Return the setup receipt and draft development/validation/test manifest first.
