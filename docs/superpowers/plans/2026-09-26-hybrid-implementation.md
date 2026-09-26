# Hybrid implementation checklist

Approved design: ../specs/2026-09-26-hybrid-plate-recognition-design.md.

1. Add native crop buffers and test copying, timestamps, caps and cleanup.
2. Add checked ECC reconstruction and optional learned-flow alignment. Test
   identity, known motion, bad alignment, masks and distinct-observation support.
3. Add isolated, provenance-checked RAFT/NAFNet loaders; benchmark availability,
   GPU memory and fallback. Never silently substitute missing model weights.
4. Add sequence-aware original-frame fusion and bounded two-row hypotheses.
   Test insertions/deletions, EOS, unknowns and correlated restoration outputs.
5. Integrate opt-in hybrid mode and a separate licensed local HTML viewer.
   Keep live privacy and aggregate JSON unchanged. Test ranking, escaping and
   NO-READ output; never accept just to fill a result list.
6. Run regression and licensed multi-frame comparisons; inspect the viewer;
   record exact measured outcomes and unfinished gates. Commit and push code
   only after verification, excluding assets, labels, credentials and reports.

Each checkpoint gets a progress update. Existing temporal-v2 remains default.
The optional writing-plans skill is absent; this is the explicit fallback plan.

## Execution record

Checkpoints 1-5 implemented and unit tested. Checkpoint 6: 134 edge tests, 17 API
tests and the synthetic workflow passed; licensed GPU comparisons, no-deblurring
ablation and desktop/mobile viewer inspection completed. See
../../HYBRID_RECOGNITION.md for measured limits. Accuracy/promotion, broader
held-out evaluation, learned-model parity and live shadow gates remain unmet.
