"""Bounded live reconstruction; derived OCR is not an independent vote."""
from dataclasses import dataclass, field
import time

from .ctc_fusion import CTCPlateReader, greedy_text
from .reconstruction import reconstruct


@dataclass(repr=False)
class ReconstructionState:
    status: str = 'waiting'
    reason: str = 'Need three distinct original crops surviving alignment'
    attempts: int = 0
    signature: tuple = ()
    result: object | None = None
    rows: object | None = None
    candidates: list = field(default_factory=list)
    elapsed_ms: float = 0.


def update(state, reader, samples, ocr, pts_ms):
    """At most six attempts per evidence segment; native images remain intact."""
    if len(samples) < 3:
        return False
    signature = tuple(s.source_id for s in samples)
    if signature == state.signature:
        return False
    if state.attempts >= 6:
        # Never keep a derived view after its original evidence window changes.
        state.result = state.rows = None
        state.candidates = []
        reader.propose(())
        state.status, state.reason = 'budget_exhausted', 'Six reconstruction attempts used in this track segment; original OCR continues'
        return False
    state.signature = signature
    state.attempts += 1
    state.result = state.rows = None
    state.candidates = []
    reader.propose(())
    started = time.monotonic()
    try:
        # Bound alignment and retained working images independently of capture.
        if any(s.image.nbytes > 1024 * 512 * 3 for s in samples):
            state.status, state.reason = 'unavailable', 'Crop exceeds live reconstruction working-image budget'
            return True
        result = reconstruct(samples)
        if result is None:
            state.status, state.reason = 'alignment_rejected', 'Fewer than three aligned crops or insufficient valid overlap; original OCR continues'
            return True
        rows = ocr.probabilities(result.image)
        diagnostic = CTCPlateReader(ocr.tokens)
        diagnostic.add(rows, timestamp_ms=pts_ms, source_id='derived-reconstruction')
        state.result, state.rows = result, rows
        state.candidates = diagnostic.candidates()[:32]
        reader.propose(c.plate for c in state.candidates)
        state.status, state.reason = 'reconstructed', 'ECC aligned native-grid fusion; suggestions rescored on originals, zero extra observations'
    except Exception:
        # Do not leak model errors, input text or paths to either interface.
        state.result = state.rows = None
        state.candidates = []
        reader.propose(())
        state.status, state.reason = 'error', 'Reconstruction or derived OCR failed; original OCR continues'
    finally:
        state.elapsed_ms = (time.monotonic() - started) * 1000
    return True


def disagreement(state, accepted, threshold):
    if state is None or accepted is None or not state.candidates:
        return False
    best = state.candidates[0]
    return (best.score >= threshold and best.minimum_character_score >= .75
            and best.plate != accepted.plate)


def snapshot(state, tokens):
    from .live_detection_viewer import encoded
    from .recognition_debug import characters
    value = dict(status=state.status, reason=state.reason, attempts=state.attempts,
                 attempt_limit=6, elapsed_ms=round(state.elapsed_ms, 2),
                 extra_observations=0, learned_restoration='not_run')
    if state.result is None:
        return value
    result = state.result
    value.update(image=encoded(result.image, '.png'), width=result.image.shape[1],
                 height=result.image.shape[0], alignment=result.alignment,
                 contributing_originals=len(result.source_ids), rejected_originals=result.rejected,
                 valid_fraction=round(float(result.valid_mask.mean()), 3),
                 reference_pts_ms=result.reference_pts_ms,
                 text=greedy_text(state.rows, tokens), characters=characters(state.rows, tokens),
                 candidates=[dict(plate=c.plate, score=round(c.score, 4)) for c in state.candidates])
    return value
