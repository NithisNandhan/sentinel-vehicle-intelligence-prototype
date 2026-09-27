"""Observable OCR diagnostics, not calibrated confidence or model reasoning."""
from .ctc_fusion import greedy_text


def characters(rows, tokens):
    result, previous = [], None
    for row in rows:
        index = max(range(len(row)), key=row.__getitem__)
        if index and index != previous:
            choices = sorted(range(1, len(row)), key=row.__getitem__, reverse=True)[:3]
            result.append({'character': tokens[index], 'score': round(float(row[index]), 4),
                           'alternatives': [{'character': tokens[i], 'score': round(float(row[i]), 4)} for i in choices]})
        previous = index
    return result[:25]


def gate_reasons(reader, candidates, threshold):
    reasons = []
    if reader.observation_count < 3:
        reasons.append('Need at least three distinct original observations')
    if not candidates:
        reasons.append('No complete supported Indian-format candidate in the retained search')
        return reasons
    best = candidates[0]
    if best.score < threshold:
        reasons.append(f'Mean character score is below {threshold:g}')
    if best.minimum_character_score < .75:
        reasons.append('Weakest character score is below 0.75')
    if len(candidates) > 1 and best.log_score - candidates[1].log_score < .5:
        reasons.append('Leading candidate is too close to an alternative (log margin below 0.5)')
    peaks = reader._peaks.get(best.plate, [])
    if any(sum(values[i] >= .6 for values in peaks) < 2 for i in range(len(best.plate))):
        reasons.append('Each character needs support of at least 0.6 in two distinct frames')
    if not reasons:
        reasons.append('Model gates passed; correctness has not been independently verified')
    return reasons


def describe_track(track_id, history, samples, threshold):
    from .live_detection_viewer import encoded
    reader = history.reader
    candidates = reader.candidates() if reader.observation_count >= 3 else []
    accepted = reader.accepted(threshold) if candidates else None
    lookup = {sample.source_id: sample for sample in samples}
    frames = []
    for pts, source, rows, _, quality in reader.samples:
        sample = lookup.get(source)
        if sample is None:
            continue
        frames.append({'pts_ms': pts, 'width': sample.image.shape[1], 'height': sample.image.shape[0],
                       'quality': round(quality, 3), 'text': greedy_text(rows, reader.tokens),
                       'characters': characters(rows, reader.tokens), 'image': encoded(sample.image, '.png')})
    from .privacy import mask_plate
    reasons = gate_reasons(reader, candidates, threshold)
    if not reader.observation_count:
        reasons.insert(0, history.last_plate_failure or ('Vehicle ROI has not been searched in the rotating budget yet' if not history.searches else 'No distinct usable plate observation'))
    return {'track_id': track_id, 'vehicle_box': history.box, 'vehicle_class': history.vehicle_class,
            'seen_pts_ms': history.seen_pts, 'observations': reader.observation_count,
            'recognition_state': 'model_accepted' if accepted else 'uncertain' if candidates else 'reading' if reader.observation_count < 3 else 'unreadable',
            'masked_plate': mask_plate(candidates[0].plate) if candidates else None,
            'candidates': [{'plate': c.plate, 'score': round(c.score, 4),
                            'minimum_character_score': round(c.minimum_character_score, 4),
                            'log_score': round(c.log_score, 4), 'observations': c.observations} for c in candidates],
            'gate_reasons': reasons, 'frames': frames,
            'reconstruction': {'status': 'not_run', 'reason': 'Native OCR baseline; reconstruction is not enabled in this session'}}
