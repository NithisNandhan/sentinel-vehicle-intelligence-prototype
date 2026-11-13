"""Human-reviewed accuracy of an immutable, complete recorded-video run.

Ground truth never enters inference. Track fragments are grouped only by the
reviewer's explicit duplicate links, and missed vehicles remain in denominators.
All outputs describe this previously developed-on clip, not unseen CCTV accuracy.
"""
from collections import defaultdict
import hashlib
import json
import math
import re


STATUSES = frozenset({'pending', 'readable', 'unreadable', 'no_plate', 'not_vehicle', 'duplicate'})
VEHICLE_STATUSES = frozenset({'readable', 'unreadable', 'no_plate'})


def _number(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{where} must be a finite number')
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _human_plate(value):
    if not isinstance(value, str) or len(value) > 64 or not re.fullmatch(r'[A-Za-z0-9\s.\-:/]+', value):
        raise ValueError('A readable plate needs a complete alphanumeric transcription')
    result = re.sub(r'[\s.\-:/]', '', value).upper()
    if not re.fullmatch(r'[A-Z0-9]{4,16}', result):
        raise ValueError('A readable plate needs 4 to 16 letters/digits; mark uncertainty unreadable')
    return result


def _prediction(value):
    # Keep ? intact: removing an unknown character could create a false exact match.
    if value is None or value == '':
        return None
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError('Invalid frozen prediction')
    result = re.sub(r'[\s.\-:/]', '', value).upper()
    if not re.fullmatch(r'[A-Z0-9?]{1,32}', result):
        raise ValueError('Invalid frozen prediction characters')
    return result


def _manifest_ids(manifest):
    tracks = manifest.get('tracks')
    if not isinstance(tracks, list):
        raise ValueError('Review manifest must contain tracks')
    ids = [row.get('id') for row in tracks if isinstance(row, dict)]
    if len(ids) != len(tracks) or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Review manifest track IDs must be unique strings')
    return ids


def validate_review(manifest, labels, manifest_sha256, require_complete=False):
    """Return a canonical review document; reject altered identities or invalid truth."""
    track_ids = _manifest_ids(manifest)
    if not isinstance(labels, dict) or labels.get('schema_version') != 1:
        raise ValueError('Unsupported human review schema')
    if not re.fullmatch(r'[a-f0-9]{64}', manifest_sha256 or '') or labels.get('manifest_sha256') != manifest_sha256:
        raise ValueError('Labels belong to a different frozen run')
    labeller = labels.get('labeller', '')
    if not isinstance(labeller, str) or len(labeller) > 160 or (require_complete and not labeller.strip()):
        raise ValueError('Name the person who reviewed the video')
    inventory = labels.get('inventory_complete', False)
    if not isinstance(inventory, bool):
        raise ValueError('Full-video inventory declaration must be true or false')
    revision = labels.get('revision', 0)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError('Invalid label revision')
    extras = labels.get('extras', [])
    if not isinstance(extras, list) or len(extras) > 1000:
        raise ValueError('Invalid missed-vehicle list')
    source = manifest.get('source', {})
    width = _number(source.get('width'), 'Source width')
    height = _number(source.get('height'), 'Source height')
    duration = _number(source.get('duration_ms'), 'Source duration')
    if min(width, height, duration) <= 0:
        raise ValueError('Invalid source dimensions or duration')
    known = set(track_ids)
    clean_extras = []
    for extra in extras:
        if not isinstance(extra, dict):
            raise ValueError('Invalid missed-vehicle entry')
        eid = extra.get('id')
        if not isinstance(eid, str) or not re.fullmatch(r'manual-[a-f0-9-]{8,64}', eid) or eid in known:
            raise ValueError('Missed vehicles need unique manual IDs')
        known.add(eid)
        position = _number(extra.get('position_ms'), 'Missed-vehicle timestamp')
        if not 0 <= position <= duration:
            raise ValueError('Missed-vehicle timestamp is outside the recording')
        box = extra.get('box')
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError('Missed-vehicle box must contain four native coordinates')
        box = [_number(v, 'Box coordinate') for v in box]
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError('Missed-vehicle box is outside the original video')
        image = extra.get('image_path')
        if not isinstance(image, str) or '..' in image or not re.fullmatch(r'assets/[A-Za-z0-9_-][A-Za-z0-9_.-]{0,140}\.(?:png|jpe?g|webp)', image):
            raise ValueError('Missed-vehicle evidence path is invalid')
        clean_extras.append(dict(id=eid, position_ms=position, box=box, image_path=image))
    entries = labels.get('labels', [])
    if not isinstance(entries, list) or len(entries) > len(known):
        raise ValueError('Invalid human label list')
    clean = {}
    for item in entries:
        if not isinstance(item, dict) or item.get('id') not in known or item.get('id') in clean:
            raise ValueError('Unknown or duplicate human label ID')
        sid = item['id']
        status = item.get('status')
        if status not in STATUSES:
            raise ValueError('Unknown human readability status')
        note = item.get('note', '')
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError('A review note must be text of at most 2000 characters')
        plate = None
        if status == 'readable':
            try:
                plate = _human_plate(item.get('plate'))
            except ValueError:
                if require_complete:
                    raise
                status = 'pending'
        elif item.get('plate') not in (None, ''):
            raise ValueError('Only a readable label can contain a plate transcription')
        target = item.get('duplicate_of')
        if status == 'duplicate':
            if target not in known or target == sid:
                raise ValueError('A duplicate must point to another known vehicle entry')
        elif target not in (None, ''):
            raise ValueError('Only duplicate entries can have a duplicate target')
        clean[sid] = dict(id=sid, status=status, plate=plate,
                          duplicate_of=target if status == 'duplicate' else None, note=note.strip())
    for sid in track_ids + [e['id'] for e in clean_extras]:
        clean.setdefault(sid, dict(id=sid, status='pending', plate=None, duplicate_of=None, note=''))
    for sid, item in clean.items():
        current = sid
        visited = set()
        while clean[current]['status'] == 'duplicate':
            if current in visited:
                raise ValueError('Duplicate links contain a cycle')
            visited.add(current)
            current = clean[current]['duplicate_of']
        if item['status'] == 'duplicate' and clean[current]['status'] == 'not_vehicle':
            raise ValueError('A duplicate cannot resolve to a non-vehicle entry')
    if require_complete and any(r['status'] == 'pending' for r in clean.values()):
        raise ValueError('Review every detected track and added vehicle before calculating accuracy')
    return dict(schema_version=1, manifest_sha256=manifest_sha256, labeller=labeller.strip(),
                inventory_complete=inventory, revision=revision, labels=list(clean.values()), extras=clean_extras)


def _ratio(numerator, denominator, reason=None):
    result = dict(numerator=numerator, denominator=denominator,
                  value=numerator / denominator if denominator else None)
    if reason:
        result['reason'] = reason
    return result


def score_review(manifest, predictions, labels, manifest_sha256):
    """Score complete human passages without selecting whichever guess matches truth."""
    truth = validate_review(manifest, labels, manifest_sha256, require_complete=True)
    if manifest.get('run', {}).get('reached_eof') is not True:
        raise ValueError('The frozen analysis did not reach the end of the recording')
    if (not isinstance(predictions, dict) or predictions.get('schema_version') != 1
            or predictions.get('manifest_sha256') != manifest_sha256
            or predictions.get('source_sha256') != manifest.get('source', {}).get('sha256')):
        raise ValueError('Predictions do not belong to this frozen recording/run')
    auto = {r['id']: r for r in manifest['tracks']}
    values = predictions.get('tracks')
    if not isinstance(values, dict) or set(values) != set(auto):
        raise ValueError('Frozen predictions must cover exactly the recorded track IDs')
    strongest = predictions.get('prediction_policy') == 'passed_vehicles_strongest_window_v1'
    if strongest and manifest.get('settings', {}).get('prediction_policy') != predictions['prediction_policy']:
        raise ValueError('Prediction selection policy does not match the frozen manifest')
    clean_predictions = {}
    for sid, row in values.items():
        if not isinstance(row, dict):
            raise ValueError('Invalid frozen track prediction')
        top = _prediction(row.get('top_guess'))
        candidates = row.get('candidates', [])
        accepted = row.get('accepted', [])
        if not isinstance(candidates, list) or not isinstance(accepted, list):
            raise ValueError('Frozen candidates/acceptances must be lists')
        ranked = [_prediction(c.get('plate') if isinstance(c, dict) else c) for c in candidates]
        ranked = list(dict.fromkeys(c for c in ranked if c))
        accepted_strings = [_prediction(c.get('plate') if isinstance(c, dict) else c) for c in accepted]
        accepted_strings = list(dict.fromkeys(c for c in accepted_strings if c))
        if top and ranked and top != ranked[0]:
            raise ValueError('Frozen top guess disagrees with its candidate ranking')
        if top and not ranked:
            ranked = [top]
        if not top and ranked:
            raise ValueError('Frozen candidate ranking has no top guess')
        candidate_ms = row.get('candidate_ms')
        if candidate_ms is None:
            candidate_ms = row.get('last_ms', auto[sid]['last_ms'])
        candidate_ms = _number(candidate_ms, 'Candidate timestamp')
        strength = ()
        if strongest and top:
            raw = row.get('selection_strength')
            if (not isinstance(raw, list) or len(raw) != 5
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in raw)
                    or type(raw[0]) is not int or not 0 <= raw[0] <= 3
                    or type(raw[1]) is not int or not 0 <= raw[1] <= 3
                    or not 0 <= raw[2] <= 1 or not 0 <= raw[3] <= 1
                    or type(raw[4]) is not int or raw[4] < raw[1] or raw[1] != min(raw[4], 3)):
                raise ValueError('Strongest-window prediction lacks valid frozen strength')
            strength = tuple(raw)
        clean_predictions[sid] = dict(top=top, ranked=ranked, accepted=accepted_strings,
                                     candidate_ms=candidate_ms, strength=strength)
    by_id = {r['id']: r for r in truth['labels']}
    groups = defaultdict(list)
    for sid in by_id:
        root = sid
        while by_id[root]['status'] == 'duplicate':
            root = by_id[root]['duplicate_of']
        groups[root].append(sid)
    passages = []
    accepted_correct = accepted_wrong = accepted_unverifiable = 0
    for root, members in groups.items():
        label = by_id[root]
        status = label['status']
        auto_ids = [sid for sid in members if sid in auto]
        candidates = [(clean_predictions[sid]['candidate_ms'], sid, clean_predictions[sid])
                      for sid in auto_ids if clean_predictions[sid]['top']]
        # Legacy captures keep their original latest-window policy. New captures
        # use the frozen display strength; truth never participates in selection.
        chosen = (min(candidates, default=None, key=lambda x: (tuple(-v for v in x[2]['strength']), x[0], x[1]))
                  if strongest else max(candidates, default=None, key=lambda x: (x[0], x[1])))
        chosen = chosen[2] if chosen else dict(top=None, ranked=[])
        accepts = sorted({plate for sid in auto_ids for plate in clean_predictions[sid]['accepted']})
        plate = label['plate']
        correct_accepts = int(bool(plate and plate in accepts))
        if status == 'unreadable':
            wrong_accepts = None
            accepted_unverifiable += len(accepts)
        else:
            wrong_accepts = sum(value != plate for value in accepts)
            accepted_wrong += wrong_accepts
            accepted_correct += correct_accepts
        has_crop = any(auto[sid].get('plate_crop_count', 0) > 0 for sid in auto_ids)
        exact = chosen['top'] == plate if status == 'readable' else False if status in ('no_plate', 'not_vehicle') else None
        rank = chosen['ranked'].index(plate) + 1 if plate and plate in chosen['ranked'] else None
        passages.append(dict(id=root, status=status, plate=plate, top_guess=chosen['top'], exact=exact,
                             accepted=accepts, accepted_correct=correct_accepts, accepted_wrong=wrong_accepts,
                             rank=rank, auto_track_ids=auto_ids, linked_entry_ids=members,
                             has_plate_crop=has_crop, missed_vehicle=not auto_ids and status in VEHICLE_STATUSES,
                             ever_wrong_acceptance=bool(wrong_accepts) if wrong_accepts is not None else None))
    vehicles = [p for p in passages if p['status'] in VEHICLE_STATUSES]
    readable = [p for p in vehicles if p['status'] == 'readable']
    cropped = [p for p in readable if p['has_plate_crop']]
    guessed = [p for p in readable if p['top_guess']]
    detected = sum(bool(p['auto_track_ids']) for p in vehicles)
    counts = dict(automatic_tracks=len(auto), human_entries=len(by_id),
                  duplicate_entries=sum(r['status'] == 'duplicate' for r in by_id.values()),
                  manual_entries=len(truth['extras']), vehicle_passages=len(vehicles),
                  readable_passages=len(readable), unreadable_passages=sum(p['status'] == 'unreadable' for p in vehicles),
                  no_plate_passages=sum(p['status'] == 'no_plate' for p in vehicles),
                  false_auto_tracks=sum(len(p['auto_track_ids']) for p in passages if p['status'] == 'not_vehicle'),
                  detected_vehicle_passages=detected, missed_vehicle_passages=len(vehicles)-detected,
                  readable_with_plate_crop=len(cropped), readable_with_guess=len(guessed),
                  exact_best_guesses=sum(p['exact'] is True for p in readable),
                  accepted_correct=accepted_correct, accepted_wrong=accepted_wrong,
                  accepted_unverifiable=accepted_unverifiable,
                  passages_with_wrong_acceptance=sum(p['ever_wrong_acceptance'] is True for p in passages))
    metrics = dict(
        best_guess_exact_coverage=_ratio(counts['exact_best_guesses'], len(readable)),
        ocr_exact_on_cropped_readable=_ratio(sum(p['exact'] is True for p in cropped), len(cropped)),
        guess_precision_on_readable=_ratio(sum(p['exact'] is True for p in guessed), len(guessed)),
        plate_crop_coverage_on_readable=_ratio(len(cropped), len(readable)),
        accepted_exact_coverage=_ratio(sum(p['accepted_correct'] for p in readable), len(readable)),
        accepted_precision=_ratio(accepted_correct, accepted_correct+accepted_wrong),
        top5_candidate_recall=_ratio(sum(p['rank'] is not None and p['rank'] <= 5 for p in readable), len(readable)),
        top20_candidate_recall=_ratio(sum(p['rank'] is not None and p['rank'] <= 20 for p in readable), len(readable)),
        vehicle_detection_coverage=(_ratio(detected, len(vehicles)) if truth['inventory_complete']
                                   else dict(numerator=None, denominator=None, value=None,
                                             reason='Reviewer has not declared a complete full-video vehicle inventory')))
    return dict(schema_version=1, manifest_sha256=manifest_sha256,
                source_sha256=manifest['source']['sha256'], labels_canonical_sha256=_hash(truth),
                predictions_canonical_sha256=_hash(predictions), labeller=truth['labeller'],
                revision=truth['revision'], inventory_complete=truth['inventory_complete'],
                models=manifest.get('models', {}), run=manifest['run'], counts=counts, metrics=metrics,
                passages=passages, audit=dict(
                    scope=('Complete human-declared vehicle inventory. ' if truth['inventory_complete'] else
                           'Reviewed tracks and manually added samples only; full-video miss coverage is not established. ')
                          + 'Frozen run on the previously developed-on Mumbai clip; not a fresh unseen or independent-camera test.',
                    unit='Human-labelled vehicle passage; duplicate tracker fragments are explicitly linked by the reviewer.',
                    best_guess=('Strongest frozen fusion window across linked fragments; equal strength uses earliest timestamp then track ID. No truth-based selection.'
                                if strongest else 'Latest nonempty frozen hypothesis across linked fragments; no ground-truth-based selection.'),
                    precision='Distinct accepted strings per passage; earlier wrong strings remain errors even if later corrected. Unreadable truth is unverifiable and excluded from precision.',
                    inventory='End-to-end detection coverage requires the reviewer to review the complete clip and add all missed vehicles.',
                    scores='OCR model scores are not calibrated correctness probabilities.',
                    no_training=True, labels_used_for_inference=False))
