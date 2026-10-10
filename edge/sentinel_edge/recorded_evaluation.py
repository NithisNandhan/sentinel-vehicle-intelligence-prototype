"""Score recorded-footage recognition against independent human labels.

Labels come from a person, never from this pipeline's own output. Every number
is reported with its denominator. Nothing here changes the acceptance gate:
the evaluator observes the same `observe_tracked` path the interfaces use.
Reports carry masked plates only; full strings stay in memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import threading
import time

from .privacy import mask_plate

SPLITS = ('development', 'validation', 'test')
IOU_RULE = 0.5
# One sampling interval: a labelled instant and an analysed sample can differ by this much.
TIME_TOLERANCE_MS = 500
RULES = {
    'matching': 'A track belongs to a passage when their video-time ranges overlap (tolerance 500 ms).',
    'vehicle_coverage': 'Over passages with a labelled vehicle box: a track box within 500 ms of a labelled instant has IoU >= 0.5.',
    'plate_localisation': 'Over labelled plate boxes: a predicted plate box within 500 ms has IoU >= 0.5.',
    'ocr_observation_yield': 'Over passages with a visible plate: an overlapping track produced at least one distinct OCR observation.',
    'exact_read': 'Over readable passages: an overlapping track was accepted by the unchanged gate with exactly the labelled string.',
    'accepted_precision': 'Correct accepted readings / (correct + wrong). Undefined when nothing was accepted. Readings on passages no human could read, or outside every labelled passage, are counted separately and excluded.',
    'candidate_recall': 'Over readable passages: the labelled string is at rank 1, within 5, or within K of an overlapping track\'s final ranking.',
    'independence': 'Outcomes are per labelled vehicle passage. Neighbouring frames and reconstructions are not separate successes.',
}


class LabelError(ValueError):
    pass


def normalise(plate):
    return re.sub(r'[^A-Z0-9]', '', str(plate).upper())


def _box(value, where):
    if (not isinstance(value, list) or len(value) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)
            or value[0] >= value[2] or value[1] >= value[3] or min(value) < 0):
        raise LabelError(f'{where}: a box is [x1, y1, x2, y2] in native pixels')
    return tuple(float(v) for v in value)


@dataclass(frozen=True)
class Passage:
    passage_id: str
    vehicle_id: str
    start_ms: int
    end_ms: int
    plate: str | None          # None: no plate visible, or a human could not transcribe it
    readable: bool
    plate_visible: bool
    boxes: tuple = ()           # ({'position_ms', 'vehicle', 'plate'}, ...)


def load_labels(path, *, split):
    """Return ({recording_id: {'sha256', 'passages': [Passage]}}, label file SHA256) for one split."""
    if split not in SPLITS:
        raise LabelError('Split must be development, validation or test')
    raw = Path(path).read_bytes()
    try:
        data = json.loads(raw)
    except ValueError:
        raise LabelError('Label file is not JSON') from None
    if not isinstance(data, dict) or data.get('schema_version') != '1.0' or not isinstance(data.get('recordings'), list):
        raise LabelError('Unsupported label file')
    if not isinstance(data.get('labeller'), str) or not data['labeller'].strip():
        raise LabelError('Labels must name the person who made them; model output is not ground truth')
    selected, vehicles = {}, {}
    for entry in data['recordings']:
        rid = entry.get('recording_id') if isinstance(entry, dict) else None
        if not isinstance(rid, str) or not re.fullmatch(r'rec-[a-f0-9]{12}', rid) or rid in selected:
            raise LabelError('Each recording needs a unique recording_id (rec- plus 12 hex digits)')
        if entry.get('split') not in SPLITS:
            raise LabelError(f'{rid}: split must be development, validation or test')
        if not isinstance(entry.get('sha256'), str) or not entry['sha256'].startswith(rid[4:]):
            raise LabelError(f'{rid}: sha256 must be the labelled file\'s hash')
        passages, seen = [], set()
        for item in entry.get('passages') or []:
            where = f'{rid}/{item.get("passage_id") if isinstance(item, dict) else "?"}'
            if not isinstance(item, dict) or not isinstance(item.get('passage_id'), str) or item['passage_id'] in seen:
                raise LabelError(f'{where}: passages need unique passage_id values')
            seen.add(item['passage_id'])
            start, end = item.get('start_ms'), item.get('end_ms')
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)) or not 0 <= start < end:
                raise LabelError(f'{where}: start_ms and end_ms are video positions in milliseconds')
            readable, plate = item.get('readable'), item.get('plate')
            if not isinstance(readable, bool) or not isinstance(item.get('vehicle_id'), str):
                raise LabelError(f'{where}: readable (true/false) and vehicle_id are required')
            if readable:
                if not isinstance(plate, str) or not 5 <= len(normalise(plate)) <= 12:
                    raise LabelError(f'{where}: a readable passage needs its transcribed plate')
                plate = normalise(plate)
            elif plate is not None:
                raise LabelError(f'{where}: an unreadable passage must have plate null')
            boxes = []
            for row in item.get('boxes') or []:
                position = row.get('position_ms') if isinstance(row, dict) else None
                if isinstance(position, bool) or not isinstance(position, int) or not start <= position <= end:
                    raise LabelError(f'{where}: box position_ms must lie inside the passage')
                boxes.append({'position_ms': position,
                              'vehicle': _box(row['vehicle'], where) if row.get('vehicle') is not None else None,
                              'plate': _box(row['plate'], where) if row.get('plate') is not None else None})
            visible = item.get('plate_visible', readable or any(b['plate'] for b in boxes))
            if not isinstance(visible, bool) or (readable and not visible):
                raise LabelError(f'{where}: plate_visible must be true for a readable passage')
            passages.append(Passage(item['passage_id'], item['vehicle_id'], start, end, plate, readable, visible, tuple(boxes)))
            vehicles.setdefault(item['vehicle_id'], set()).add(entry['split'])
        if entry['split'] == split:
            selected[rid] = {'sha256': entry['sha256'], 'passages': passages}
    leaked = sorted(vehicle for vehicle, splits in vehicles.items() if len(splits) > 1)
    if leaked:
        raise LabelError('The same vehicle appears in more than one split: ' + ', '.join(leaked[:5]))
    return selected, hashlib.sha256(raw).hexdigest()


@dataclass
class TrackOutcome:
    """What the pipeline produced for one vehicle track. Full strings stay in memory."""
    first_ms: int
    last_ms: int
    vehicle_boxes: list = field(default_factory=list)   # (position_ms, box)
    plate_boxes: list = field(default_factory=list)
    observations: int = 0
    candidates: list = field(default_factory=list)      # final ranking, best first
    accepted: list = field(default_factory=list)        # distinct strings, in order


def iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.
    overlap = width * height
    return overlap / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap)


def _near(boxes, position, target):
    return any(abs(when-position) <= TIME_TOLERANCE_MS and iou(box, target) >= IOU_RULE for when, box in boxes)


def _ratio(numerator, denominator):
    return {'count': numerator, 'of': denominator, 'rate': round(numerator/denominator, 4) if denominator else None}


def score(passages, tracks, *, top_k=20):
    """Score one recording. `tracks` is a list of TrackOutcome."""
    def overlapping(passage):
        return [t for t in tracks if t.first_ms <= passage.end_ms+TIME_TOLERANCE_MS
                and t.last_ms >= passage.start_ms-TIME_TOLERANCE_MS]
    rows = []
    boxed = covered = time_overlap = plate_boxes = plate_hits = visible = observed = 0
    readable = correct = wrong_read = abstained = top1 = top5 = topk = 0
    for passage in passages:
        near = overlapping(passage)
        time_overlap += bool(near)
        vehicle_instants = [b for b in passage.boxes if b['vehicle']]
        vehicle_hit = None
        if vehicle_instants:
            boxed += 1
            vehicle_hit = any(_near(t.vehicle_boxes, b['position_ms'], b['vehicle']) for b in vehicle_instants for t in near)
            covered += vehicle_hit
        for b in passage.boxes:
            if b['plate']:
                plate_boxes += 1
                plate_hits += any(_near(t.plate_boxes, b['position_ms'], b['plate']) for t in near)
        has_observation = any(t.observations for t in near)
        if passage.plate_visible:
            visible += 1
            observed += has_observation
        outcome, rank = 'not_readable_by_a_person', None
        accepted = [plate for t in near for plate in t.accepted]
        if passage.readable:
            readable += 1
            ranks = [t.candidates[:top_k].index(passage.plate)+1 for t in near if passage.plate in t.candidates[:top_k]]
            rank = min(ranks) if ranks else None
            top1 += rank == 1; top5 += bool(rank and rank <= 5); topk += bool(rank)
            if passage.plate in accepted:
                outcome = 'correct'; correct += 1
            elif accepted:
                outcome = 'wrong_read'; wrong_read += 1
            else:
                outcome = 'abstained'; abstained += 1
        rows.append({'passage_id': passage.passage_id, 'vehicle_id': passage.vehicle_id,
                     'start_ms': passage.start_ms, 'end_ms': passage.end_ms, 'outcome': outcome,
                     'labelled_plate_masked': mask_plate(passage.plate) if passage.plate else None,
                     'accepted_masked': sorted({mask_plate(p) for p in accepted}),
                     'track_overlaps_in_time': bool(near), 'vehicle_box_matched': vehicle_hit,
                     'ocr_observations': max((t.observations for t in near), default=0),
                     'true_plate_rank': rank,
                     'failing_stage': _stage(passage, near, vehicle_hit, has_observation, rank, outcome)})
    # Accepted readings, each judged once against the passages it overlaps in time.
    accepted_correct = accepted_wrong = on_unreadable = outside = 0
    for track in tracks:
        near = [p for p in passages if track.first_ms <= p.end_ms+TIME_TOLERANCE_MS
                and track.last_ms >= p.start_ms-TIME_TOLERANCE_MS]
        for plate in track.accepted:
            if any(p.readable and p.plate == plate for p in near):
                accepted_correct += 1
            elif not near:
                outside += 1
            elif not any(p.readable for p in near):
                on_unreadable += 1
            else:
                accepted_wrong += 1
    judged = accepted_correct + accepted_wrong
    return {
        'passages': len(passages),
        'time_overlap_coverage': _ratio(time_overlap, len(passages)),
        'vehicle_coverage': _ratio(covered, boxed),
        'plate_localisation': _ratio(plate_hits, plate_boxes),
        'ocr_observation_yield': _ratio(observed, visible),
        'exact_read_coverage': _ratio(correct, readable),
        'wrong_reads': _ratio(wrong_read, readable),
        'abstentions': _ratio(abstained, readable),
        'unreadable_passages': len(passages)-readable,
        'accepted_readings': {'correct': accepted_correct, 'wrong': accepted_wrong,
                              'on_passages_no_person_could_read': on_unreadable, 'outside_labelled_passages': outside},
        # Undefined, not zero and not perfect, when nothing was accepted.
        'accepted_precision': _ratio(accepted_correct, judged) if judged else
                              {'count': 0, 'of': 0, 'rate': None, 'note': 'undefined: nothing was accepted'},
        'candidate_recall': {'top_1': _ratio(top1, readable), 'top_5': _ratio(top5, readable),
                             f'top_{top_k}': _ratio(topk, readable)},
        'per_passage': rows,
    }


def _stage(passage, near, vehicle_hit, has_observation, rank, outcome):
    """First stage that failed, for diagnosis before any tuning."""
    if outcome in ('correct', 'not_readable_by_a_person'):
        return None
    if not near or vehicle_hit is False:
        return 'vehicle_detection_or_tracking'
    if not has_observation:
        return 'plate_localisation_or_crop'
    if rank is None:
        return 'ocr_candidate_missing'
    if outcome == 'wrong_read':
        return 'fusion_or_association'
    return 'acceptance_gate_withheld_rank_%d' % rank


def merge(results):
    """Add up per-recording counts. Rates are recomputed from totals, never averaged."""
    def total(key, inner=None):
        items = [r[key][inner] if inner else r[key] for r in results]
        count, of = sum(i['count'] for i in items), sum(i['of'] for i in items)
        return _ratio(count, of)
    if not results:
        raise ValueError('Nothing was evaluated')
    accepted = {key: sum(r['accepted_readings'][key] for r in results) for key in results[0]['accepted_readings']}
    judged = accepted['correct'] + accepted['wrong']
    recall_keys = list(results[0]['candidate_recall'])
    return {
        'passages': sum(r['passages'] for r in results),
        **{key: total(key) for key in ('time_overlap_coverage', 'vehicle_coverage', 'plate_localisation',
                                       'ocr_observation_yield', 'exact_read_coverage', 'wrong_reads', 'abstentions')},
        'unreadable_passages': sum(r['unreadable_passages'] for r in results),
        'accepted_readings': accepted,
        'accepted_precision': _ratio(accepted['correct'], judged) if judged else
                              {'count': 0, 'of': 0, 'rate': None, 'note': 'undefined: nothing was accepted'},
        'candidate_recall': {key: total('candidate_recall', key) for key in recall_keys},
    }


class _Collector:
    """Stands in for delivery and the interface state; keeps everything in memory."""
    def __init__(self, recording, top_k):
        self.recording, self.top_k = recording, top_k
        self.tracks, self.samples, self.last_position = {}, 0, None
        self.segment = 0

    # LiveState stand-in: snapshots are not needed for scoring.
    def publish(self, snapshot, stop=None):
        pass

    def depth(self):
        return 0

    def accept(self, camera_id, track_id, candidate, pts_ms, *, recorded=None):
        position = pts_ms - self.recording.start_pts_ms
        outcome = self.tracks.setdefault((self.segment, track_id), TrackOutcome(position, position))
        if candidate.plate not in outcome.accepted:
            outcome.accepted.append(candidate.plate)
        return True

    def on_sample(self, frame, tracks, displays, pipeline):
        position = frame.pts_ms - self.recording.start_pts_ms
        # Tracker IDs restart after a scene reset; keep those vehicles distinct.
        self.segment = pipeline.metrics['scene_resets']
        self.samples += 1
        self.last_position = position
        for track in tracks:
            outcome = self.tracks.setdefault((self.segment, track.track_id), TrackOutcome(position, position))
            outcome.last_ms = position
            outcome.vehicle_boxes.append((position, tuple(track.box)))
        for display in displays:
            outcome = self.tracks.setdefault((self.segment, display['track_id']), TrackOutcome(position, position))
            outcome.plate_boxes.append((position, tuple(display['box'])))
            outcome.observations = max(outcome.observations, display['observations'])
            history = pipeline.histories.get(display['track_id'])
            if history is not None and display['candidate_count']:
                outcome.candidates = [c.plate for c in history.reader.candidates()[:self.top_k]]


def analyse_recording(config, recording, model, detector, vehicle, *, detector_profile, top_k=20,
                      plate_recovery=False, stop=None):
    """Run the unchanged shared pipeline over one file, unpaced and without dropping frames."""
    from .recorded_source import RecordedRun, new_session_id
    from .tracked_live_viewer import observe_tracked
    collector = _Collector(recording, top_k)
    started = time.monotonic()
    summary = observe_tracked(config, recording, model, detector, vehicle, collector, stop or threading.Event(),
                              delivery=collector, detector_profile=detector_profile, plate_recovery=plate_recovery,
                              recorded=RecordedRun(recording, new_session_id(), paced=False),
                              on_sample=collector.on_sample)
    wall = time.monotonic()-started
    duration = recording.duration_ms
    return list(collector.tracks.values()), {
        'wall_seconds': round(wall, 2), 'video_seconds': round(duration/1000, 2) if duration else None,
        'speed_relative_to_real_time': round(duration/1000/wall, 3) if duration and wall else None,
        'analysed_frames': summary.get('analysed_frames', 0), 'ocr_calls': summary.get('ocr_calls', 0),
        'vehicle_tracks': len(collector.tracks),
        'reached_end': duration is None or collector.last_position is None or duration-collector.last_position <= 2000,
    }


def evaluate(labels_path, split, recordings_dir, config, models, *, detector_profile, top_k=20,
             plate_recovery=False):
    """Evaluate every labelled recording of one split and return the report."""
    from .recorded_source import RecordingLibrary
    labelled, label_hash = load_labels(labels_path, split=split)
    if not labelled:
        raise LabelError(f'No recordings are labelled for the {split} split')
    available, _ = RecordingLibrary(recordings_dir).scan()
    missing = sorted(set(labelled)-set(available))
    if missing:
        raise LabelError('Labelled recordings are not present with a valid receipt: ' + ', '.join(missing))
    model, detector, vehicle = models
    per_recording, results = [], []
    for recording_id, entry in sorted(labelled.items()):
        recording = available[recording_id]
        if recording.sha256 != entry['sha256']:
            raise LabelError(f'{recording_id}: the labels were made for a different file')
        tracks, runtime = analyse_recording(config, recording, model, detector, vehicle,
                                            detector_profile=detector_profile, top_k=top_k,
                                            plate_recovery=plate_recovery)
        result = score(entry['passages'], tracks, top_k=top_k)
        results.append(result)
        per_recording.append({'recording_id': recording_id, 'runtime': runtime, **result})
    return {'schema_version': '1.0', 'split': split, 'labels_sha256': label_hash,
            'settings': {'detector_profile': detector_profile, 'acceptance_threshold': config.ocr_min_confidence,
                         'sampling_interval_ms': 250 if plate_recovery else 500, 'plate_recovery': plate_recovery,
                         'top_k': top_k, 'pacing': 'unpaced', 'frames_dropped_by_design': 0},
            'rules': RULES, 'totals': merge(results), 'recordings': per_recording}
