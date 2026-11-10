"""Bounded, source-PTS vehicle identities; predictions never become observations."""
import math
from collections import Counter, OrderedDict, deque
from numbers import Integral, Real


def _number(value):
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


class VehicleLifecycle:
    """Conservative geometry/appearance association with bounded active state.

    Only qualified observations create parents. Eligibility then persists, but
    callers must check ``observed`` before owner confirmation or evidence work.
    Signatures are normalized on input. Closed summaries retain at most 8 times
    max_active records; lifetime counters remain available after pruning.
    """

    def __init__(self, frame_shape, max_prediction_ms=1000, max_lost_ms=5000,
                 max_active=128, excluded=()):
        if (len(frame_shape) != 2 or any(not _number(n) or n <= 0 for n in frame_shape)
                or not _number(max_prediction_ms) or max_prediction_ms < 0
                or not _number(max_lost_ms) or max_lost_ms < max_prediction_ms
                or isinstance(max_active, bool) or not isinstance(max_active, int) or max_active < 1):
            raise ValueError('Invalid lifecycle bounds')
        self.height, self.width = frame_shape
        self.max_prediction_ms, self.max_lost_ms = max_prediction_ms, max_lost_ms
        self.max_active, self.excluded = max_active, set(excluded)
        self.counters = Counter()
        self._active, self._closed = {}, OrderedDict()
        self._pts, self._sequence = None, 0

    @property
    def native_aliases(self):
        return {native: key for key, state in self._active.items() for native in state['native_ids']}

    @property
    def records(self):
        return {**{key: dict(record) for key, record in self._closed.items()},
                **{key: self._summary(state) for key, state in self._active.items()}}

    def _summary(self, state):
        return {key: state[key] for key in ('id', 'vehicle_class', 'first_pts_ms',
                'last_observed_pts_ms', 'observation_count', 'status')} | {
                    'native_ids': list(state['native_ids']), 'qualified': True}

    def _close(self, key, reason, pts):
        record = self._summary(self._active.pop(key))
        record.update(status='closed', close_reason=reason, closed_pts_ms=pts)
        self._closed[key] = record
        self.counters[reason] += 1
        while len(self._closed) > self.max_active * 8:
            self._closed.popitem(last=False)
            self.counters['records_pruned'] += 1

    def _predicted(self, state, pts):
        last_pts, box = state['history'][-1]
        vx = (state['velocity'][0]+state['velocity'][2])/2
        vy = (state['velocity'][1]+state['velocity'][3])/2
        velocity = [vx, vy, vx, vy]  # Preserve observed size; extrapolated scale can invert boxes.
        dt = min(pts-last_pts, self.max_prediction_ms)
        return [b+v*dt for b, v in zip(box, velocity)], velocity

    def _exit(self, state, pts):
        box, velocity = self._predicted(state, pts)
        last = state['history'][-1][1]
        margin = max(5., min(self.width, self.height)*.03)
        return ((box[2] < 0 and last[0] <= margin and velocity[0] < 0 and velocity[2] < 0)
                or (box[0] > self.width and last[2] >= self.width-margin and velocity[0] > 0 and velocity[2] > 0)
                or (box[3] < 0 and last[1] <= margin and velocity[1] < 0 and velocity[3] < 0)
                or (box[1] > self.height and last[3] >= self.height-margin and velocity[1] > 0 and velocity[3] > 0))

    def _geometry(self, state, obs, pts):
        a, _ = self._predicted(state, pts)
        b = obs['box']
        area_ratio = ((b[2]-b[0])*(b[3]-b[1])) / ((a[2]-a[0])*(a[3]-a[1])) if a[2] > a[0] and a[3] > a[1] else 0
        distance = math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)
        diagonal = math.hypot(a[2]-a[0], a[3]-a[1])
        return .25 <= area_ratio <= 4 and distance <= max(20., diagonal*.9)

    def _score(self, state, obs, pts):
        if not self._geometry(state, obs, pts):
            return None
        if obs['id'] in state['native_ids']:
            return 2.
        if state['vehicle_class'] != obs['vehicle_class']:
            return None
        signature = obs['signature']
        previous = state['signatures']
        cosine = max((sum(a*b for a, b in zip(signature, old)) for old in previous
                      if len(old) == len(signature)), default=-1.) if signature else -1.
        motion = obs['motion_support'].get(state['id'], 0.)
        return cosine if cosine >= .85 or (motion >= .3 and cosine >= .60) else None

    def _validate(self, observations, pts):
        if not _number(pts) or pts < 0 or (self._pts is not None and pts <= self._pts):
            raise ValueError('Strictly increasing finite source PTS required')
        clean, seen = [], set()
        for raw in observations:
            native, box = raw['id'], list(raw['box'])
            if native is None or native in seen:
                raise ValueError('Unique non-null native IDs required')
            seen.add(native)
            if len(box) != 4 or any(not _number(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('Finite positive-area xyxy box required')
            cls = raw['vehicle_class']
            raw_signature = raw.get('signature')
            signature = list(raw_signature) if raw_signature is not None else []
            if isinstance(cls, bool) or not isinstance(cls, Integral) or not isinstance(raw['qualified'], bool):
                raise ValueError('Integer class and boolean qualification required')
            if any(not _number(v) for v in signature):
                raise ValueError('Finite signature required')
            norm = math.hypot(*signature)
            if signature and norm == 0:
                raise ValueError('Nonzero signature required')
            motion = dict(raw.get('motion_support') or {})
            if any(not _number(v) or not 0 <= v <= 1 for v in motion.values()):
                raise ValueError('Motion IoU must be finite and between zero and one')
            clean.append(dict(id=native, box=box, vehicle_class=cls, qualified=raw['qualified'],
                              signature=[v/norm for v in signature], motion_support=motion))
        return clean

    def update(self, observations, pts_ms, scene_reset=False):
        observations = self._validate(observations, pts_ms)  # No mutation before full validation.
        self._pts = pts_ms
        if scene_reset:
            for key in list(self._active):
                self._close(key, 'scene_reset', pts_ms)
        for key, state in list(self._active.items()):
            if pts_ms-state['last_observed_pts_ms'] > self.max_lost_ms:
                self._close(key, 'unresolved_timeout', pts_ms)
        eligible = [obs for obs in observations if obs['id'] not in self.excluded]
        self.counters['excluded_observations'] += len(observations)-len(eligible)
        scores = {i: {key: score for key, state in self._active.items()
                       if (score := self._score(state, obs, pts_ms)) is not None}
                  for i, obs in enumerate(eligible)}
        assignments, used = {}, set()
        # Require a unique best match in both directions; ties abstain.
        for i, choices in scores.items():
            ranked = sorted(choices.items(), key=lambda pair: pair[1], reverse=True)
            if not ranked or (len(ranked) > 1 and ranked[0][1]-ranked[1][1] < .1):
                continue
            key, score = ranked[0]
            rivals = [candidate[key] for j, candidate in scores.items() if j != i and key in candidate]
            if key in used or (rivals and score-max(rivals) < .1):
                self.counters['ambiguous_associations'] += 1
                continue
            assignments[i] = key
            used.add(key)
        for i, obs in enumerate(eligible):
            key = assignments.get(i)
            if key is None:
                if not obs['qualified']:
                    continue
                if len(self._active) >= self.max_active:
                    self.counters['capacity_rejected'] += 1
                    continue
                self._sequence += 1
                key = f'VT{self._sequence:04d}'
                self._active[key] = dict(id=key, vehicle_class=obs['vehicle_class'],
                    first_pts_ms=pts_ms, native_ids=deque(maxlen=32), history=deque(maxlen=2),
                    signatures=deque(maxlen=2), classes=deque(maxlen=10),velocity=[0.]*4, observation_count=0)
                self.counters['created'] += 1
            state = self._active[key]
            # A reused native ID belongs only to its most recent valid parent.
            for other in self._active.values():
                if other is not state and obs['id'] in other['native_ids']:
                    other['native_ids'].remove(obs['id'])
            if obs['id'] not in state['native_ids']:
                state['native_ids'].append(obs['id'])
            if state['history']:
                previous_pts, previous_box = state['history'][-1]
                dt = pts_ms-previous_pts
                measured = [(b-a)/dt for a, b in zip(previous_box, obs['box'])]
                alpha = 1. if state['observation_count'] == 1 else -math.expm1(-dt/250.)
                state['velocity'] = [a+alpha*(b-a) for a, b in zip(state['velocity'], measured)]
            state['history'].append((pts_ms, obs['box']))
            state['classes'].append(obs['vehicle_class'])
            votes=Counter(state['classes']);peak=max(votes.values())
            if votes[state['vehicle_class']]<peak:
                state['vehicle_class']=next(cls for cls in reversed(state['classes']) if votes[cls]==peak)
            if obs['signature']:
                state['signatures'].append(obs['signature'])
            state.update(last_observed_pts_ms=pts_ms, status='visible')
            state['observation_count'] += 1
            used.add(key)
        output = []
        for key, state in list(self._active.items()):
            observed = state['last_observed_pts_ms'] == pts_ms
            gap = pts_ms-state['last_observed_pts_ms']
            if not observed and self._exit(state, pts_ms):
                self._close(key, 'boundary_exit', pts_ms)
                continue
            state['status'] = 'visible' if observed else ('predicted' if gap <= self.max_prediction_ms else 'lost_unresolved')
            box = self._predicted(state, pts_ms)[0] if gap <= self.max_prediction_ms else None
            output.append(self._summary(state) | dict(box=box, observed=observed,
                observed_box=list(state['history'][-1][1]) if observed else None,
                signature=list(state['signatures'][-1]) if state['signatures'] else [],
                velocity_px_per_ms=[(state['velocity'][0]+state['velocity'][2])/2,
                                    (state['velocity'][1]+state['velocity'][3])/2],
                observation_span_ms=state['last_observed_pts_ms']-state['first_pts_ms']))
        return output
