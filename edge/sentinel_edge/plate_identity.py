"""Bounded plate association heuristics, independent of vehicle tracker IDs.

This is not proof of physical identity. A continuation needs appearance or
strong complete OCR agreement as well as geometry. Predicted center distance
must be <= 1.5 * max(previous diagonal, current diagonal, 160 native pixels).
The 160 pixel floor permits small plates to move between sampled frames.
Only two observations and one strongest OCR anchor are retained per group.
"""
from collections import Counter
from dataclasses import dataclass
import math

from .owned_plate import box_iou


def _finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _valid_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(_finite(v) for v in box):
        return False
    width, height = box[2] - box[0], box[3] - box[1]
    return (width > 0 and height > 0
            and all(_finite(v) for v in (width, height, width * height,
                                         box[0] + width / 2, box[1] + height / 2)))


def _signature(values):
    if not isinstance(values, (tuple, list)) or not values or not all(_finite(v) for v in values):
        return None
    norm = math.hypot(*values)
    return tuple(v / norm for v in values) if math.isfinite(norm) and norm > 0 else None


def _reading(values):
    if not isinstance(values, (tuple, list)):
        return None
    result = []
    for value in values:
        if (not isinstance(value, (tuple, list)) or len(value) != 2
                or not isinstance(value[0], str) or len(value[0]) != 1
                or not _finite(value[1]) or not 0 <= value[1] <= 1):
            return None
        result.append((value[0], float(value[1])))
    return tuple(result)


def _strength(reading):
    return sum(p >= .9 for _, p in reading), sum(p for _, p in reading)


def _conflict(left, right):
    return (len(left) == len(right)
            and sum(a != b and pa >= .9 and pb >= .9
                    for (a, pa), (b, pb) in zip(left, right)) >= 3)


def _same_text(left, right):
    return (len(left) >= 7 and len(left) == len(right)
            and all(a == b for (a, _), (b, _) in zip(left, right))
            and sum(p for _, p in left) / len(left) >= .85
            and sum(p for _, p in right) / len(right) >= .85)


def _geometry(box):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return width, height, x1 + width / 2, y1 + height / 2


@dataclass
class _Group:
    observations: list
    anchor: tuple
    candidate: object = None
    owner_votes: int = 0


class PlateIdentityTracker:
    """One observation per plate group per strictly advancing video timestamp.

    Input proposals have global xyxy ``box``, ``confidence``, finite nonzero
    ``signature`` vector and greedy OCR ``reading`` pairs (possibly empty).
    Other fields are copied unchanged. ``rows`` never supplies association votes.
    Vehicle boxes, including excluded/tentative neighbours, veto ownership when
    they cover >=20% of the plate. An eligible unique owner must cover >=90%.
    ``owner`` is published only after two supporting independent timestamps;
    ``owner_candidate`` exposes an eligible current candidate before that.
    Ambiguous frames never publish a historical owner lock.
    """
    max_gap_ms = 1000
    max_active = 512

    def __init__(self, excluded=(), *, retain_owned=False):
        self.excluded = set(excluded)
        self.retain_owned = retain_owned
        self.retained_parent_ids = set()
        self.counters = Counter()
        self.groups = {}
        self.last_pts = None
        self._serial = 0

    def _edge(self, group, current, pts_ms, identity):
        previous, previous_pts = group.observations[-1]
        if _conflict(previous['reading'], current['reading']) or _conflict(group.anchor, current['reading']):
            self.counters['association_text_conflict'] += 1
            return None
        pw, ph, px, py = _geometry(previous['box'])
        cw, ch, cx, cy = _geometry(current['box'])
        if max(pw / cw, cw / pw, ph / ch, ch / ph) > 2.5:
            self.counters['association_scale_change'] += 1
            return None
        if len(group.observations) == 2:
            older, older_pts = group.observations[0]
            _, _, ox, oy = _geometry(older['box'])
            ratio = (pts_ms - previous_pts) / (previous_pts - older_pts)
            px, py = px + (px - ox) * ratio, py + (py - oy) * ratio
        distance = math.hypot(cx - px, cy - py)
        limit = 1.5 * max(math.hypot(pw, ph), math.hypot(cw, ch), 160.)
        if distance > limit:
            self.counters['association_displacement'] += 1
            return None
        left, right = previous['signature'], current['signature']
        cosine = sum(a * b for a, b in zip(left, right)) if len(left) == len(right) else -1.
        cosine = max(-1., min(1., cosine))
        text_match = _same_text(previous['reading'], current['reading'])
        motion=current.get('motion_support',{}).get(identity,0.)
        if cosine < .75 and not text_match and motion < .30:
            self.counters['association_appearance'] += 1
            return None
        if motion>=.30:
            self.counters['motion_supported_edges']+=1
            return .7*(1-motion)+.3*distance/limit
        return .7 * distance / limit + .3 * (1 - cosine if cosine >= .75 else 0.)

    def _owner(self, proposal, vehicles, group=None, pts_ms=None):
        b = proposal['box']
        area = (b[2] - b[0]) * (b[3] - b[1])
        overlaps = []
        for vehicle in vehicles:
            v = vehicle['box']
            overlap = max(0, min(b[2], v[2]) - max(b[0], v[0])) * max(0, min(b[3], v[3]) - max(b[1], v[1]))
            overlaps.append((overlap / area, vehicle))
        containing = [v for coverage, v in overlaps if coverage >= .9]
        if len(containing) != 1:
            return None, 'ambiguous_owner' if len(containing) > 1 else 'unassigned_owner'
        owner = containing[0]
        if any(coverage >= .2 and v['id'] != owner['id'] for coverage, v in overlaps):
            return None, 'ambiguous_owner'
        if owner['id'] in self.excluded:
            return None, 'excluded_owner'
        if not owner['qualified'] or owner.get('flagged', False):
            return None, 'unqualified_owner'
        return owner['id'], 'unique_owner'

    def route(self, proposals, vehicles, pts_ms):
        if (type(pts_ms) is not int or pts_ms < 0
                or (self.last_pts is not None and pts_ms <= self.last_pts)):
            self.counters['invalid_timestamp'] += 1
            raise ValueError('Forward nonnegative integer video PTS required')
        vehicles = list(vehicles)
        ids = [v['id'] for v in vehicles]
        if len(set(ids)) != len(ids) or any(not _valid_box(v.get('box')) for v in vehicles):
            raise ValueError('Unique vehicle IDs and valid global boxes required')
        self.last_pts = pts_ms
        retained_owners={v['id'] for v in vehicles if v.get('status') in ('visible','predicted','lost_unresolved')}
        retained_owners.update(self.retained_parent_ids)
        for identity, group in list(self.groups.items()):
            if pts_ms - group.observations[-1][1] > self.max_gap_ms:
                if (self.retain_owned and group.owner_votes>=2 and group.candidate in retained_owners
                        and pts_ms-group.observations[-1][1]<=5000):
                    self.counters['retained_through_parent_gap']+=1
                    continue
                del self.groups[identity]
                self.counters['expired_groups'] += 1
        prepared = []
        for proposal in proposals:
            if not _valid_box(proposal.get('box')) or not _finite(proposal.get('confidence', 0.)):
                self.counters['invalid_proposal'] += 1
                continue
            signature, reading = _signature(proposal.get('signature')), _reading(proposal.get('reading', ()))
            if signature is None or reading is None:
                self.counters['invalid_signature' if signature is None else 'invalid_reading'] += 1
                continue
            prepared.append({**proposal, 'signature': signature, 'reading': reading})
        unique = []
        for proposal in sorted(prepared, key=lambda p: -p.get('confidence', 0.)):
            if any(box_iou(proposal['box'], other['box']) >= .5 for other in unique):
                self.counters['duplicate_proposals'] += 1
            else:
                unique.append(proposal)
        edges = []
        for index, proposal in enumerate(unique):
            for identity, group in self.groups.items():
                cost = self._edge(group, proposal, pts_ms, identity)
                if cost is not None:
                    edges.append((cost, index, identity))
        # Competing candidates on either side abstain; global mutual best edges
        # enforce one-to-one assignment without a solver or unsafe fallback edge.
        by_proposal, by_group = {}, {}
        for edge in sorted(edges):
            by_proposal.setdefault(edge[1], []).append(edge)
            by_group.setdefault(edge[2], []).append(edge)
        ambiguous = set()
        for values in by_proposal.values():
            if len(values) > 1 and values[1][0] - values[0][0] <= .10 + 1e-12:
                ambiguous.add(values[0][1])
        for values in by_group.values():
            if len(values) > 1 and values[1][0] - values[0][0] <= .10 + 1e-12:
                ambiguous.update(edge[1] for edge in values if edge[0] - values[0][0] <= .10 + 1e-12)
        matches = {}
        for index, values in by_proposal.items():
            best = values[0]
            if index not in ambiguous and by_group[best[2]][0] == best:
                matches[index] = best[2]
        output = []
        for index, proposal in enumerate(unique):
            identity = matches.get(index)
            if identity is None:
                self._serial += 1
                identity = f'PT{self._serial:04d}'
                self.groups[identity] = _Group([], proposal['reading'])
                self.counters['new_groups'] += 1
                if index in ambiguous:
                    self.counters['ambiguous_association'] += 1
            else:
                self.counters['matched_observations'] += 1
            group = self.groups[identity]
            candidate, status = self._owner(proposal, vehicles, group, pts_ms)
            self.counters[status] += 1
            if candidate is not None:
                if candidate != group.candidate:
                    if group.candidate is not None:
                        self.counters['owner_conflict'] += 1
                    group.candidate, group.owner_votes = candidate, 1
                else:
                    group.owner_votes = min(2, group.owner_votes + 1)
            locked = candidate is not None and candidate == group.candidate and group.owner_votes >= 2
            evidence = {'box': tuple(proposal['box']), 'signature': proposal['signature'],
                        'reading': proposal['reading']}
            group.observations = (group.observations + [(evidence, pts_ms)])[-2:]
            if _strength(proposal['reading']) > _strength(group.anchor):
                group.anchor = proposal['reading']
            output.append({**proposal, 'group': identity, 'position_ms': pts_ms,
                           'owner': candidate if locked else None, 'owner_candidate': candidate,
                           'owner_locked': locked, 'ownership_status': status})
        while len(self.groups) > self.max_active:
            oldest = min(self.groups, key=lambda name: (self.groups[name].observations[-1][1], name))
            del self.groups[oldest]
            self.counters['capacity_evictions'] += 1
        self.counters['plate_observations'] += len(output)
        return output
