"""Running list of the vehicles seen in one analysis run, for the side panels.

Every tracked vehicle gets a row the moment it appears and the row is updated
as plate evidence arrives, so the interfaces can show vehicles "passing". It is
bounded, lives in memory only and is discarded with the run.

Two views: the protected engineering console gets the unverified best reading
and a small native crop; the operator view gets masked text only. A watchlist
flag is computed locally from keyed identifiers, is always labelled
unverified, and never replaces the central alert, which still requires an
accepted reading.
"""
from collections import OrderedDict
import math

from .privacy import plate_hmac
from .recognition_contract import recognition_state

CLASS_NAMES = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
RANK = {'unreadable': 0, 'reading': 1, 'uncertain': 2, 'model_accepted': 3}


class PassedVehicles:
    def __init__(self, *, limit=200, keep_text=False, hmac_key=None):
        self.rows = OrderedDict()
        self.limit, self.keep_text, self.hmac_key = limit, keep_text, hmac_key
        self.watchlist = None   # None: not known (central unreachable). A set once fetched.

    def update(self, segment, position_ms, tracks, displays, readings, *, reading_details=None, passage_ids=None):
        """Retain the strongest observed fusion window, not the last OCR string.

        `reading_details` contains the already-fused PlateCandidate objects.
        Its scores are ranking signals, never correctness probabilities.
        This display-only choice does not change automatic acceptance/delivery.
        """
        identities = dict(passage_ids or {})
        for display in displays:
            if display.get('passage_id'):
                identities[display['track_id']] = display['passage_id']
        for track in tracks:
            row = self._row(segment, track.track_id, position_ms, identities.get(track.track_id))
            row['last_ms'] = position_ms
            row['vehicle_class'] = CLASS_NAMES.get(track.vehicle_class, 'vehicle')
            confidence = getattr(track, 'confidence', None)
            if confidence is not None:
                row['confidence'] = max(row['confidence'] or 0., round(float(confidence), 3))
        for display in displays:
            row = self._row(segment, display['track_id'], position_ms, identities.get(display['track_id']))
            row['last_ms'] = position_ms
            state = recognition_state(display['ocr_status'])
            reading = readings.get(display['track_id'])
            detail = (reading_details or {}).get(display['track_id'])
            strength = self._strength(state, display['observations'], detail)
            chosen = bool(reading) and (row['_reading'] is None or strength > row['_reading_strength'])
            if chosen:
                row['state'] = state
                from .privacy import mask_plate
                row['masked_plate'] = mask_plate(reading)
                row['_reading'] = reading
                row['_reading_strength'] = strength
                row['_thumb'] = self._thumbnail(display['crop'])
            elif row['_reading'] is None and RANK[state] >= RANK[row['state']]:
                row['state'] = state
            # Count evidence progress even when a later weak window is rejected.
            # Never replace the selected reading's image with a different guess.
            row['observations'] = max(row['observations'], display['observations'])
            if row['_thumb'] is None and (not row['_reading'] or reading == row['_reading']):
                row['_thumb'] = self._thumbnail(display['crop'])
        while len(self.rows) > self.limit:
            self.rows.popitem(last=False)

    def _row(self, segment, track_id, position_ms, passage_id=None):
        # Tracker IDs can switch owners within a scene; conversely one passage
        # can be reassociated to a new tracker ID. Only the passage owns OCR.
        key = (segment, 'passage', passage_id) if passage_id else (segment, 'track', str(track_id))
        row = self.rows.get(key)
        if row is None:
            row = self.rows[key] = {'key': f'{segment}-p-{passage_id}' if passage_id else f'{segment}-{track_id}',
                                    'track_id': str(track_id), 'passage_id': passage_id, 'vehicle_class': 'vehicle',
                                    'confidence': None, 'first_ms': position_ms, 'last_ms': position_ms,
                                    'observations': 0, 'state': 'unreadable', 'masked_plate': None,
                                    '_reading': None, '_thumb': None, '_reading_strength': ()}
        row['track_id'] = str(track_id)
        return row

    @staticmethod
    def _strength(state, observations, detail):
        def score(name):
            value = getattr(detail, name, None)
            return float(value) if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 else 0.
        # A multi-frame window outranks an isolated high-score suggestion.
        # After three originals, prefer the least ambiguous character, then
        # the mean fused score; extra observations break otherwise equal ties.
        count = observations if type(observations) is int and observations >= 0 else 0
        return (RANK[state], min(count, 3), score('minimum_character_score'), score('score'), count)

    @staticmethod
    def _thumbnail(crop):
        import cv2
        from .live_detection_viewer import encoded
        if crop is None or not crop.size:
            return None
        if crop.shape[1] > 220:  # Shrink only; enlarging would add nothing.
            crop = cv2.resize(crop, (220, max(1, round(crop.shape[0]*220/crop.shape[1]))), interpolation=cv2.INTER_AREA)
        return encoded(crop, '.png')

    def _watch(self, row):
        if self.watchlist is None or not self.hmac_key or not row['_reading']:
            return None
        try:
            return 'possible_match' if plate_hmac(row['_reading'], self.hmac_key) in self.watchlist else 'no_match'
        except ValueError:
            return None

    def _public(self, row):
        return {key: value for key, value in row.items() if not key.startswith('_')}

    def engineering(self):
        """Newest first, with the unverified reading and crop. Protected view only."""
        rows = []
        for row in reversed(self.rows.values()):
            item = self._public(row)
            item.update(reading=row['_reading'] if self.keep_text else None, thumb=row['_thumb'], watchlist=self._watch(row))
            rows.append(item)
        return rows

    def masked(self, limit=200):
        """Newest first, masked text only. A watchlist flag appears only for an accepted reading."""
        rows = []
        for row in list(reversed(self.rows.values()))[:limit]:
            item = self._public(row)
            item['watchlist'] = self._watch(row) if row['state'] == 'model_accepted' else None
            rows.append(item)
        return rows
