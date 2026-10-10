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

from .privacy import plate_hmac
from .recognition_contract import recognition_state

CLASS_NAMES = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
RANK = {'unreadable': 0, 'reading': 1, 'uncertain': 2, 'model_accepted': 3}


class PassedVehicles:
    def __init__(self, *, limit=200, keep_text=False, hmac_key=None):
        self.rows = OrderedDict()
        self.limit, self.keep_text, self.hmac_key = limit, keep_text, hmac_key
        self.watchlist = None   # None: not known (central unreachable). A set once fetched.

    def update(self, segment, position_ms, tracks, displays, readings):
        """`readings` maps track_id to the current best full string, or None."""
        for track in tracks:
            row = self._row(segment, track.track_id, position_ms)
            row['last_ms'] = position_ms
            row['vehicle_class'] = CLASS_NAMES.get(track.vehicle_class, 'vehicle')
            confidence = getattr(track, 'confidence', None)
            if confidence is not None:
                row['confidence'] = max(row['confidence'] or 0., round(float(confidence), 3))
        for display in displays:
            row = self._row(segment, display['track_id'], position_ms)
            row['last_ms'] = position_ms
            state = recognition_state(display['ocr_status'])
            improved = display['observations'] > row['observations']
            if RANK[state] >= RANK[row['state']]:
                row['state'] = state
                row['masked_plate'] = display['masked_plate'] or row['masked_plate']
                reading = readings.get(display['track_id'])
                if reading:
                    row['_reading'] = reading
            if improved or row['_thumb'] is None:
                row['observations'] = max(row['observations'], display['observations'])
                row['_thumb'] = self._thumbnail(display['crop'])
        while len(self.rows) > self.limit:
            self.rows.popitem(last=False)

    def _row(self, segment, track_id, position_ms):
        key = (segment, str(track_id))
        row = self.rows.get(key)
        if row is None:
            row = self.rows[key] = {'key': f'{segment}-{track_id}', 'track_id': str(track_id), 'vehicle_class': 'vehicle',
                                    'confidence': None, 'first_ms': position_ms, 'last_ms': position_ms,
                                    'observations': 0, 'state': 'unreadable', 'masked_plate': None,
                                    '_reading': None, '_thumb': None}
        return row

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

    def masked(self, limit=50):
        """Newest first, masked text only. A watchlist flag appears only for an accepted reading."""
        rows = []
        for row in list(reversed(self.rows.values()))[:limit]:
            item = self._public(row)
            item['watchlist'] = self._watch(row) if row['state'] == 'model_accepted' else None
            rows.append(item)
        return rows
