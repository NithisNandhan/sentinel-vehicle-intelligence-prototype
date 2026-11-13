"""Bounded RAM replay of actual recorded analysis, never live-camera evidence."""
import json
import re


class ReplayUnavailable(ValueError):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(detail)


class RecordedReplay:
    def __init__(self, *, max_bytes=64*1024*1024, max_samples=300):
        if (type(max_bytes) is not int or type(max_samples) is not int
                or not 1 <= max_bytes <= 64*1024*1024 or not 1 <= max_samples <= 300):
            raise ValueError('Replay budget exceeds its RAM bound')
        self.max_bytes, self.max_samples = max_bytes, max_samples
        self.clear()

    def clear(self):
        self.session_id = self.camera_id = self.duration_ms = None
        self.samples, self.bytes_used = [], 0
        self.reason = None

    def fail(self, reason):
        self.samples, self.bytes_used = [], 0
        self.reason = reason

    def append(self, snapshot, *, session_id, camera_id, duration_ms, frame_builder):
        if self.session_id != session_id:
            self.clear()
            self.session_id, self.camera_id, self.duration_ms = session_id, camera_id, duration_ms
        if self.reason is not None:
            return
        if type(duration_ms) is not int or duration_ms <= 0:
            self.fail('Replay requires a known recording duration')
            return
        position = snapshot.get('video_position_ms')
        if type(position) is not int or not 0 <= position <= duration_ms:
            self.fail('Replay requires valid source timestamps')
            return
        if self.samples:
            previous = self.samples[-1]['video_position_ms']
            if position == previous:
                return
            if position < previous:
                self.fail('Source timestamp reset prevents a faithful normal-speed replay')
                return
        if len(self.samples) >= self.max_samples:
            self.fail('Recording exceeds the replay sample budget; analysis results are unaffected')
            return
        source = dict(snapshot, source_mode='recorded', session_id=session_id)
        try:
            overlay = frame_builder(source, mode='overlays')
            original = frame_builder(source, mode='original')
            if not overlay or not overlay.get('frame') or not original or not original.get('frame'):
                self.fail('A sampled source frame is unavailable for replay')
                return
            if any(not isinstance(view['frame'], str) or len(view['frame']) > 650_000
                   or not re.fullmatch(r'[A-Za-z0-9+/]+={0,2}', view['frame']) for view in (overlay, original)):
                self.fail('A sampled frame exceeds the replay transport limit')
                return
            fields = ('key', 'track_id', 'vehicle_class', 'confidence', 'first_ms', 'last_ms',
                      'observations', 'state', 'reading', 'watchlist')
            vehicles = [dict({key: row.get(key) for key in fields}, thumb=None)
                        for row in overlay.get('vehicles', [])[:200] if isinstance(row, dict)]
            status_fields = tuple(key for key in fields if key != 'reading') + ('masked_plate',)
            seen = [{key: row.get(key) for key in status_fields}
                    for row in snapshot.get('vehicles_seen', [])[:200] if isinstance(row, dict)]
            sample = dict(video_position_ms=position, frames=dict(overlays=overlay['frame'], original=original['frame']),
                          vehicles=vehicles, vehicles_seen=seen,
                          unique_model_accepted_unverified=snapshot.get('unique_model_accepted_unverified', 0))
            payload = json.dumps(sample, allow_nan=False, separators=(',', ':'))
            size = len(payload.encode('utf-8'))
            if self.bytes_used + size > self.max_bytes:
                self.fail('Recording exceeds the replay RAM budget; analysis results are unaffected')
                return
            self.samples.append(json.loads(payload))  # Own copies, not mutable tracker state.
            self.bytes_used += size
        except (ValueError, TypeError, KeyError):
            self.fail('Sampled analysis could not be retained for replay')

    def page(self, session_id, *, completed, offset=0, limit=1, mode='overlays'):
        if session_id != self.session_id:
            raise ReplayUnavailable(409, 'The recognition session changed')
        if not completed:
            raise ReplayUnavailable(409, 'Replay becomes available after complete analysis')
        if self.reason or not self.samples:
            raise ReplayUnavailable(503, self.reason or 'No completed analysis is retained for replay')
        if (type(offset) is not int or not 0 <= offset <= len(self.samples)
                or type(limit) is not int or limit != 1 or mode not in ('overlays', 'original')):
            raise ReplayUnavailable(400, 'Invalid replay page')
        end = min(len(self.samples), offset + limit)
        samples = []
        for stored in self.samples[offset:end]:
            item = {key: value for key, value in stored.items() if key != 'frames'}
            item['frame'] = stored['frames'][mode]
            samples.append(item)
        result = dict(schema_version='1.0', source_mode='recorded', session_id=session_id,
                      camera_id=self.camera_id, video_duration_ms=self.duration_ms, available=True,
                      reason=None, sample_count=len(self.samples), offset=offset,
                      next_offset=end if end < len(self.samples) else None, samples=samples)
        return json.loads(json.dumps(result, allow_nan=False))
