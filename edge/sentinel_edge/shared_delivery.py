"""Durable masked delivery for the opt-in shared pipeline. No media on disk."""
from collections import OrderedDict
import time
from uuid import NAMESPACE_URL, uuid5

from .api import CentralClient
from .events import sighting
from .outbox import Outbox
from .privacy import plate_hmac

RECORDED_EVENTS = uuid5(NAMESPACE_URL, 'https://sentinel.invalid/recorded-sightings')


class SharedDelivery:
    def __init__(self, config, profile, *, clock=time.monotonic):
        self.config, self.profile, self.clock = config, profile, clock
        self.outbox = Outbox(config.db_path)
        self.client = CentralClient(config.central_api_url, config.edge_api_key, self.outbox)
        self.sent = OrderedDict()
        self.sent_recorded = OrderedDict()

    def start(self, stop):
        return self.client.start_retry_loop(stop)

    def depth(self):
        return self.outbox.depth()

    def accept(self, camera_id, track_id, candidate, pts_ms, *, recorded=None):
        if recorded is not None:
            return self._accept_recorded(camera_id, track_id, candidate, pts_ms, recorded)
        now = self.clock()
        # A short camera+identifier TTL also handles tracker-ID resets. Different
        # cameras remain separate sightings, preserving route reconstruction.
        key = (camera_id, plate_hmac(candidate.plate, self.config.hmac_key))
        for old, stamp in list(self.sent.items()):
            if now-stamp > 60:
                del self.sent[old]
        if key in self.sent:
            return False
        event = sighting(edge_id=self.config.edge_id+'-shared', camera_id=camera_id,
                         plate=candidate.plate, hmac_key=self.config.hmac_key,
                         confidence=candidate.score, source_mode='live', stream_pts_ms=pts_ms,
                         track_id=track_id, model_version='tracked-'+self.profile+'+awiros-ctc+ecc-reconstruction-proposals;unverified')
        self.outbox.put('/v1/sightings',event)
        self.sent[key] = now
        while len(self.sent) > 4096:
            self.sent.popitem(last=False)
        return True

    def _accept_recorded(self, camera_id, track_id, candidate, pts_ms, run):
        """One sighting per identifier per 60 s of *video* time within a run.

        A file can be processed faster or slower than real time, so the
        deduplication window follows the video clock, not the wall clock.
        """
        identifier = plate_hmac(candidate.plate, self.config.hmac_key)
        key = (run.session_id, camera_id, identifier)
        last = self.sent_recorded.get(key)
        if last is not None and 0 <= pts_ms-last <= 60_000:
            return False
        observed, basis = run.observed_at(pts_ms)
        # The ID is a function of the run and video position: resending this
        # event is idempotent centrally, while a restart or seek (a new
        # session) produces distinct, traceable sightings.
        event_id = str(uuid5(RECORDED_EVENTS, f'{run.session_id}|{camera_id}|{identifier}|{pts_ms}'))
        event = sighting(edge_id=self.config.edge_id+'-shared', camera_id=camera_id,
                         plate=candidate.plate, hmac_key=self.config.hmac_key,
                         confidence=candidate.score, source_mode='recorded', stream_pts_ms=pts_ms,
                         track_id=track_id, observed_at=observed.isoformat().replace('+00:00', 'Z'),
                         model_version='recorded;tracked-'+self.profile+'+awiros-ctc+ecc-reconstruction-proposals;unverified',
                         session_id=run.session_id, time_basis=basis, event_id=event_id)
        self.outbox.put('/v1/sightings', event)
        self.sent_recorded[key] = pts_ms
        self.sent_recorded.move_to_end(key)
        while len(self.sent_recorded) > 4096:
            self.sent_recorded.popitem(last=False)
        return True
