"""Durable masked delivery for the opt-in shared pipeline. No media on disk."""
from collections import OrderedDict
import time

from .api import CentralClient
from .events import sighting
from .outbox import Outbox
from .privacy import plate_hmac


class SharedDelivery:
    def __init__(self, config, profile, *, clock=time.monotonic):
        self.config, self.profile, self.clock = config, profile, clock
        self.outbox = Outbox(config.db_path)
        self.client = CentralClient(config.central_api_url, config.edge_api_key, self.outbox)
        self.sent = OrderedDict()

    def start(self, stop):
        return self.client.start_retry_loop(stop)

    def depth(self):
        return self.outbox.depth()

    def accept(self, camera_id, track_id, candidate, pts_ms):
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
