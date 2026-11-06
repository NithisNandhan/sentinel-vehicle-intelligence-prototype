"""Durable masked delivery for the opt-in shared pipeline. No media on disk."""
from collections import OrderedDict
import time
from uuid import NAMESPACE_URL, uuid5

from .api import CentralClient
from .events import recognition_result, sighting, utc_now
from .outbox import Outbox
from .plate_options import PlateOptions
from .privacy import plate_hmac

RECORDED_EVENTS = uuid5(NAMESPACE_URL, 'https://sentinel.invalid/recorded-sightings')
RESULT_EVENTS = uuid5(NAMESPACE_URL, 'https://sentinel.invalid/recognition-results')
# Fragments of one vehicle passage that the tracker split: suppressed, counted.
FRAGMENT_WINDOW_MS = 1500
PARTIAL_DUPLICATE_WINDOW_MS = 10_000


class SharedDelivery:
    def __init__(self, config, profile, *, clock=time.monotonic, options=None, evidence=None, snapshot=None):
        self.config, self.profile, self.clock = config, profile, clock
        self.options = options or PlateOptions()
        self.outbox = Outbox(config.db_path)
        self.client = CentralClient(config.central_api_url, config.edge_api_key, self.outbox,
                                    edge_id=config.edge_id+'-shared')
        self.sent = OrderedDict()
        self.sent_recorded = OrderedDict()
        self.sighting_events = OrderedDict()   # (camera_id, track_id) -> sighting event_id
        self.recent_results = []               # Bounded list for passage-fragment deduplication.
        self.evidence = evidence               # EvidenceStore, or None: no review evidence kept.
        self.snapshot = snapshot               # WatchlistSnapshot for partial leads, or None.
        from collections import Counter
        self.result_metrics = Counter()

    def start(self, stop):
        if self.snapshot is not None:
            import threading
            def refresh():
                while not stop.is_set():
                    self.snapshot.refresh()
                    stop.wait(min(300, max(30, self.snapshot.max_age // 3)))
            threading.Thread(target=refresh, daemon=True, name='watchlist-snapshot-refresh').start()
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
        self._remember_sighting(camera_id, track_id, event['event_id'])
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
        self._remember_sighting(camera_id, track_id, event_id)
        self.sent_recorded[key] = pts_ms
        self.sent_recorded.move_to_end(key)
        while len(self.sent_recorded) > 4096:
            self.sent_recorded.popitem(last=False)
        return True

    def _remember_sighting(self, camera_id, track_id, event_id):
        self.sighting_events[(camera_id, str(track_id))] = event_id
        while len(self.sighting_events) > 4096:
            self.sighting_events.popitem(last=False)

    # ----- Result states: one event per vehicle passage -------------------------

    def _duplicate(self, camera_id, outcome, when_ms):
        """True when this passage is a fragment of one already reported."""
        from .vision import track_consistent
        result = outcome.result
        self.recent_results = [item for item in self.recent_results
                               if 0 <= when_ms - item['when'] <= PARTIAL_DUPLICATE_WINDOW_MS][-256:]
        for item in self.recent_results:
            if item['camera'] != camera_id:
                continue
            text = result.plate or result.pattern
            if text and item['text'] and len(text) == len(item['text']) and all(
                    a == b or '?' in (a, b) for a, b in zip(text, item['text'])) and result.state != 'confirmed':
                return 'compatible_with_recent_reading'
            if (result.state == 'unreadable' and outcome.first_box is not None and item['box'] is not None
                    and 0 <= outcome.first_pts - item['last_pts'] <= FRAGMENT_WINDOW_MS
                    and track_consistent(item['box'], outcome.first_box)):
                return 'tracker_fragment'
        return None

    def finalized(self, camera_id, outcome, *, recorded=None, secondary=None):
        """Queue the result event for one finished passage. Returns the event or None."""
        result = outcome.result
        source_mode = 'recorded' if recorded is not None else 'live'
        when_ms = outcome.last_pts if outcome.last_pts is not None else int(self.clock() * 1000)
        state = 'confirmed' if outcome.confirmed_plate else result.state
        duplicate = self._duplicate(camera_id, outcome, when_ms) if state != 'confirmed' else None
        self.recent_results.append({'camera': camera_id, 'when': when_ms, 'last_pts': outcome.last_pts,
                                    'box': outcome.last_box, 'text': outcome.confirmed_plate or result.plate or result.pattern})
        if duplicate:
            self.result_metrics['suppressed_'+duplicate] += 1
            return None
        evidence_ref = None
        if self.evidence is not None and state == 'partial' and (source_mode == 'recorded' or self.options_live_evidence):
            from .evidence_store import review_record
            evidence_ref = self.evidence.put(review_record(outcome, camera_id=camera_id, source_mode=source_mode,
                                                           secondary=secondary))
        lead = None
        if state == 'partial' and self.options.partial_watchlist and self.snapshot is not None:
            from .watchlist_snapshot import partial_leads
            lead = partial_leads(result.pattern, self.config.hmac_key, self.snapshot.entries(),
                                 length_certain=result.length_certain)
            self.result_metrics['partial_lead_'+('skipped_'+lead['skipped'] if 'skipped' in lead else
                                                ('found' if lead['candidate_hmacs'] else 'none'))] += 1
            if 'skipped' in lead or not lead['candidate_hmacs']:
                lead = None
        session = recorded.session_id if recorded is not None else None
        event_id = str(uuid5(RESULT_EVENTS, f'{session or self.config.edge_id}|{camera_id}|{outcome.passage_id}'))
        common = dict(edge_id=self.config.edge_id+'-shared', camera_id=camera_id, passage_id=outcome.passage_id,
                      source_mode=source_mode, event_id=event_id, stream_pts_ms=outcome.first_pts,
                      end_pts_ms=outcome.last_pts, track_ids=outcome.track_ids, observations=result.observations,
                      quality=outcome.indicators, layout=result.layout, profile=self.options.profile,
                      policy_version=result.policy_version, review_required=result.review_required or outcome.revisited,
                      model_version=('recorded;' if recorded is not None else '')
                                    + 'tracked-'+self.profile+'+awiros-ctc;result-states')
        if recorded is not None:
            first, basis = recorded.observed_at(outcome.first_pts)
            last, _ = recorded.observed_at(outcome.last_pts)
            common.update(observed_at=first.isoformat().replace('+00:00', 'Z'),
                          last_observed_at=last.isoformat().replace('+00:00', 'Z'),
                          session_id=recorded.session_id, time_basis=basis)
        else:
            stamp = utc_now()
            common.update(observed_at=stamp, last_observed_at=stamp)
        if state == 'confirmed':
            sighting_id = next((self.sighting_events.get((camera_id, str(t))) for t in reversed(outcome.track_ids)
                                if (camera_id, str(t)) in self.sighting_events), None)
            event = recognition_result(result_state='confirmed', plate=outcome.confirmed_plate,
                                       hmac_key=self.config.hmac_key, format_status=result.format_status,
                                       sighting_event_id=sighting_id, **common)
        elif state == 'partial':
            event = recognition_result(result_state='partial', masked_pattern=result.masked(),
                                       plate_length=result.length, unknown_positions=result.unknown_positions,
                                       reason=result.reason, length_certain=result.length_certain,
                                       format_status=result.format_status, evidence_ref=evidence_ref,
                                       watchlist_lead=lead, **common)
        else:
            event = recognition_result(result_state='unreadable', reason=result.reason, **common)
        self.outbox.put('/v1/recognition-results', event)
        self.result_metrics['delivered_'+state] += 1
        return event

    @property
    def options_live_evidence(self):
        import os
        # Live-camera imagery stays inside the edge process unless explicitly allowed.
        return os.getenv('SENTINEL_REVIEW_EVIDENCE_LIVE', '').strip().lower() == 'true'
