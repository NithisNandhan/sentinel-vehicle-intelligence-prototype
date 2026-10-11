"""In-memory overlays for the opt-in, per-vehicle tracked OCR experiment."""
from collections import deque
import statistics
import time

from .crop_collector import CropCollector as LatestFrames
from .live_detection_viewer import encoded
from .tracked_ocr import TrackedOCR


def observe_tracked(config,camera,model,detector,vehicle,state,stop, *, debug=False, delivery=None,
                    detector_profile=None, plate_recovery=False, recorded=None, on_sample=None):
    """Analyse one source until it stops.

    `recorded` is a RecordedRun for a permitted local file. A recording waits
    for the analysis instead of dropping frames, is labelled `recorded` in
    every snapshot and event, and ending normally at EOF is not an error.
    `on_sample(frame, tracks, displays, pipeline)` lets the evaluator observe
    each analysed sample without changing what is analysed or accepted.
    """
    import cv2
    from .live_detection_viewer import viewer_frames
    callback = None
    if delivery:
        callback = ((lambda track, candidate, pts: delivery.accept(camera.camera_id, track, candidate, pts, recorded=recorded))
                    if recorded else
                    (lambda track, candidate, pts: delivery.accept(camera.camera_id, track, candidate, pts)))
    from .plate_search import TargetedPlateSearch
    pipeline = TrackedOCR(model,TargetedPlateSearch(detector),threshold=config.ocr_min_confidence, debug=debug, on_accepted=callback,
                          reconstruction=True)
    # Running list of vehicles seen in this run, for the side panels.
    from .passed_vehicles import PassedVehicles
    seen = PassedVehicles(keep_text=debug, hmac_key=getattr(config,'hmac_key',None))
    client = getattr(delivery,'client',None)
    if client is not None:
        def fetch_watchlist():
            # Keyed identifiers only. If central is unreachable the flag stays unknown.
            try:seen.watchlist = {entry['plate_hmac'] for entry in client.watchlist_snapshot()['entries']}
            except Exception:pass
        import threading
        threading.Thread(target=fetch_watchlist,daemon=True,name='watchlist-snapshot').start()
    sequence = 0
    last_pts = sampled_pts = first = last_crop = None
    timings = deque(maxlen=256)
    collector_options = dict(adaptive_sampling=True,full_image_fallback=True) if plate_recovery else {}
    if recorded:
        collector_options['block'] = True
    source = recorded.frames if recorded else (lambda token:viewer_frames(config,camera,token))
    with LatestFrames(source,stop,vehicle,pipeline.detector,
                      **collector_options) as tap:
        if not hasattr(tap,'vehicle'):vehicle.reset()
        while not stop.is_set():
            # Pause holds analysis itself, not only decoding: frames already
            # queued wait here, so the position stops at the sample in hand.
            if recorded and recorded.paused is not None and recorded.paused.is_set():
                stop.wait(.1)
                continue
            item = tap.take(sequence)
            if item is None:
                # The decoder sets `finished` after its last append, so an
                # empty queue here means every sample has been taken.
                if tap.finished and not getattr(tap,'queue',None):break
                continue
            sequence,frame = item
            if first is None:first = time.monotonic()
            if frame.discontinuity or (last_pts is not None and frame.pts_ms is not None and
                                       (frame.pts_ms < last_pts or frame.pts_ms-last_pts > 2500)):
                pipeline.reset()
                if not hasattr(frame,'plate_cache'):vehicle.reset()
                sampled_pts = last_crop = None
            if frame.pts_ms is None:continue
            last_pts = frame.pts_ms
            if sampled_pts is not None and frame.pts_ms-sampled_pts < (250 if plate_recovery else 500):continue
            sampled_pts = frame.pts_ms
            started = time.monotonic()
            tracks = frame.tracks if hasattr(frame,'tracks') else vehicle.track(frame.image)
            vehicle_ms = frame.vehicle_ms if hasattr(frame,'vehicle_ms') else (time.monotonic()-started)*1000
            displays = pipeline.process(frame.image,tracks,frame.pts_ms,plate_cache=getattr(frame,'plate_cache',None))
            if on_sample is not None:on_sample(frame,tracks,displays,pipeline)
            view = frame.image.copy()
            from .overlay import draw_plate_callout, draw_vehicle, scale_for
            scale = scale_for(view)
            boxes = {track.track_id:track.box for track in tracks}
            plain,shown = {},{}
            for display in displays:
                history = pipeline.histories.get(display['track_id'])
                ranked = history.reader.candidates() if history is not None and history.reader.observation_count else []
                plain[display['track_id']] = ranked[0].plate if ranked else None
                # The protected engineering view may show the full unverified
                # hypothesis; any other viewer only ever gets the masked form.
                shown[display['track_id']] = plain[display['track_id']] if debug else display['masked_plate']
            seen.update(pipeline.metrics['scene_resets'],
                        frame.pts_ms-(recorded.recording.start_pts_ms if recorded else 0),tracks,displays,plain)
            for track in tracks:
                draw_vehicle(view,track,scale,shown.get(track.track_id))
            for display in displays:
                draw_plate_callout(view,boxes.get(display['track_id'],display['box']),display['box'],display['crop'],
                                   shown[display['track_id']],display['ocr_status'],display['observations'],scale)
            if displays:
                selected = max(displays,key=lambda d:d['observations'])
                crop = selected['crop']
                last_crop = {'image':encoded(crop,'.png'),'width':crop.shape[1],'height':crop.shape[0],
                    'seen_at':time.monotonic(),'observations':selected['observations'],
                    'ocr_status':selected['ocr_status'],'masked_plate':selected['masked_plate']}
            now = time.monotonic()
            crop_display = dict(last_crop) if last_crop and now-last_crop['seen_at'] <= 10 else None
            if crop_display:crop_display['age_seconds'] = round(now-crop_display.pop('seen_at'),1)
            collection_ms = getattr(frame,'collection_ms',0.)
            timings.append((now-started)*1000+collection_ms)
            details = pipeline.debug_snapshot() if debug else {}
            if debug:
                details['stage_ms']['vehicle'] = round(vehicle_ms,2)
                details['stage_ms'] = {key:round(value,2) for key,value in details['stage_ms'].items()}
                details['overwritten_frames'] = (getattr(tap,'input_dropped',0)+getattr(tap,'dropped_samples',0))
                details['sampled_frames'] = getattr(tap,'sampled',0)
                details['dropped_sampled_frames'] = getattr(tap,'dropped_samples',0)
                details['decode_queue_peak_bytes'] = getattr(tap,'peak_bytes',0)
                details['collector_input_drops'] = getattr(tap,'input_dropped',0)
                details['collected_native_crops'] = getattr(tap,'collected_crops',0)
                details['plate_roi_searches'] = getattr(tap,'roi_searches',0)
                details['plate_roi_unsearched_tracks'] = getattr(tap,'roi_unsearched_tracks',0)
                details['eligible_vehicle_tracks'] = pipeline.metrics['eligible_vehicle_tracks']
                details['full_image_detector_calls'] = getattr(tap,'full_image_calls',0)
                details['full_image_recovered_crops'] = getattr(tap,'full_image_crops',0)
                details['adaptive_sampling_enabled'] = plate_recovery
                details['original_frame'] = encoded(frame.image,'.jpg')
            # A live frame's age starts when it was captured. A recording has
            # no capture clock; its age is the time since it was analysed.
            published_at = now if recorded else (getattr(frame,'captured_at',now) or now)
            snapshot = dict(pipeline.summary(),camera_id=camera.camera_id,
                status='processing' if recorded else 'live',published_at=published_at,
                pts_ms=frame.pts_ms,frame=encoded(view,'.jpg'),crop=crop_display,
                vehicle_boxes=len(tracks),decoded_frames=tap.decoded,
                plate_in_frame=bool(displays),resolution=[frame.image.shape[1],frame.image.shape[0]],
                observed_seconds=round(now-first,2),median_processing_ms=round(statistics.median(timings)),
                processing_ms=round((time.monotonic()-started)*1000+collection_ms),ocr_status='tracked multi-frame experiment',
                model_profile=detector_profile or 'general')
            snapshot['crop_first_search'] = True
            snapshot['vehicles_seen'] = seen.masked()
            if debug:details['passed'] = seen.engineering()
            if recorded:
                snapshot['video_position_ms'] = frame.pts_ms-recorded.recording.start_pts_ms
                snapshot['pacing'] = 'paced' if recorded.paced else 'unpaced'
            if debug:
                snapshot['debug'] = details
            if delivery:
                snapshot['delivery'] = {'enabled':True,'outbox_depth':delivery.depth()}
            state.publish(snapshot,stop)
        if recorded:
            # End of file or an operator stop. Only a decoder failure is an error.
            if tap.failed:
                error = RuntimeError('Recording could not be analysed')
                error.failure_code = getattr(tap,'failure_code','RecordingFailure')
                raise error
            return pipeline.summary()
        if tap.failed or not pipeline.metrics['analysed_frames'] or not stop.is_set():
            error = RuntimeError('Experimental viewer stream unavailable')
            error.failure_code = getattr(tap,'failure_code','NoUsableFrames')
            raise error
