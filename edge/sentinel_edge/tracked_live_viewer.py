"""In-memory overlays for the opt-in, per-vehicle tracked OCR experiment."""
from collections import deque
import statistics
import time

from .crop_collector import CropCollector as LatestFrames
from .live_detection_viewer import encoded
from .tracked_ocr import TrackedOCR


def observe_tracked(config,camera,model,detector,vehicle,state,stop, *, debug=False, delivery=None,
                    detector_profile=None, plate_recovery=False):
    import cv2
    from .live_detection_viewer import viewer_frames
    callback = (lambda track, candidate, pts: delivery.accept(camera.camera_id, track, candidate, pts)) if delivery else None
    from .plate_search import TargetedPlateSearch
    pipeline = TrackedOCR(model,TargetedPlateSearch(detector),threshold=config.ocr_min_confidence, debug=debug, on_accepted=callback,
                          reconstruction=True)
    sequence = 0
    last_pts = sampled_pts = first = last_crop = None
    timings = deque(maxlen=256)
    collector_options = dict(adaptive_sampling=True,full_image_fallback=True) if plate_recovery else {}
    with LatestFrames(lambda token:viewer_frames(config,camera,token),stop,vehicle,pipeline.detector,
                      **collector_options) as tap:
        if not hasattr(tap,'vehicle'):vehicle.reset()
        while not stop.is_set():
            item = tap.take(sequence)
            if item is None:
                if tap.finished:break
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
            view = frame.image.copy()
            for track in tracks:
                cv2.rectangle(view,track.box[:2],track.box[2:],(90,220,90),2)
                cv2.putText(view,'vehicle '+track.track_id,track.box[:2],cv2.FONT_HERSHEY_SIMPLEX,.5,(90,220,90),1)
            for display in displays:
                box = display['box']
                cv2.rectangle(view,box[:2],box[2:],(0,210,255),2)
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
                details['full_image_detector_calls'] = getattr(tap,'full_image_calls',0)
                details['full_image_recovered_crops'] = getattr(tap,'full_image_crops',0)
                details['adaptive_sampling_enabled'] = plate_recovery
                details['original_frame'] = encoded(frame.image,'.jpg')
            snapshot = dict(pipeline.summary(),camera_id=camera.camera_id,status='live',published_at=getattr(frame,'captured_at',now) or now,
                pts_ms=frame.pts_ms,frame=encoded(view,'.jpg'),crop=crop_display,
                vehicle_boxes=len(tracks),decoded_frames=tap.decoded,
                plate_in_frame=bool(displays),resolution=[frame.image.shape[1],frame.image.shape[0]],
                observed_seconds=round(now-first,2),median_processing_ms=round(statistics.median(timings)),
                processing_ms=round((time.monotonic()-started)*1000+collection_ms),ocr_status='tracked multi-frame experiment',
                model_profile=detector_profile or 'general')
            snapshot['crop_first_search'] = True
            if debug:
                snapshot['debug'] = details
            if delivery:
                snapshot['delivery'] = {'enabled':True,'outbox_depth':delivery.depth()}
            state.publish(snapshot,stop)
        if tap.failed or not pipeline.metrics['analysed_frames'] or not stop.is_set():
            error = RuntimeError('Experimental viewer stream unavailable')
            error.failure_code = getattr(tap,'failure_code','NoUsableFrames')
            raise error
