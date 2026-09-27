"""In-memory overlays for the opt-in, per-vehicle tracked OCR experiment."""
from collections import deque
import statistics
import time

from .detector_ocr_shadow import LatestFrames
from .live_detection_viewer import encoded
from .tracked_ocr import TrackedOCR


def observe_tracked(config,camera,model,detector,vehicle,state,stop, *, debug=False, delivery=None, detector_profile=None):
    import cv2
    from .live_detection_viewer import viewer_frames
    callback = (lambda track, candidate, pts: delivery.accept(camera.camera_id, track, candidate, pts)) if delivery else None
    pipeline = TrackedOCR(model,detector,threshold=config.ocr_min_confidence, debug=debug, on_accepted=callback,
                          reconstruction=True)
    vehicle.reset()
    sequence = 0
    last_pts = sampled_pts = first = last_crop = None
    timings = deque(maxlen=256)
    with LatestFrames(lambda token:viewer_frames(config,camera,token),stop) as tap:
        while not stop.is_set():
            item = tap.take(sequence)
            if item is None:
                if tap.finished:break
                continue
            sequence,frame = item
            if first is None:first = time.monotonic()
            if frame.discontinuity or (last_pts is not None and frame.pts_ms is not None and
                                       (frame.pts_ms < last_pts or frame.pts_ms-last_pts > 2500)):
                pipeline.reset(); vehicle.reset()
                sampled_pts = last_crop = None
            if frame.pts_ms is None:continue
            last_pts = frame.pts_ms
            if sampled_pts is not None and frame.pts_ms-sampled_pts < 500:continue
            sampled_pts = frame.pts_ms
            started = time.monotonic()
            tracks = vehicle.track(frame.image)
            vehicle_ms = (time.monotonic()-started)*1000
            displays = pipeline.process(frame.image,tracks,frame.pts_ms)
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
            timings.append((now-started)*1000)
            details = pipeline.debug_snapshot() if debug else {}
            if debug:
                details['stage_ms']['vehicle'] = round(vehicle_ms,2)
                details['stage_ms'] = {key:round(value,2) for key,value in details['stage_ms'].items()}
                details['overwritten_frames'] = getattr(tap,'overwritten',0)
                details['original_frame'] = encoded(frame.image,'.jpg')
            snapshot = dict(pipeline.summary(),camera_id=camera.camera_id,status='live',published_at=now,
                pts_ms=frame.pts_ms,frame=encoded(view,'.jpg'),crop=crop_display,
                vehicle_boxes=len(tracks),decoded_frames=tap.decoded,
                plate_in_frame=bool(displays),resolution=[frame.image.shape[1],frame.image.shape[0]],
                observed_seconds=round(now-first,2),median_processing_ms=round(statistics.median(timings)),
                processing_ms=round((time.monotonic()-started)*1000),ocr_status='tracked multi-frame experiment',
                model_profile=detector_profile or 'general')
            if debug:
                snapshot['debug'] = details
            if delivery:
                snapshot['delivery'] = {'enabled':True,'outbox_depth':delivery.depth()}
            state.publish(snapshot,stop)
        if tap.failed or not pipeline.metrics['analysed_frames'] or not stop.is_set():
            raise RuntimeError('Experimental viewer stream unavailable')
