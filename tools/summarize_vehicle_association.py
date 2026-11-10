"""Publish tracker comparison counts and selected identity checks, never plate text."""
import argparse
import json
from pathlib import Path


def summarize(general,vehicle):
    first=json.loads((general/'report.json').read_bytes());second=json.loads((vehicle/'report.json').read_bytes())
    checks=json.loads((general/'identity-checks.json').read_bytes());vehicle_checks=json.loads((vehicle/'identity-checks.json').read_bytes())
    for key in ('source_sha256','model_sha256','detections_sha256','decoded_frames'):
        if first[key]!=second[key]:raise ValueError('Different frozen comparison inputs')
    if checks['annotations_sha256']!=vehicle_checks['annotations_sha256']:raise ValueError('Different selected review boxes')
    entries=[('bytetrack_screened',first['strategies']['bytetrack_screened'],checks['strategies']['bytetrack_screened']),
             ('botsort_motion',first['strategies']['botsort_motion'],checks['strategies']['botsort_motion']),
             ('botsort_person_encoder_diagnostic',first['strategies']['botsort_reid'],checks['strategies']['botsort_reid']),
             ('botsort_vehicle_encoder',second['strategies']['botsort_reid'],vehicle_checks['strategies']['botsort_reid'])]
    fields=('native_track_ids','passage_records','qualified_passages','tracker_config','tracker_config_sha256',
            'replay_wall_seconds','median_tracker_ms','p95_tracker_ms','encoded_features','feature_dimension','appearance')
    return dict(date='2026-11-10',scope=checks['scope'],source_sha256=first['source_sha256'],
        frozen_detections_sha256=first['detections_sha256'],detector_sha256=first['model_sha256'],frames=first['decoded_frames'],
        selected_review_annotations_sha256=checks['annotations_sha256'],review_annotations_used_during_inference=False,
        match_iou=checks['match_iou'],unique_match_margin=checks['unique_match_margin'],
        selected_tracker='existing ByteTrack retained for integration',
        reason='No candidate improved the selected identity checks. Lower/higher track counts alone do not measure identity accuracy.',
        limitations=['Only eight assistant-reviewed approximate source-frame rectangles across four vehicle cases; no human-verified MOT ground truth.',
                     'All methods match these eight anchors, preserve two of three repeatedly observed cases and fragment the silver-car case; the fourth vehicle is a single-anchor negative check only. No selected cross-vehicle ID merge was observed.',
                     'The Ultralytics encoder diagnostic is pedestrian-trained (MSMT17 metadata), at its installed default224; it is not a vehicle-trained model comparison.',
                     'The vehicle encoder uses the publisher RGB raw208x208 contract and a VeRi-776 vehicle model. Domain coverage of Indian autorickshaws is not established.',
                     'Observed timings include native decode/crop work on this laptop and are not capacity or real-time claims.'],
        strategies={name:dict(**{k:result.get(k) for k in fields},selected_identity_checks={k:v for k,v in review.items() if k!='cases'})
                    for name,result,review in entries},central_events_sent=0,ocr_loaded=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--general',type=Path,required=True)
    parser.add_argument('--vehicle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();document=summarize(args.general,args.vehicle)
    args.output.write_text(json.dumps(document,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(selected_tracker=document['selected_tracker'],frames=document['frames'],strategies=list(document['strategies']))))
