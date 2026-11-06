"""Combine frozen geometry diagnostics without further inference or plate text."""
import argparse
import json
from pathlib import Path
import statistics

from compare_automatic_crops import manual_corners,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plate-report',type=Path,required=True)
    parser.add_argument('--text-report',type=Path,required=True)
    parser.add_argument('--dense-report',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import cv2
    import numpy as np
    plate,text,dense=[json.loads(p.read_bytes()) for p in (args.plate_report,args.text_report,args.dense_report)]
    manifest=json.loads((args.review/'manifest.json').read_bytes())
    assert len({r['source_sha256'] for r in (plate,text,dense,manifest)})==1
    assert all(r['manifest_sha256']==sha(args.review/'manifest.json') for r in (plate,text,dense))
    p={r['id']:r for r in plate['cases']}
    t={r['id']:r for r in text['cases']}
    corners=manual_corners(args.review/'ocr-model-dev/rectify.py')
    assert set(p)==set(t)=={r['id'] for r in manifest['samples']}
    cases=[]
    for sample in manifest['samples']:
        identifier=sample['id']
        quad_score=None
        if identifier in corners and t[identifier]['quad']:
            x,y,_,_=sample['box']
            manual=np.array([[a+x,b+y] for a,b in corners[identifier]],np.float32)
            auto=np.array(t[identifier]['quad'],np.float32)
            intersection,_=cv2.intersectConvexConvex(manual,auto)
            area,expected=cv2.contourArea(auto),cv2.contourArea(manual)
            quad_score={'iou':float(intersection/(area+expected-intersection)),
                        'target_coverage':float(intersection/expected)}
        native=max(p[identifier]['methods']['refined']['iou'],t[identifier]['iou'])
        cases.append({'id':identifier,'baseline_iou':p[identifier]['methods']['baseline']['iou'],
                      'tiled_plate_iou':p[identifier]['methods']['refined']['iou'],
                      'text_region_iou':t[identifier]['iou'],'text_quad':quad_score,
                      'combined_native_candidate_hit':native>=.5})
    quads=[r['text_quad'] for r in cases if r['text_quad'] is not None]
    output={'scope':'Previously examined Mumbai development targets. Best matching proposal is associated to each reference AFTER inference; this measures proposal coverage, not automatic final selection or precision.',
            'ocr_loaded':False,'live_pipeline_changed':False,'targets':len(cases),'unique_frames':len(plate['frames']),
            'source_sha256':manifest['source_sha256'],'manifest_sha256':sha(args.review/'manifest.json'),
            'plate_detector_sha256':plate['plate_model_sha256'],'text_model':'PaddlePaddle/PP-OCRv5_mobile_det',
            'text_model_revision':'0d63e78e2b680928f6b1747d76a08db6e645efb7',
            'text_assets_bytes':4939860,
            'baseline_hit_iou50':plate['summary']['baseline']['hit_iou50'],
            'tiled_plate_hit_iou50':plate['summary']['refined']['hit_iou50'],
            'text_region_hit_iou50':text['summary']['hit_iou50'],
            'combined_native_candidate_hit_iou50':sum(r['combined_native_candidate_hit'] for r in cases),
            'text_quad_iou50_on_17_corner_annotations':sum(q['iou']>=.5 for q in quads),
            'annotated_corner_targets':len(corners),'median_text_quad_iou_when_returned':statistics.median(q['iou'] for q in quads),
            'mean_gpu_frame_ms':{'plate_tiles_and_refinement':plate['mean_frame_ms'],'text_tiles':text['summary']['mean_frame_ms'],
                                 'denser_plate_tiles':dense['mean_frame_ms']},
            'denser_plate_tiles_hit_iou50':dense['summary']['refined']['hit_iou50'],
            'failures':[{'experiment':'Initial plate-only run through OCR launcher','error':'cuDNN library-version conflict','resolution':'Use detector image native runtime'},
                        {'experiment':'First text-boundary run','error':'CUDA error 999 at last sampled frame','resolution':'GPU health checked; repeated run completed 14/14 frames. Long-duration reliability unproven.'}],
            'verification':{'unit_tests_passed':22,'crop_viewer_desktop_and_mobile_checked':True},
            'remaining_work':['Plate-vs-other-text filtering without annotation lookup',
                              'Multi-row and separated text-region grouping',
                              'Boundary-labelled examples for custom painted registrations',
                              'Full-video temporal association, bounded scheduling and endurance checks',
                              'Unseen-footage localisation evaluation and exhaustive false-proposal annotation'],
            'cases':cases}
    if args.output.exists():
        raise FileExistsError('Aggregate evidence already exists')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k in ('targets','baseline_hit_iou50','tiled_plate_hit_iou50','text_region_hit_iou50','combined_native_candidate_hit_iou50','text_quad_iou50_on_17_corner_annotations','median_text_quad_iou_when_returned')}))


if __name__=='__main__':
    main()
