"""Replay real distinct frames to separate single-frame and fusion failures."""
import argparse
import hashlib
import json
from pathlib import Path

from sentinel_edge.awiros_experiment import AwirosExperiment, file_hash
from sentinel_edge.ctc_fusion import CTCPlateReader, greedy_text
from sentinel_edge.ocr_crop_experiment import contrast_view, padded_view
from sentinel_edge.plate_quality import crop_quality


def run(args):
    import cv2
    document = json.loads(args.sequence.read_text(encoding='utf-8-sig'))
    if document['dataset'].get('license') != 'CC-BY-4.0':
        raise ValueError('Explicit licensed sequence required')
    if args.output.exists():
        raise ValueError('Preserve the existing replay')
    selected = json.loads((args.study/'report.json').read_text())
    baseline = args.cache/'adaptations/indian-corpus-encoder.safetensors'
    checkpoints = dict(baseline=baseline)
    if selected.get('saved_checkpoint_sha256'):
        checkpoints['candidate'] = args.study/'candidate.safetensors'
    results = []
    for name, checkpoint in checkpoints.items():
        reader = AwirosExperiment(args.cache,device='gpu',checkpoint=checkpoint)
        for passage in document['passages']:
            readers = {policy:CTCPlateReader(reader.tokens) for policy in ('native','padding','contrast')}
            frames, hashes = [], set()
            for frame in passage['frames']:
                path = (args.sequence.parent/frame['path']).resolve()
                if not path.is_relative_to(args.sequence.parent.resolve()):
                    raise ValueError('Unsafe frame path')
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError('Missing sequence frame')
                source = hashlib.sha256(image.tobytes()).hexdigest()
                if source in hashes:
                    raise ValueError('Repeated original frame')
                hashes.add(source)
                x1,y1,x2,y2 = frame['plate_box']
                crop = image[y1:y2,x1:x2].copy()
                observations = {}
                for policy, view in dict(native=crop,padding=padded_view(crop),contrast=contrast_view(crop)).items():
                    rows = reader.probabilities(view)
                    reading = greedy_text(rows,reader.tokens)
                    readers[policy].add(rows,timestamp_ms=frame['pts_ms'],source_id=source,quality=crop_quality(crop))
                    candidates = readers[policy].candidates()
                    observations[policy] = dict(reading=reading,exact=reading==passage['ground_truth'],
                        running_fused_top=candidates[0].plate if candidates else None)
                frames.append(dict(pts_ms=frame['pts_ms'],source_sha256=file_hash(path),variants=observations))
            variants = {}
            for policy, temporal in readers.items():
                candidates = temporal.candidates()
                accepted = temporal.accepted()
                top = candidates[0].plate if candidates else None
                single_correct = sum(f['variants'][policy]['exact'] for f in frames)
                variants[policy] = dict(distinct_observations=temporal.observation_count,
                    correct_single_frames=single_correct,total_frames=len(frames),
                    fused_top=top,fused_exact=top==passage['ground_truth'],
                    fusion_lost_correct_frame=bool(single_correct and top!=passage['ground_truth']),
                    accepted=accepted.plate if accepted else None,
                    correct_accepted=bool(accepted and accepted.plate==passage['ground_truth']))
            result = dict(model=name,checkpoint_sha256=file_hash(checkpoint),passage_id=passage['id'],
                expected=passage['ground_truth'],timing_mode=passage['timing_mode'],frames=frames,variants=variants)
            results.append(result)
            print(json.dumps(dict(model=name,passage=passage['id'],summary={p:{k:v for k,v in r.items() if k not in ('fused_top','accepted')} for p,r in variants.items()})),flush=True)
        del reader
    args.output.write_text(json.dumps(dict(sequence_manifest_sha256=file_hash(args.sequence),results=results,
        scope='One previously examined development passage, manually supplied boxes, assigned replay timing; not full-video accuracy or natural timestamps'),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--sequence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    run(parser.parse_args())
