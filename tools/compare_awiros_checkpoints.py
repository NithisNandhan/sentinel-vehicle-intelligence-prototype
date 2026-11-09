"""Compare the original Indian-plate Awiros release with our deployed adaptation.

No training, checkpoint selection, threshold tuning or deployment. Identical
frozen crops and preprocessing isolate the effect of the changed weights.
Training photographs are a separately labelled control, never test evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import time

from sentinel_edge.awiros_experiment import AwirosExperiment,MODEL_REVISION,WEIGHTS_SHA256,file_hash
from sentinel_edge.corpus_ocr_adaptation import validate
from sentinel_edge.ctc_fusion import CTCPlateReader,greedy_text
from sentinel_edge.ocr_weight_study import disjoint_training,metrics

DEPLOYED_SHA256='c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d'


def paired_summary(original,adapted):
    if len({r['id'] for r in original})!=len(original) or len({r['id'] for r in adapted})!=len(adapted):
        raise ValueError('Duplicate comparison identity')
    index={row['id']:row for row in original}
    if set(index)!={r['id'] for r in adapted}:raise ValueError('Different comparison samples')
    pairs=[]
    for row in adapted:
        parent=index[row['id']]
        if parent['expected']!=row['expected']:raise ValueError('Different comparison labels')
        pairs.append(dict(id=row['id'],expected=row['expected'],original=parent['reading'],adapted=row['reading'],
            recovered=parent['reading']!=row['expected'] and row['reading']==row['expected'],
            regressed=parent['reading']==row['expected'] and row['reading']!=row['expected']))
    return dict(original=metrics(original),adapted=metrics(adapted),
        recovered=sum(p['recovered'] for p in pairs),regressed=sum(p['regressed'] for p in pairs),
        unchanged_readings=sum(p['original']==p['adapted'] for p in pairs),pairs=pairs)


def run(args):
    import cv2
    import numpy as np
    if args.output.exists():raise ValueError('Preserve existing comparison; use a fresh output')
    root=args.data
    read=lambda p:json.loads(p.read_text())
    corpus_path=root/'indian-corpus/adaptation_manifest.json'
    corpus=read(corpus_path);rows=validate(corpus,corpus_path.parent)
    checkpoint=args.cache/'adaptations/indian-corpus-encoder.safetensors'
    metadata=read(checkpoint.with_suffix('.json'))
    if file_hash(checkpoint)!=DEPLOYED_SHA256 or metadata['parent_sha256']!=WEIGHTS_SHA256:
        raise ValueError('Comparison checkpoint identity changed')
    if metadata['manifest_sha256']!=file_hash(corpus_path):raise ValueError('Original adaptation training manifest changed')
    training=[r for r in rows if r['split']=='train']
    external_path=root/'fresh-indian-20261101/academic/adaptation_manifest.json'
    external=read(external_path)
    if not external.get('no_training'):raise ValueError('External evaluation-only contract changed')
    review=root/'recorded-footage-20261104/manual-review'
    manual_path=review/'ocr-model-dev/rectified-extended-manifest.json'
    manual=read(manual_path)
    if manual['review_manifest_sha256']!=file_hash(review/'manifest.json'):
        raise ValueError('Human review binding changed')
    specs={
        'training_photographs_control':([r for r in training if r['origin']=='photograph_publisher_declared'],corpus_path.parent),
        'corpus_validation':([r for r in rows if r['split']=='validation'],corpus_path.parent),
        'corpus_test':([r for r in rows if r['split']=='test'],corpus_path.parent),
        'external_regression':([r for r in external['samples'] if r['split']=='test'],external_path.parent),
        'mumbai_manual':([r for r in manual['samples'] if r['id']!='21' and not r.get('quarantined')],manual_path.parent)}
    images={};datasets={}
    for name,(selected,directory) in specs.items():
        if name!='training_photographs_control':disjoint_training(training,selected)
        datasets[name]=selected
        for row in selected:
            path=(directory/row['crop_path']).resolve()
            if not path.is_relative_to(directory.resolve()) or file_hash(path)!=row['crop_sha256']:
                raise ValueError('Unsafe or modified comparison crop')
            image=cv2.imread(str(path))
            if image is None:raise ValueError('Unreadable comparison crop')
            images[(name,row['id'])]=image
    road_path=root/'roadtext3k-20261109/cohort-v4/manifest.json'
    road=read(road_path)
    selected=[r for r in road['samples'] if r['split']=='test'];disjoint_training(training,selected)
    datasets['roadtext_test']=selected
    archive=road_path.parent/road['crop_archive']
    if file_hash(archive)!=road['crop_archive_sha256']:raise ValueError('RoadText crops changed')
    all_paths={r['crop_path']:r for r in road['samples']};seen=set()
    ids={r['id'] for r in selected}
    with tarfile.open(archive,mode='r|') as bundle:
        for member in bundle:
            if not member.isfile() or member.name not in all_paths or member.name in seen:
                raise ValueError('Unexpected RoadText crop member')
            seen.add(member.name);row=all_paths[member.name]
            if row['id'] not in ids:continue
            blob=bundle.extractfile(member).read()
            if hashlib.sha256(blob).hexdigest()!=row['crop_sha256']:raise ValueError('RoadText PNG changed')
            image=cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR)
            if image is None or hashlib.sha256(image.tobytes()).hexdigest()!=row['decoded_pixel_sha256']:
                raise ValueError('RoadText native pixels changed')
            images[('roadtext_test',row['id'])]=image
    if seen!=set(all_paths):raise ValueError('Incomplete RoadText crop archive')

    # Both loaders strictly check the publisher architecture and full checkpoint.
    original=AwirosExperiment(args.cache,device='gpu')
    adapted=AwirosExperiment(args.cache,device='gpu',checkpoint=checkpoint)
    if original.tokens!=adapted.tokens:raise ValueError('Different character dictionaries')
    warm=next(iter(images.values()))
    for reader in (original,adapted):
        for _ in range(3):reader.probabilities(warm)
    report=dict(schema_version=1,goal='Measure the effect of our existing fine-tuning against its unchanged original Awiros parent',
        original_sha256=WEIGHTS_SHA256,adapted_sha256=DEPLOYED_SHA256,publisher_revision=MODEL_REVISION,
        source_sha256=file_hash(Path(__file__)),
        manifest_hashes=dict(corpus=file_hash(corpus_path),external=file_hash(external_path),
            manual=file_hash(manual_path),roadtext=file_hash(road_path)),
        original_adaptation=dict(training_photographs=metadata['training_photographs'],
            training_synthetic=metadata['training_synthetic'],scope=metadata['training_scope'],
            epochs_run=metadata['epochs_run'],selected_epoch=metadata['best_validation_epoch']),
        protocol='Same frozen native crops, input_array, dictionary and greedy CTC; alternating model order; no retraining or threshold selection',
        runtime=dict(device='gpu:0',paddle=original.paddle.__version__,numpy=np.__version__),
        training_overlap_checks='All 1,102 adaptation training registrations and crop hashes checked against each evaluation slice',
        source_caveats=dict(corpus=corpus['evaluation_caveat'],roadtext=road['evaluation_caveat']),
        caveats=['Training control is in-sample and cannot establish generalisation.',
            'Corpus validation selected our original checkpoint; it is not an untouched test.',
            'Other regression sets have been examined before; publisher original pretraining membership is unknown.',
            'Crop OCR comparison does not measure automatic detection or full-video accuracy.',
            'OCR latency excludes decode, detection, candidate fusion and alert delivery.'],
        production_changed=False,new_training=False,datasets={})
    args.output.mkdir(parents=True)
    for name,selected in datasets.items():
        outputs={'original':[],'adapted':[]};durations={'original':[],'adapted':[]}
        decoder={'original':[],'adapted':[]}
        for index,row in enumerate(selected):
            pair=[('original',original),('adapted',adapted)]
            if index%2:pair.reverse()
            for identity,reader in pair:
                start=time.perf_counter();posterior=reader.probabilities(images[(name,row['id'])])
                durations[identity].append((time.perf_counter()-start)*1000)
                outputs[identity].append(dict(id=row['id'],expected=row['transcription'],reading=greedy_text(posterior,reader.tokens)))
                if name=='mumbai_manual':
                    fusion=CTCPlateReader(reader.tokens)
                    fusion.add(posterior,timestamp_ms=0,source_id=row['crop_sha256'])
                    candidates=fusion.candidates()
                    decoder[identity].append(dict(id=row['id'],expected=row['transcription'],reading=candidates[0].plate if candidates else ''))
        result=paired_summary(outputs['original'],outputs['adapted'])
        result.update(distinct_registrations=len({r['transcription'] for r in selected}),
            in_sample_control=name=='training_photographs_control',
            timing={identity:dict(median_ocr_ms=statistics.median(values),p95_ocr_ms=float(np.percentile(values,95))) for identity,values in durations.items()})
        if name=='mumbai_manual':result['same_single_frame_candidate_decoder']=paired_summary(decoder['original'],decoder['adapted'])
        report['datasets'][name]=result
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps(dict(dataset=name,**{k:v for k,v in result.items() if k not in ('pairs','same_single_frame_candidate_decoder')})),flush=True)
    print(json.dumps(dict(comparison_complete=True,production_changed=False,new_training=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('/workspace/data/benchmarks'))
    parser.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
