"""Warm-start RoadText-only adaptation, followed by frozen regression evaluation.

Select using RoadText validation only. Stop after this dataset's report; never
start another dataset, alter runtime weights, or weaken alert gates here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

from sentinel_edge.awiros_experiment import AwirosExperiment,WEIGHTS_SHA256,adaptation_keys,file_hash,input_array
from sentinel_edge.ctc_fusion import CTCPlateReader,greedy_text
from sentinel_edge.ocr_weight_study import disjoint_training,metrics,quality_key
from sentinel_edge.roadtext_corpus import check_splits

BASELINE_SHA256='c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d'


def run(args):
    import cv2
    import numpy as np
    from safetensors.numpy import save_file
    if args.output.exists():raise ValueError('Use a fresh RoadText training output')
    args.output.mkdir(parents=True)
    document=json.loads(args.manifest.read_text())
    rows=document['samples'];check_splits(rows)
    archive=args.manifest.parent/document['crop_archive']
    if file_hash(archive)!=document['crop_archive_sha256']:raise ValueError('RoadText native-crop archive changed')
    expected={r['crop_path']:r for r in rows}
    images={}
    with tarfile.open(archive,mode='r|') as bundle:
        for member in bundle:
            if not member.isfile() or member.name not in expected or member.size>8*1024*1024:
                raise ValueError('Unexpected RoadText crop member')
            row=expected[member.name]
            blob=bundle.extractfile(member).read()
            if hashlib.sha256(blob).hexdigest()!=row['crop_sha256']:raise ValueError('RoadText crop bytes changed')
            image=cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR)
            if (image is None or list(image.shape[:2])!=[row['height'],row['width']]
                    or hashlib.sha256(image.tobytes()).hexdigest()!=row['decoded_pixel_sha256']):
                raise ValueError('RoadText native pixels changed')
            if row['id'] in images:raise ValueError('Duplicate RoadText sample identity')
            images[row['id']]=image
    if len(images)!=len(rows):raise ValueError('Incomplete RoadText crop archive')
    training=[r for r in rows if r['split']=='train']
    validation=[r for r in rows if r['split']=='validation']
    testing=[r for r in rows if r['split']=='test']
    for split in (training,validation,testing):
        if len({r['transcription'] for r in split})<5:raise ValueError('Insufficient grouped RoadText data')
    disjoint_training(training,validation+testing)
    checkpoint=args.cache/'adaptations/indian-corpus-encoder.safetensors'
    if file_hash(checkpoint)!=BASELINE_SHA256:raise ValueError('Approved baseline checkpoint changed')
    reader=AwirosExperiment(args.cache,device='gpu',checkpoint=checkpoint)
    p,model=reader.paddle,reader.model
    seed=20261109;p.seed(seed)
    original={k:v.numpy().copy() for k,v in model.state_dict().items()}
    model.eval();model.head.ctc_encoder.encoder.use_guide=False
    lookup={char:i for i,char in enumerate(reader.tokens)}
    criterion=p.nn.CTCLoss(blank=0,reduction='none')

    def restore(state):model.set_state_dict({k:p.to_tensor(v) for k,v in state.items()})
    def logits(arrays):
        tensor=p.to_tensor(np.concatenate([input_array(image) for image in arrays]))
        return model.head.ctc_head.fc(model.head.ctc_encoder(model.backbone(tensor)))
    def read(image):
        with p.no_grad():return p.nn.functional.softmax(logits([image]),axis=2).numpy()[0].tolist()
    def evaluate(selected,lookup_images,temporal=False):
        results=[];durations=[];groups={}
        for row in selected:
            start=time.monotonic();posterior=read(lookup_images[row['id']]);reading=greedy_text(posterior,reader.tokens)
            durations.append((time.monotonic()-start)*1000)
            results.append(dict(id=row['id'],expected=row['transcription'],reading=reading))
            if temporal:groups.setdefault((row['video_id'],row['track_id']),[]).append((row,posterior,reading))
        fused=[]
        for (video,track),observations in groups.items():
            fusion=CTCPlateReader(reader.tokens);readings=[]
            truth=observations[0][0]['transcription']
            first_accepted=None;ever_wrong=False;ever_correct=False
            for row,posterior,reading in sorted(observations,key=lambda value:value[0]['pts_ms']):
                if fusion.samples and row['pts_ms']-fusion.samples[-1][0]>2500:
                    fusion=CTCPlateReader(reader.tokens);readings=[]
                if fusion.add(posterior,timestamp_ms=row['pts_ms'],source_id=row['decoded_pixel_sha256'],quality=row['quality']):
                    readings.append(reading)
                    interim=fusion.accepted()
                    if interim:
                        if first_accepted is None:first_accepted=interim.plate
                        ever_wrong=ever_wrong or interim.plate!=truth
                        ever_correct=ever_correct or interim.plate==truth
            candidates=fusion.candidates();accepted=fusion.accepted()
            top=candidates[0].plate if candidates else None
            fused.append(dict(video_id=video,track_id=track,expected=truth,reading=top or '',
                observations=fusion.observation_count,exact=top==truth,
                accepted=accepted.plate if accepted else None,
                accepted_correct=bool(accepted and accepted.plate==truth),
                accepted_wrong=bool(accepted and accepted.plate!=truth),
                ever_wrong_accepted=ever_wrong,ever_correct_accepted=ever_correct,
                first_accepted=first_accepted,
                fusion_lost_correct_frame=bool(truth in readings and top!=truth)))
        return dict(summary=metrics(results),median_ocr_ms=float(np.median(durations)),results=results,
            track_summary=dict(tracks=len(fused),exact=sum(r['exact'] for r in fused),
                accepted_correct=sum(r['accepted_correct'] for r in fused),
                accepted_wrong=sum(r['accepted_wrong'] for r in fused),
                ever_wrong_accepted_tracks=sum(r['ever_wrong_accepted'] for r in fused),
                ever_correct_accepted_tracks=sum(r['ever_correct_accepted'] for r in fused),
                first_acceptance_wrong=sum(bool(r['first_accepted'] and r['first_accepted']!=r['expected']) for r in fused),
                fusion_lost_correct_frame=sum(r['fusion_lost_correct_frame'] for r in fused)) if temporal else None,
            tracks=fused)

    # Validate legacy admission before gradients; these labels never enter fitting or checkpoint choice.
    legacy_root=args.legacy
    corpus_document=json.loads((legacy_root/'indian-corpus/adaptation_manifest.json').read_text())
    external_document=json.loads((legacy_root/'fresh-indian-20261101/academic/adaptation_manifest.json').read_text())
    if not external_document.get('no_training'):raise ValueError('External regression-only contract changed')
    review=legacy_root/'recorded-footage-20261104/manual-review'
    manual_document=json.loads((review/'ocr-model-dev/rectified-extended-manifest.json').read_text())
    if manual_document['review_manifest_sha256']!=file_hash(review/'manifest.json'):
        raise ValueError('Manual reference manifest changed')
    specs=dict(corpus_validation=([r for r in corpus_document['samples'] if r['split']=='validation'],legacy_root/'indian-corpus'),
        corpus_test=([r for r in corpus_document['samples'] if r['split']=='test'],legacy_root/'indian-corpus'),
        external=([r for r in external_document['samples'] if r['split']=='test'],legacy_root/'fresh-indian-20261101/academic'),
        mumbai_manual=([r for r in manual_document['samples'] if r['id']!='21'],review/'ocr-model-dev'))
    regression={}
    for name,(selected,root) in specs.items():
        disjoint_training(training,selected)
        decoded={}
        for row in selected:
            path=(root/row['crop_path']).resolve()
            if not path.is_relative_to(root.resolve()) or file_hash(path)!=row['crop_sha256']:
                raise ValueError('Unsafe or changed regression crop')
            image=cv2.imread(str(path))
            if image is None:raise ValueError('Cannot decode regression image')
            decoded[row['id']]=image
        regression[name]=(selected,decoded)

    first=images[validation[0]['id']]
    if not np.allclose(read(first),reader.probabilities(first),atol=1e-5,rtol=1e-4):
        raise ValueError('RoadText training/serving forward mismatch')
    baseline_validation=evaluate(validation,images)
    report=dict(schema_version=1,dataset=document['dataset'],manifest_sha256=file_hash(args.manifest),
        baseline_checkpoint_sha256=file_hash(checkpoint),source_sha256=file_hash(Path(__file__)),
        runtime=dict(paddle=p.__version__,numpy=np.__version__,device=p.device.get_device(),
                     batch_size=4,epochs_per_configuration=args.epochs),
        seed=seed,split_policy=document['split_policy'],source_selection=document['selection'],
        cohort=document['summary'],train_only_data='RoadText train split; no other dataset added to optimizer batches',
        temporal_caveat='Timestamped replay of retained native annotated crops; not end-to-end detector/tracker video accuracy',
        pretraining_caveat='New adaptation splits are disjoint; publisher pretraining data membership cannot be established',
        selection='RoadText validation exact first, character edits second; final test and regressions used only after selection',
        evaluation_caveat=document['evaluation_caveat'],baseline_validation=baseline_validation,
        history=[],production_promoted=False,next_dataset_started=False)
    print(json.dumps(dict(stage='roadtext_baseline',**baseline_validation['summary'])),flush=True)
    best=None;best_state=None;best_scope=None;best_identity=None
    configs=[('tail','bounded_image_encoder_tail_and_ctc',3e-5),('full','full_image_encoder_and_ctc',5e-6)]
    for name,scope,learning_rate in configs:
        restore(original);p.seed(seed);rng=np.random.default_rng(seed)
        keys=adaptation_keys(original,scope)
        for key,param in model.named_parameters():param.stop_gradient=key not in keys
        optimizer=p.optimizer.Adam(learning_rate=learning_rate,
            parameters=[param for key,param in model.named_parameters() if key in keys],
            grad_clip=p.nn.ClipGradByGlobalNorm(1.))
        for epoch in range(1,args.epochs+1):
            order=rng.permutation(len(training));losses=[];started=time.monotonic()
            for offset in range(0,len(order),4):
                batch=[training[i] for i in order[offset:offset+4]]
                arrays=[]
                for row in batch:
                    image=images[row['id']]
                    if rng.random()<.5:
                        image=np.clip(image.astype(np.float32)*rng.uniform(.9,1.1)+rng.uniform(-8,8),0,255).astype(np.uint8)
                    arrays.append(image)
                labels=[[lookup[c] for c in row['transcription']] for row in batch]
                padded=np.zeros((len(batch),max(map(len,labels))),np.int32)
                for i,label in enumerate(labels):padded[i,:len(label)]=label
                values=logits(arrays)
                loss=criterion(values.transpose((1,0,2)),p.to_tensor(padded),
                    p.to_tensor([values.shape[1]]*len(batch),dtype='int64',place=p.CPUPlace()),
                    p.to_tensor([len(label) for label in labels],dtype='int64')).mean()
                scalar=float(loss.numpy())
                if not np.isfinite(scalar):raise ValueError('Nonfinite RoadText training loss')
                loss.backward();optimizer.step();optimizer.clear_grad();losses.append(scalar)
            measured=evaluate(validation,images)
            entry=dict(configuration=name,scope=scope,learning_rate=learning_rate,epoch=epoch,
                loss=float(np.mean(losses)),optimizer_steps=len(losses),validation=measured['summary'],
                seconds=round(time.monotonic()-started,2))
            report['history'].append(entry);print(json.dumps(dict(stage='roadtext_training',**entry)),flush=True)
            current={k:v.numpy().copy() for k,v in model.state_dict().items()}
            if any(not np.array_equal(current[k],original[k]) for k in original if k not in keys):
                raise ValueError('Frozen auxiliary or excluded weights changed')
            if best is None or quality_key(measured['summary'])<quality_key(best):
                best=measured['summary'];best_state=current;best_scope=scope
                best_identity=dict(configuration=name,epoch=epoch,learning_rate=learning_rate)
            (args.output/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        del optimizer
    # Save and freeze the best trained checkpoint before opening the test results.
    destination=args.output/'roadtext-candidate.safetensors'
    save_file(best_state,str(destination))
    metadata=dict(parent_sha256=WEIGHTS_SHA256,dictionary_sha256=file_hash(args.cache/'en_dict.txt'),
        initial_checkpoint_sha256=file_hash(checkpoint),weights_sha256=file_hash(destination),
        training_scope=best_scope,frozen_state_unchanged=True,manifest_sha256=file_hash(args.manifest),
        selected=best_identity,production_promoted=False,source_sha256=file_hash(Path(__file__)))
    destination.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    report['selected']=best_identity;report['candidate_checkpoint_sha256']=metadata['weights_sha256']
    report['validation_improved']=quality_key(best)<quality_key(baseline_validation['summary'])
    (args.output/'selection.json').write_text(json.dumps(dict(selected=best_identity,validation=best,test_used_for_selection=False),indent=2))

    outcome={}
    for identity,state in [('baseline',original),('candidate',best_state)]:
        restore(state)
        results=dict(roadtext_validation=evaluate(validation,images),roadtext_test=evaluate(testing,images,temporal=True))
        results.update({name:evaluate(selected,decoded) for name,(selected,decoded) in regression.items()})
        outcome[identity]=results
        print(json.dumps(dict(stage='roadtext_regressions',model=identity,
            summary={name:value['summary'] for name,value in results.items()})),flush=True)
    eligible=report['validation_improved'];reasons=[]
    if not eligible:reasons.append('No RoadText validation improvement')
    for name in outcome['baseline']:
        baseline=outcome['baseline'][name]['summary'];candidate=outcome['candidate'][name]['summary']
        if candidate['exact']<baseline['exact'] or candidate['character_errors']>baseline['character_errors']:
            eligible=False;reasons.append('Regression on '+name)
    baseline_tracks=outcome['baseline']['roadtext_test']['track_summary']
    candidate_tracks=outcome['candidate']['roadtext_test']['track_summary']
    if (candidate_tracks['exact']<baseline_tracks['exact']
            or candidate_tracks['accepted_wrong']>baseline_tracks['accepted_wrong']
            or candidate_tracks['ever_wrong_accepted_tracks']>baseline_tracks['ever_wrong_accepted_tracks']):
        eligible=False;reasons.append('Regression on RoadText temporal track results')
    restored=AwirosExperiment(args.cache,device='gpu',checkpoint=destination)
    restore(best_state)
    if not np.allclose(restored.probabilities(first),read(first),atol=1e-5,rtol=1e-4):
        raise ValueError('RoadText saved checkpoint failed serving parity')
    report.update(results=outcome,promotion_eligible=eligible,promotion_reasons=reasons,
        saved_reload_parity_passed=True,benchmark_review_required_before_next_dataset=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(stage='roadtext_complete',selected=best_identity,promotion_eligible=eligible,
        reasons=reasons,next_dataset_started=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--legacy',type=Path,default=Path('/workspace/data/benchmarks'))
    parser.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    parser.add_argument('--epochs',type=int,default=4,choices=range(1,9))
    run(parser.parse_args())
