"""Isolated, bounded encoder adaptation on explicitly tiered Indian stills.

Publisher training labels are silver. New evaluation is AI-reviewed diagnostic
data, NOT independent human ground truth. No promotion or live-feed recording.
"""
import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from .awiros_experiment import AwirosExperiment, WEIGHTS_SHA256, adaptation_keys, file_hash, input_array
from .ctc_fusion import greedy_text
from .indian_ocr_finetune import augment, edit_distance


def prepare(root):
    corpus = [json.loads(line) for line in (root / 'corpus.jsonl').read_text().splitlines()]
    review = json.loads((root / 'blind-review/transcriptions.json').read_text())
    index = json.loads((root / 'blind-review/review_index.json').read_text())
    agreed = {r['id'] for r in index if review['readings'].get(str(r['number'])) == r['transcription']}
    excluded, rows = [], []
    for row in corpus:
        if row['quarantined']:
            continue
        split = row['split']
        if row['origin'] == 'photograph_publisher_declared' and split == 'train':
            if row['width'] < 60 or row['height'] < 18 or not 1.2 <= row['width'] / row['height'] <= 9:
                excluded.append({'id': row['id'], 'reason': 'size_or_aspect_quality_gate'})
                continue
            tier = 'legacy_visual_review' if row['review'] == 'visually_verified' else 'publisher_silver'
        elif row['id'] in agreed:
            split = 'validation' if int(hashlib.sha256(row['transcription'].encode()).hexdigest(), 16) % 2 else 'test'
            tier = 'AI_blind_visual_publisher_agreement_not_human_truth'
        elif split in {'legacy_validation', 'legacy_test'}:
            split = split.replace('legacy_', '')
            tier = 'legacy_visual_review'
        else:
            continue
        rows.append({**row, 'split': split, 'label_tier': tier})
    # Supplement real-photo training with at most 25% generated examples per batch.
    synthetic = sorted((r for r in corpus if r['origin'] == 'synthetic' and r['split'] == 'train'), key=lambda r: r['id'])
    rows += [{**r, 'label_tier': 'generator_label'} for r in synthetic[:300]]
    document = {'schema_version': 1, 'corpus_sha256': file_hash(root / 'corpus.jsonl'),
                'review_sha256': file_hash(root / 'blind-review/transcriptions.json'),
                'evaluation_caveat': 'AI visual review, publisher agreement and legacy reviews; not independent human test truth; pretraining overlap unknown',
                'samples': rows, 'quality_rejections': excluded,
                'synthetic_per_batch_max': .25, 'production_promoted': False}
    validate(document, root)
    (root / 'adaptation_manifest.json').write_text(json.dumps(document, indent=2))
    print(json.dumps({'prepared': len(rows), 'splits': {s: sum(r['split'] == s for r in rows) for s in ('train','validation','test')},
                      'photographic_train': sum(r['split']=='train' and r['origin']=='photograph_publisher_declared' for r in rows),
                      'quality_rejected': len(excluded), 'AI_agreed_holdout_crops': len(agreed)}), flush=True)


def validate(document, root):
    if document.get('schema_version') != 1 or not document.get('evaluation_caveat'):
        raise ValueError('Explicit corpus adaptation contract required')
    rows = document['samples']
    if not 50 <= len(rows) <= 2000:
        raise ValueError('Adaptation corpus exceeds bounded experiment')
    groups, hashes = {}, set()
    for row in rows:
        if row['split'] not in {'train','validation','test'} or row['origin'] not in {'synthetic','photograph_publisher_declared'}:
            raise ValueError('Unsupported split or origin')
        if row['split'] != 'train' and (row['origin']=='synthetic' or row['label_tier']=='publisher_silver'):
            raise ValueError('Unreviewed evaluation label')
        path = (root / row['crop_path']).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != row['crop_sha256']:
            raise ValueError('Modified or unsafe crop')
        text = row['transcription']
        if not 6 <= len(text) <= 15 or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' for c in text):
            raise ValueError('Unsupported transcription')
        if text in groups and groups[text] != row['split']:
            raise ValueError('Registration split leakage')
        if row['id'] in hashes:
            raise ValueError('Duplicate native crop')
        groups[text] = row['split']
        hashes.add(row['id'])
    if any(sum(r['split']==s for r in rows)<8 for s in ('validation','test')):
        raise ValueError('Insufficient reviewed diagnostic holdout')
    return rows


def train(manifest, cache, output, report, epochs=12):
    import cv2
    import numpy as np
    from safetensors.numpy import save_file
    document = json.loads(manifest.read_text())
    rows = validate(document, manifest.parent)
    if not 1 <= epochs <= 30:
        raise ValueError('Epoch limit exceeded')
    reader = AwirosExperiment(cache, device='gpu')
    p, model = reader.paddle, reader.model
    seed = 20260928
    random.seed(seed); np.random.seed(seed); p.seed(seed)
    rng = np.random.default_rng(seed)
    state = {k:v.numpy().copy() for k,v in model.state_dict().items()}
    keys = adaptation_keys(state, 'bounded_image_encoder_tail_and_ctc')
    for name, parameter in model.named_parameters():
        parameter.stop_gradient = name not in keys
    # Freeze batch-normalisation statistics/dropout, but allow selected gradients.
    model.eval()
    # The publisher's guide detaches backbone features. Disable that training-
    # only gradient detach; forward values and inference architecture are unchanged.
    model.head.ctc_encoder.encoder.use_guide = False
    images = {r['id']: cv2.imread(str(manifest.parent / r['crop_path'])) for r in rows}
    if any(image is None for image in images.values()):
        raise ValueError('Cannot decode training crop')
    lookup = {c:i for i,c in enumerate(reader.tokens)}
    loss_fn = p.nn.CTCLoss(blank=0, reduction='none')

    def forward(batch, jitter=False):
        arrays = [input_array(augment(images[r['id']], rng.integers(0,4), rng) if jitter else images[r['id']]) for r in batch]
        x = p.to_tensor(np.concatenate(arrays))
        return model.head.ctc_head.fc(model.head.ctc_encoder(model.backbone(x)))

    def evaluate(split):
        selected = [r for r in rows if r['split']==split]
        exact=errors=characters=0; readings=[]; durations=[]
        with p.no_grad():
            for r in selected:
                start=time.perf_counter()
                probs=p.nn.functional.softmax(forward([r]),axis=2).numpy()[0]
                durations.append((time.perf_counter()-start)*1000)
                text=greedy_text(probs.tolist(),reader.tokens)
                truth=r['transcription']; exact += text==truth
                errors += edit_distance(text,truth); characters += len(truth)
                readings.append({'id':r['id'],'expected':truth,'reading':text,'label_tier':r['label_tier']})
        return {'samples':len(selected),'exact':exact,'character_errors':errors,'characters':characters,
                'cer':errors/characters,'median_ocr_ms':float(np.median(durations)), 'readings':readings}

    baseline=evaluate('validation')
    photo=[r for r in rows if r['split']=='train' and r['origin']=='photograph_publisher_declared']
    synthetic=[r for r in rows if r['split']=='train' and r['origin']=='synthetic']
    parameters=[parameter for name,parameter in model.named_parameters() if name in keys]
    optimizer=p.optimizer.Adam(learning_rate=.00002,parameters=parameters,grad_clip=p.nn.ClipGradByGlobalNorm(1.0))
    best=baseline; best_state={k:state[k].copy() for k in keys}; best_epoch=0; stale=0; history=[]
    for epoch in range(1,epochs+1):
        random.shuffle(photo); losses=[]
        for offset in range(0,len(photo),3):
            batch=photo[offset:offset+3]
            if len(batch)==3 and synthetic:
                batch=batch+[random.choice(synthetic)]
            labels=[[lookup[c] for c in r['transcription']] for r in batch]
            padded=np.zeros((len(batch),max(map(len,labels))),dtype=np.int32)
            for i,label in enumerate(labels): padded[i,:len(label)]=label
            logits=forward(batch,jitter=True)
            loss=loss_fn(logits.transpose((1,0,2)),p.to_tensor(padded),
                         p.to_tensor([logits.shape[1]]*len(batch),dtype='int64',place=p.CPUPlace()),
                         p.to_tensor([len(v) for v in labels],dtype='int64')).mean()
            if not np.isfinite(loss.numpy()).all(): raise ValueError('Nonfinite training loss')
            loss.backward(); optimizer.step(); optimizer.clear_grad()
            losses.append(float(loss.numpy()))
        metrics=evaluate('validation')
        entry={'epoch':epoch,'training_loss':sum(losses)/len(losses),'validation':{k:v for k,v in metrics.items() if k!='readings'}}
        history.append(entry); print(json.dumps(entry),flush=True)
        if (-metrics['exact'],metrics['cer']) < (-best['exact'],best['cer']):
            best=metrics; best_epoch=epoch; stale=0
            best_state={k:v.numpy().copy() for k,v in model.state_dict().items() if k in keys}
        else: stale+=1
        if stale>=4: break
    final={k:v.numpy().copy() for k,v in model.state_dict().items()}
    if any(not np.array_equal(state[k],final[k]) for k in state if k not in keys):
        raise ValueError('Frozen encoder or batch norm changed')
    if not any(not np.array_equal(state[k],final[k]) for k in keys if k.startswith('backbone.')):
        raise ValueError('Selected image encoder did not learn')
    # Keep trained rejected weights for an honest before/after diagnostic.
    chosen=dict(state); chosen.update(best_state if best_epoch else {k:final[k] for k in keys})
    output.parent.mkdir(parents=True,exist_ok=True)
    save_file(chosen,str(output))
    model.set_state_dict({k:p.to_tensor(v) for k,v in chosen.items()})
    tuned_validation=evaluate('validation'); tuned_test=evaluate('test')
    with p.no_grad():
        parity_rows=[r for r in rows if r['split']=='test'][:3]
        expected_probs=[p.nn.functional.softmax(forward([r]),axis=2).numpy()[0] for r in parity_rows]
    model.set_state_dict({k:p.to_tensor(v) for k,v in state.items()})
    baseline_test=evaluate('test')
    result={'generated_at':datetime.now(timezone.utc).isoformat(),'parent_sha256':WEIGHTS_SHA256,
            'dictionary_sha256':file_hash(cache/'en_dict.txt'),'weights_sha256':file_hash(output),
            'training_scope':'bounded_image_encoder_tail_and_ctc','frozen_state_unchanged':True,
            'selected_keys':sorted(keys),'trainable_parameters':sum(state[k].size for k in keys),
            'manifest_sha256':file_hash(manifest),'device':'gpu','seed':seed,'learning_rate':.00002,
            'synthetic_per_batch_max':.25,'batch_size':4,'epochs_run':len(history),'best_validation_epoch':best_epoch,
            'validation_preferred_adaptation':best_epoch>0,'baseline_validation':baseline,'tuned_validation':tuned_validation,
            'baseline_test':baseline_test,'tuned_test':tuned_test,'history':history,
            'evaluation_caveat':document['evaluation_caveat'],'pretraining_overlap_unknown':True,
            'production_promoted':False,'photographic_training_labels_include_unverified_publisher_silver':True,
            'training_photographs':len(photo),'training_synthetic':len(synthetic)}
    output.with_suffix('.json').write_text(json.dumps(result,indent=2))
    reloaded=AwirosExperiment(cache,device='gpu',checkpoint=output)
    for row,expected in zip(parity_rows,expected_probs):
        if not np.allclose(reloaded.probabilities(images[row['id']]),expected,atol=1e-5):
            raise ValueError('Reloaded encoder checkpoint failed inference parity')
    result['full_checkpoint_reload_parity_passed']=True
    output.with_suffix('.json').write_text(json.dumps(result,indent=2))
    report.parent.mkdir(parents=True,exist_ok=True); report.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('epochs_run','best_validation_epoch','trainable_parameters','validation_preferred_adaptation')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--epochs',type=int,default=12)
    args=parser.parse_args()
    if args.prepare: prepare(args.prepare)
    elif all((args.manifest,args.cache,args.output,args.report)): train(args.manifest,args.cache,args.output,args.report,args.epochs)
    else: parser.error('Provide preparation root or complete training paths')
