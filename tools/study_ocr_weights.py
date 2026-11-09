"""GPU crop/decoder diagnosis and train-only OCR weight search. No live promotion.

Detailed reports contain plate labels and stay under ignored data/. All candidate
selection uses the existing corpus validation split, never the external labels.
This is a regression study on previously examined data, not a fresh blind test.
"""
import argparse
import json
import random
import time
from pathlib import Path

from sentinel_edge.awiros_experiment import AwirosExperiment, WEIGHTS_SHA256, adaptation_keys, file_hash, input_array
from sentinel_edge.corpus_ocr_adaptation import validate
from sentinel_edge.ctc_fusion import CTCPlateReader, greedy_text, prefix_candidates, sequence_score
from sentinel_edge.indian_ocr_finetune import augment
from sentinel_edge.ocr_crop_experiment import contrast_view, padded_view
from sentinel_edge.ocr_weight_study import character_errors, disjoint_training, metrics, promotion_decision, quality_key


def run(args):
    import cv2
    import numpy as np
    from safetensors.numpy import save_file
    if args.output.exists():
        raise ValueError('Use a new output directory; preserve earlier measurements')
    args.output.mkdir(parents=True)
    seed = 20261109
    random.seed(seed)
    rng = np.random.default_rng(seed)
    corpus_path = args.corpus / 'adaptation_manifest.json'
    corpus = json.loads(corpus_path.read_text())
    corpus_rows = validate(corpus, args.corpus)
    training = [r for r in corpus_rows if r['split'] == 'train' and r['origin'] == 'photograph_publisher_declared']
    validation = [r for r in corpus_rows if r['split'] == 'validation']
    corpus_test = [r for r in corpus_rows if r['split'] == 'test']
    external_path = args.external / 'adaptation_manifest.json'
    external_document = json.loads(external_path.read_text())
    if not external_document.get('no_training'):
        raise ValueError('External evaluation-only contract required')
    external = [r for r in external_document['samples'] if r['split'] == 'test']
    manual_path = args.review / 'ocr-model-dev/rectified-extended-manifest.json'
    manual_document = json.loads(manual_path.read_text())
    original_document = json.loads((args.review / 'ocr-development-manifest.json').read_text())
    if manual_document['review_manifest_sha256'] != file_hash(args.review / 'manifest.json'):
        raise ValueError('Manual reference binding changed')
    originals = {r['id']: r for r in original_document['samples']}
    # 21 was explicitly disputed by the human reviewer. Do not train/evaluate it.
    manual = [r for r in manual_document['samples'] if r['id'] != '21' and not r.get('quarantined')]
    for row in manual:
        if row['transcription'] != originals[row['id']]['transcription']:
            raise ValueError('Manual labels disagree between geometry versions')
    disjoint_training(training, validation + corpus_test + external + manual)

    def images(rows, root):
        result = {}
        for row in rows:
            path = (root / row['crop_path']).resolve()
            if not path.is_relative_to(root.resolve()) or file_hash(path) != row['crop_sha256']:
                raise ValueError('Unsafe or modified crop')
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot decode checked crop')
            result[row['id']] = image
        return result

    training_images = images(training, args.corpus)
    datasets = dict(validation=(validation, images(validation, args.corpus)),
                    manual=(manual, images(manual, manual_path.parent)),
                    corpus_test=(corpus_test, images(corpus_test, args.corpus)),
                    external=(external, images(external, args.external)))
    original_images = images([originals[r['id']] for r in manual], args.review)
    checkpoint = args.cache / 'adaptations/indian-corpus-encoder.safetensors'
    reader = AwirosExperiment(args.cache, device='gpu', checkpoint=checkpoint)
    p, model = reader.paddle, reader.model
    p.seed(seed)
    baseline_state = {k:v.numpy().copy() for k,v in model.state_dict().items()}
    model.eval()
    model.head.ctc_encoder.encoder.use_guide = False

    def restore(state):
        model.set_state_dict({k:p.to_tensor(v) for k,v in state.items()})

    def logits(arrays):
        tensor = p.to_tensor(np.concatenate([input_array(image) for image in arrays]))
        return model.head.ctc_head.fc(model.head.ctc_encoder(model.backbone(tensor)))

    def posterior(image):
        with p.no_grad():
            return p.nn.functional.softmax(logits([image]), axis=2).numpy()[0].tolist()

    def evaluation(name):
        selected, cache = datasets[name]
        rows = [dict(id=r['id'], expected=r['transcription'], reading=greedy_text(posterior(cache[r['id']]),reader.tokens)) for r in selected]
        return dict(summary=metrics(rows), rows=rows)

    # Require the training-forward path to match the actual serving path.
    first = datasets['validation'][1][validation[0]['id']]
    if not np.allclose(posterior(first), reader.probabilities(first), atol=1e-5, rtol=1e-4):
        raise ValueError('Training/inference forward parity failed')
    baseline_validation = evaluation('validation')
    print(json.dumps(dict(stage='baseline_validation', **baseline_validation['summary'])), flush=True)
    report = dict(schema_version=1, seed=seed, baseline_checkpoint_sha256=file_hash(checkpoint),
                  source_sha256=file_hash(Path(__file__)), corpus_manifest_sha256=file_hash(corpus_path),
                  external_manifest_sha256=file_hash(external_path), manual_manifest_sha256=file_hash(manual_path),
                  scope='Previously examined labelled crop regression; not independent CCTV or unseen-source accuracy',
                  train_photos=len(training), evaluation_used_for_training=False,
                  candidate_selection='corpus validation exact first, character edit count second; baseline wins ties',
                  input_crop_pipeline_changed=False, forward_parity_passed=True,
                  diagnosis=[], weight_trials=[], production_promoted=False)

    # Diagnose image geometry, pruning and fusion with frozen weights first.
    for row in ([] if args.reuse_diagnosis else manual):
        image = datasets['manual'][1][row['id']]
        variants = dict(native=original_images[row['id']], rectified=image,
                        padding=padded_view(image), contrast=contrast_view(image))
        readings = {}
        for name, view in variants.items():
            values = posterior(view)
            text = greedy_text(values,reader.tokens)
            beam = prefix_candidates(values,reader.tokens,beam_width=100,max_length=25)
            truth_log, _ = sequence_score(values,reader.tokens,row['transcription'])
            read_log, _ = sequence_score(values,reader.tokens,text)
            one = CTCPlateReader(reader.tokens)
            one.add(values,timestamp_ms=0,source_id='original')
            candidates = one.candidates()
            readings[name] = dict(reading=text, beam_top=beam[0][0] if beam else '',
                truth_in_beam_100=any(s.upper().replace(' ','') == row['transcription'] for s,_ in beam),
                truth_log_gap=float(read_log-truth_log) if np.isfinite(read_log-truth_log) else None,
                candidate_top=candidates[0].plate if candidates else '',
                errors=character_errors(row['transcription'],text),
                model_input_shape=list(input_array(view).shape))
        report['diagnosis'].append(dict(id=row['id'],expected=row['transcription'],variants=readings))
    if args.reuse_diagnosis:
        previous = json.loads(args.reuse_diagnosis.read_text())
        for key in ('baseline_checkpoint_sha256','corpus_manifest_sha256','manual_manifest_sha256','external_manifest_sha256'):
            if previous[key] != report[key]:
                raise ValueError('Cannot reuse a diagnosis from changed inputs')
        report['diagnosis'] = previous['diagnosis']
        report['reused_diagnosis_report_sha256'] = file_hash(args.reuse_diagnosis)
    (args.output/'diagnosis.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(stage='geometry_diagnosis', summaries={name:metrics([
        dict(expected=r['expected'],reading=r['variants'][name]['reading']) for r in report['diagnosis']])
        for name in ('native','rectified','padding','contrast')})),flush=True)

    # Change a real CTC bias weight (blank class), not confidence thresholds.
    # Cache validation logits: adding a bias equals changing this layer's bias.
    with p.no_grad():
        validation_logits = {r['id']:logits([datasets['validation'][1][r['id']]]).numpy()[0] for r in validation}
    best_summary = baseline_validation['summary']
    best_state, best_name, best_scope = baseline_state, 'baseline', 'bounded_image_encoder_tail_and_ctc'
    for delta in (-1., -.5, -.25, .25, .5, 1.):
        rows = []
        for r in validation:
            values = validation_logits[r['id']].copy()
            values[:,0] += delta
            values = np.exp(values - values.max(axis=1,keepdims=True))
            values /= values.sum(axis=1,keepdims=True)
            rows.append(dict(expected=r['transcription'],reading=greedy_text(values.tolist(),reader.tokens)))
        summary = metrics(rows)
        name = f'blank_bias_{delta:+g}'
        report['weight_trials'].append(dict(name=name,validation=summary,changed_parameters=1))
        if quality_key(summary) < quality_key(best_summary):
            best_state = {k:v.copy() for k,v in baseline_state.items()}
            best_state['head.ctc_head.fc.bias'][0] += delta
            best_summary, best_name = summary, name
    print(json.dumps(dict(stage='blank_weight_sweep',best=best_name,validation=best_summary)),flush=True)

    lookup = {char:i for i,char in enumerate(reader.tokens)}
    criterion = p.nn.CTCLoss(blank=0,reduction='none')
    configs = [('head_low','ctc_linear_head_only',1e-5),('head_high','ctc_linear_head_only',5e-5),
               ('encoder_low','bounded_image_encoder_tail_and_ctc',2e-6),
               ('encoder_high','bounded_image_encoder_tail_and_ctc',1e-5)]
    if args.extended:
        configs = [('head_aggressive','ctc_linear_head_only',2e-4),
                   ('encoder_aggressive','bounded_image_encoder_tail_and_ctc',5e-5),
                   ('full_encoder','full_image_encoder_and_ctc',1e-5)]
    for name, scope, learning_rate in configs:
        rng = np.random.default_rng(seed)
        p.seed(seed)
        restore(baseline_state)
        keys = adaptation_keys(baseline_state,scope)
        for key,parameter in model.named_parameters():
            parameter.stop_gradient = key not in keys
        optimizer = p.optimizer.Adam(learning_rate=learning_rate,
            parameters=[parameter for key,parameter in model.named_parameters() if key in keys],
            grad_clip=p.nn.ClipGradByGlobalNorm(1.))
        selected_summary, selected_state = baseline_validation['summary'], None
        history = []
        for epoch in range(1,args.epochs+1):
            order = list(training)
            random.Random(seed+epoch).shuffle(order)
            losses = []
            started = time.monotonic()
            for offset in range(0,len(order),4):
                batch = order[offset:offset+4]
                arrays = [augment(training_images[r['id']],int(rng.integers(0,4)),rng) for r in batch]
                labels = [[lookup[c] for c in r['transcription']] for r in batch]
                padded = np.zeros((len(batch),max(map(len,labels))),np.int32)
                for i,label in enumerate(labels):
                    padded[i,:len(label)] = label
                values = logits(arrays)
                loss = criterion(values.transpose((1,0,2)),p.to_tensor(padded),
                    p.to_tensor([values.shape[1]]*len(batch),dtype='int64',place=p.CPUPlace()),
                    p.to_tensor([len(label) for label in labels],dtype='int64')).mean()
                scalar = float(loss.numpy())
                if not np.isfinite(scalar):
                    raise ValueError('Nonfinite weight adaptation loss')
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                losses.append(scalar)
            summary = evaluation('validation')['summary']
            entry = dict(name=name,epoch=epoch,steps=len(losses),loss=float(np.mean(losses)),
                         validation=summary,seconds=round(time.monotonic()-started,2))
            history.append(entry)
            print(json.dumps(dict(stage='training',**entry)),flush=True)
            if quality_key(summary) < quality_key(selected_summary):
                selected_summary = summary
                selected_state = {k:v.numpy().copy() for k,v in model.state_dict().items() if k in keys}
        final_state = {k:v.numpy().copy() for k,v in model.state_dict().items()}
        if any(not np.array_equal(final_state[k],baseline_state[k]) for k in baseline_state if k not in keys):
            raise ValueError('Frozen state changed during training')
        changed = sum(np.count_nonzero(final_state[k] != baseline_state[k]) for k in keys)
        if not changed:
            raise ValueError('Training did not change any selected weight')
        # Evaluate a conservative blend even if the full update regressed.
        delta_state = selected_state or {k:final_state[k] for k in keys}
        for blend in (.25,.5,.75,1.):
            blended = {**baseline_state,**{k:(baseline_state[k]+blend*(delta_state[k]-baseline_state[k])).astype(baseline_state[k].dtype) for k in keys}}
            restore(blended)
            summary = evaluation('validation')['summary']
            trial = dict(name=f'{name}_blend_{blend:g}',learning_rate=learning_rate,scope=scope,
                trainable_state_elements=sum(baseline_state[k].size for k in keys),
                actually_changed_final_elements=int(changed),blend=blend,validation=summary)
            report['weight_trials'].append(trial)
            if quality_key(summary) < quality_key(best_summary):
                best_state, best_summary, best_name = blended, summary, trial['name']
                best_scope = 'full_image_encoder_and_ctc' if scope == 'full_image_encoder_and_ctc' else 'bounded_image_encoder_tail_and_ctc'
        report.setdefault('training_history',[]).extend(history)
        (args.output/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        del optimizer

    # Freeze selected weights before examining external or manual performance.
    report['selected_candidate'] = best_name
    report['selected_validation'] = best_summary
    (args.output/'selection.json').write_text(json.dumps(dict(name=best_name,validation=best_summary,external_labels_used_for_selection=False),indent=2))
    summaries, details = {}, {}
    for label,state in [('baseline',baseline_state),('candidate',best_state)]:
        restore(state)
        evaluated = {name:evaluation(name) for name in datasets}
        summaries[label] = {name:value['summary'] for name,value in evaluated.items()}
        details[label] = {name:value['rows'] for name,value in evaluated.items()}
        print(json.dumps(dict(stage='guardrail_evaluation',model=label,summary=summaries[label])),flush=True)
    eligible, reason = promotion_decision(summaries['baseline'],summaries['candidate'])
    report.update(summary=summaries,details=details,promotion_eligible=eligible,promotion_reason=reason)
    if best_name != 'baseline':
        destination = args.output/'candidate.safetensors'
        save_file(best_state,str(destination))
        metadata = dict(parent_sha256=WEIGHTS_SHA256,initial_checkpoint_sha256=file_hash(checkpoint),
            dictionary_sha256=file_hash(args.cache/'en_dict.txt'),weights_sha256=file_hash(destination),
            training_scope=best_scope,frozen_state_unchanged=True,
            selected_candidate=best_name,seed=seed,manifest_sha256=file_hash(corpus_path),
            production_promoted=False,study_source_sha256=file_hash(Path(__file__)))
        destination.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
        restored = AwirosExperiment(args.cache,device='gpu',checkpoint=destination)
        restore(best_state)
        if not np.allclose(restored.probabilities(first),posterior(first),atol=1e-5,rtol=1e-4):
            raise ValueError('Selected checkpoint failed strict serving reload parity')
        report['saved_checkpoint_sha256'] = file_hash(destination)
        report['saved_checkpoint_reload_parity_passed'] = True
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(stage='complete',selected=best_name,promotion_eligible=eligible,reason=reason)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--external',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=Path('/models/experiments/awiros'))
    parser.add_argument('--epochs',type=int,default=2,choices=range(1,7))
    parser.add_argument('--extended',action='store_true',help='Stronger learning rates and full image encoder adaptation')
    parser.add_argument('--reuse-diagnosis',type=Path,help='Reuse a hash-bound diagnosis; evaluation crops stay out of training')
    run(parser.parse_args())
