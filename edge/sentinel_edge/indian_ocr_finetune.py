"""Bounded recognition-head adaptation on visually reviewed licensed stills.

Not a live worker. No simulated temporal evidence, label-driven crops, or test
labels in checkpoint selection. Original Awiros weights remain unchanged.
"""
from __future__ import annotations
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from .awiros_experiment import AwirosExperiment, WEIGHTS_SHA256, file_hash, input_array
from .benchmark_viewer import render_viewer, validate_viewer_scope
from .ctc_fusion import CTCPlateReader, greedy_text


def validate_samples(document, root):
    """Reject unreviewed data, cross-split registrations and duplicate images."""
    rows = document['samples']
    if not 8 <= len(rows) <= 100 or document['dataset'].get('license') != 'CC-BY-4.0':
        raise ValueError('A bounded licensed reviewed pilot is required')
    groups, hashes = {}, set()
    for row in rows:
        if row.get('review') != 'visually_verified' or row.get('split') not in {'train', 'validation', 'test'}:
            raise ValueError('Unreviewed sample or invalid split')
        path = (root / row['crop_path']).resolve()
        original = (root / row['path']).resolve()
        if not path.is_relative_to(root.resolve()) or not original.is_relative_to(root.resolve()):
            raise ValueError('Sample path escapes dataset')
        if file_hash(path) != row['crop_sha256'] or file_hash(original) != row['sha256']:
            raise ValueError('Reviewed image was modified')
        label = row['transcription']
        if not 4 <= len(label) <= 25 or not all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' for c in label):
            raise ValueError('Invalid reviewed transcription')
        if label in groups and groups[label] != row['split']:
            raise ValueError('Registration leakage across splits')
        if row['sha256'] in hashes or row['crop_sha256'] in hashes:
            raise ValueError('Duplicate original or crop')
        groups[label] = row['split']
        hashes.update((row['sha256'], row['crop_sha256']))
    if any(sum(row['split'] == split for row in rows) < 2 for split in ('train', 'validation', 'test')):
        raise ValueError('Each split requires distinct reviewed vehicles')
    return rows


def crop_proposals(root):
    import cv2
    from .ocr_selection_benchmark import PlateDetectorExperiment
    proposals = json.loads((root / 'review_pending.json').read_text())
    original = json.loads((root / 'verified_sample.json').read_text())
    by_path = {row['path']: row for row in proposals}
    for row in original['verified_samples']:
        by_path[row['path']] = row
    detector = PlateDetectorExperiment()
    rows = []
    (root / 'crops').mkdir(exist_ok=True)
    for row in by_path.values():
        path = root / row['path']
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Cannot decode dataset image')
        box = detector.box(image)
        if box is None:
            print(json.dumps({'no_detection': path.name}), flush=True)
            continue
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2].copy()
        destination = root / 'crops' / (path.stem + '.png')
        if not cv2.imwrite(str(destination), crop):
            raise ValueError('Cannot save licensed native crop')
        rows.append({'path': row['path'], 'transcription': row['transcription'],
                     'sha256': file_hash(path), 'crop_path': str(destination.relative_to(root)),
                     'crop_sha256': file_hash(destination), 'box': list(box),
                     'review': 'pending', 'width': crop.shape[1], 'height': crop.shape[0]})
    (root / 'crop_proposals.json').write_text(json.dumps({'dataset': original['dataset'], 'samples': rows}, indent=2))
    print(json.dumps({'crop_proposals': len(rows)}), flush=True)


def augment(image, index, rng):
    import cv2
    import numpy as np
    if index == 0:
        return image.copy()
    result = image.copy()
    if index == 1:
        result = cv2.GaussianBlur(result, (3, 3), .6)
    elif index == 2:
        h, w = result.shape[:2]
        result = cv2.resize(result, (max(12, round(w * .8)), max(8, round(h * .8))))
    else:
        noise = rng.normal(0, 2, result.shape)
        result = np.clip(result.astype(float) * .85 + noise, 0, 255).astype(np.uint8)
    return result


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def train(manifest, cache, output, report, viewer, epochs=60):
    import cv2
    import numpy as np
    from safetensors.numpy import save_file
    document = json.loads(manifest.read_text())
    root = manifest.parent.resolve()
    rows = validate_samples(document, root)
    validate_viewer_scope(viewer, document['dataset'])
    validate_viewer_scope(report.with_suffix('.html'), document['dataset'])
    if not 1 <= epochs <= 100:
        raise ValueError('Pilot epoch bound exceeded')
    reader = AwirosExperiment(cache, device='cpu')
    paddle, model = reader.paddle, reader.model
    random.seed(20260927)
    rng = np.random.default_rng(20260927)
    paddle.seed(20260927)
    state = {key: value.numpy().copy() for key, value in model.state_dict().items()}
    trained_keys = {key for key in state if key.startswith('head.ctc_head.fc.')}
    if trained_keys != {'head.ctc_head.fc.weight', 'head.ctc_head.fc.bias'}:
        raise ValueError('Unexpected trainable head contract')
    for name, parameter in model.named_parameters():
        parameter.stop_gradient = name not in trained_keys
    # eval() keeps every batch norm buffer and dropout frozen. Train ONLY the
    # CTC head during optimisation; the auxiliary NRTR branch is never trained.
    model.eval()
    head = model.head.ctc_head
    prepared = []
    for i, row in enumerate(rows):
        image = cv2.imread(str(root / row['crop_path']))
        count = 4 if row['split'] == 'train' else 1
        for index in range(count):
            view = augment(image, index, rng)
            with paddle.no_grad():
                features = model.head.ctc_encoder(model.backbone(paddle.to_tensor(input_array(view))))
            prepared.append({'row': row, 'image': image if index == 0 else None,
                             'features': features.numpy(), 'augmentation': index})
        print(json.dumps({'encoded_sample': i + 1, 'total': len(rows), 'split': row['split']}), flush=True)
    lookup = {token: i for i, token in enumerate(reader.tokens)}
    loss_function = paddle.nn.CTCLoss(blank=0, reduction='none')

    def objective(batch):
        features = paddle.to_tensor(np.concatenate([value['features'] for value in batch]))
        labels = [[lookup[c] for c in value['row']['transcription']] for value in batch]
        lengths = [len(value) for value in labels]
        padded = np.zeros((len(batch), max(lengths)), dtype=np.int32)
        for index, label in enumerate(labels):
            padded[index, :len(label)] = label
        # Paddle CTCLoss includes softmax internally: raw head logits, NOT
        # probabilities or PyTorch-style log-softmax. Mirrors pinned OCR loss.
        logits = head.fc(features)
        loss = loss_function(logits.transpose((1, 0, 2)), paddle.to_tensor(padded),
                             paddle.to_tensor([logits.shape[1]] * len(batch), dtype='int64', place=paddle.CPUPlace()),
                             paddle.to_tensor(lengths, dtype='int64')).mean()
        if not np.isfinite(loss.numpy()).all():
            raise ValueError('Nonfinite CTC loss')
        return loss

    def evaluate(split):
        selected = [value for value in prepared if value['row']['split'] == split and value['augmentation'] == 0]
        exact = errors = characters = 0
        diagnostics = []
        with paddle.no_grad():
            loss = float(objective(selected).numpy())
            for value in selected:
                probs = paddle.nn.functional.softmax(head.fc(paddle.to_tensor(value['features'])), axis=2).numpy()[0]
                text = greedy_text(probs.tolist(), reader.tokens)
                expected = value['row']['transcription']
                exact += text == expected
                errors += edit_distance(text, expected)
                characters += len(expected)
                diagnostics.append((value, probs.tolist(), text))
        return {'samples': len(selected), 'exact': exact, 'character_errors': errors,
                'characters': characters, 'cer': errors / characters, 'ctc_loss': loss}, diagnostics

    baseline_val, _ = evaluate('validation')
    # Test metrics are NOT queried during training or checkpoint selection.
    baseline_head = {key: value.copy() for key, value in state.items() if key in trained_keys}
    best, best_head, best_epoch = baseline_val, baseline_head, 0
    training = [value for value in prepared if value['row']['split'] == 'train']
    optimizer = paddle.optimizer.Adam(learning_rate=.0002, parameters=list(head.parameters()),
                                       grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0))
    history, stale = [], 0
    for epoch in range(1, epochs + 1):
        random.shuffle(training)
        losses = []
        for offset in range(0, len(training), 8):
            loss = objective(training[offset:offset + 8])
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            losses.append(float(loss.numpy()))
        validation, _ = evaluate('validation')
        history.append({'epoch': epoch, 'train_ctc_loss': sum(losses) / len(losses), 'validation': validation})
        key = lambda value: (-value['exact'], value['cer'], value['ctc_loss'])
        if key(validation) < key(best):
            best, best_epoch, stale = validation, epoch, 0
            best_head = {name: tensor.numpy().copy() for name, tensor in model.state_dict().items() if name in trained_keys}
        else:
            stale += 1
        print(json.dumps(history[-1]), flush=True)
        if stale >= 12:
            break
    # Save the actual last trained model too, even if validation keeps epoch 0.
    output.parent.mkdir(parents=True, exist_ok=True)
    final_state = {key: value.numpy().copy() for key, value in model.state_dict().items()}
    if any(not np.array_equal(state[key], final_state[key]) for key in state if key not in trained_keys):
        raise ValueError('Frozen encoder or buffer changed')
    save_file(final_state, str(output.with_name(output.stem + '-last.safetensors')))
    # When epoch 0 is best, benchmark the trained last checkpoint explicitly;
    # report its rejection, never call the original weights newly fine-tuned.
    selected_epoch = best_epoch or history[-1]['epoch']
    selected = dict(state)
    selected.update(best_head if best_epoch else {key: final_state[key] for key in trained_keys})
    if not any(not np.array_equal(selected[key], state[key]) for key in trained_keys):
        raise ValueError('No head learning occurred')
    save_file(selected, str(output))
    model.set_state_dict({key: paddle.to_tensor(value) for key, value in selected.items()})
    tuned_test, tuned_diagnostics = evaluate('test')
    tuned_val, _ = evaluate('validation')
    model.set_state_dict({key: paddle.to_tensor(value) for key, value in state.items()})
    baseline_test, baseline_diagnostics = evaluate('test')
    for metrics, diagnostics in ((baseline_test, baseline_diagnostics), (tuned_test, tuned_diagnostics)):
        recalls = {1: 0, 10: 0, 100: 0}
        for value, probs, _ in diagnostics:
            decoder = CTCPlateReader(reader.tokens)
            decoder.add(probs, timestamp_ms=0, source_id='single-independent-still')
            candidates = decoder.candidates()
            rank = next((i for i, candidate in enumerate(candidates, 1)
                         if candidate.plate == value['row']['transcription']), None)
            for size in recalls:
                recalls[size] += rank is not None and rank <= size
        metrics['candidate_recall_counts'] = {str(k): v for k, v in recalls.items()}
    metadata = {'generated_at': datetime.now(timezone.utc).isoformat(),
                'parent_sha256': WEIGHTS_SHA256, 'dictionary_sha256': file_hash(cache / 'en_dict.txt'),
                'weights_sha256': file_hash(output), 'training_scope': 'ctc_linear_head_only',
                'frozen_state_unchanged': True, 'trainable_parameters': sum(state[key].size for key in trained_keys),
                'manifest_sha256': file_hash(manifest), 'seed': 20260927, 'device': 'cpu',
                'learning_rate': .0002, 'batch_size': 8, 'training_augmentations_per_vehicle': 4,
                'selected_epoch': selected_epoch, 'best_validation_epoch': best_epoch,
                'validation_preferred_adaptation': best_epoch > 0, 'history': history,
                'baseline_validation': baseline_val, 'tuned_validation': tuned_val,
                'baseline_test': baseline_test, 'tuned_test': tuned_test,
                'split_counts': {split: sum(row['split'] == split for row in rows) for split in ('train', 'validation', 'test')},
                'prior_pretraining_overlap_unknown': True, 'production_promoted': False,
                'dataset_license': document['dataset']['license'], 'still_images_not_temporal': True}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(metadata, indent=2))
    visuals = []
    for index, pair in enumerate(zip(baseline_diagnostics, tuned_diagnostics), 1):
        for name, (value, probs, text) in zip(('Original Awiros', 'Fine-tuned Awiros'), pair):
            decoder = CTCPlateReader(reader.tokens)
            decoder.add(probs, timestamp_ms=0, source_id='single-independent-still')
            candidates = decoder.candidates()
            expected = value['row']['transcription']
            visuals.append({'source_mode': 'benchmark', 'diagnostic_only': True,
                            'title': f'Test vehicle {index} · {name}',
                            'original': value['image'], 'candidates': candidates, 'accepted': None,
                            'expected': expected, 'outcome': 'raw-exact' if text == expected else 'raw-error',
                            'reason': 'Single still: no temporal acceptance claim',
                            'frame_diagnostics': [{'image': value['image'], 'index': 0, 'pts_ms': None,
                                                   'reading': text, 'ocr_ms': 'cached-feature diagnostic'}],
                            'score_note': 'Held out from this adaptation only. Prior publisher training overlap unknown. '
                                          'Raw exact-match result is not a calibrated confidence or accepted live alert.'})
    render_viewer(viewer, document['dataset'], visuals)
    # Strict full-checkpoint reload and inference parity, not just head scoring.
    tuned = AwirosExperiment(cache, checkpoint=output)
    for value, expected_probs, _ in tuned_diagnostics:
        actual = tuned.probabilities(value['image'])
        if not np.allclose(actual, expected_probs, atol=1e-5):
            raise ValueError('Reloaded checkpoint does not match cached-feature evaluation')
    print(json.dumps({'baseline_test': baseline_test, 'tuned_test': tuned_test,
                      'selected_epoch': selected_epoch, 'reload_parity_passed': True}), flush=True)
    return metadata


def evaluate_stills(manifest, cache, output, report, viewer):
    """Independent full-model before/after test rerun; never selects weights."""
    import cv2
    document = json.loads(manifest.read_text())
    root = manifest.parent.resolve()
    rows = [row for row in validate_samples(document, root) if row['split'] == 'test']
    validate_viewer_scope(report.with_suffix('.html'), document['dataset'])
    visuals, summary = [], {}
    for name, checkpoint in (('Original Awiros', None), ('Fine-tuned Awiros', output)):
        reader = AwirosExperiment(cache, checkpoint=checkpoint)
        exact = errors = characters = 0
        recalls = {1: 0, 10: 0, 100: 0}
        for index, row in enumerate(rows, 1):
            image = cv2.imread(str(root / row['crop_path']))
            probabilities = reader.probabilities(image)
            reading = greedy_text(probabilities, reader.tokens)
            expected = row['transcription']
            exact += reading == expected
            errors += edit_distance(reading, expected)
            characters += len(expected)
            decoder = CTCPlateReader(reader.tokens)
            decoder.add(probabilities, timestamp_ms=0, source_id=row['sha256'])
            candidates = decoder.candidates()
            rank = next((i for i, candidate in enumerate(candidates, 1) if candidate.plate == expected), None)
            for size in recalls:
                recalls[size] += rank is not None and rank <= size
            visuals.append({'source_mode': 'benchmark', 'diagnostic_only': True,
                            'title': f'Test vehicle {index} · {name}', 'original': image,
                            'candidates': candidates, 'accepted': None, 'expected': expected,
                            'outcome': 'raw-exact' if reading == expected else 'raw-error',
                            'reason': 'Single still: no temporal acceptance claim',
                            'frame_diagnostics': [{'image': image, 'index': 0, 'pts_ms': None,
                                                   'reading': reading, 'ocr_ms': 'not profiled'}],
                            'score_note': 'Different registrations from adaptation training and validation. '
                                          'Prior publisher pretraining overlap unknown; native pixels unchanged. '
                                          'Raw recognition is NOT a calibrated confidence or accepted live alert.'})
        summary[name] = {'samples': len(rows), 'exact': exact, 'character_errors': errors,
                         'characters': characters, 'cer': errors / characters,
                         'candidate_recall_counts': {str(k): v for k, v in recalls.items()}}
        del reader
    # Adjacent before/after cards for each test vehicle.
    count = len(rows)
    visuals = [value for index in range(count) for value in (visuals[index], visuals[index + count])]
    result = {'evaluation': 'Single-still diagnostic, not multi-frame acceptance',
              'models': summary, 'manifest_sha256': file_hash(manifest),
              'parent_sha256': WEIGHTS_SHA256, 'checkpoint_sha256': file_hash(output),
              'pretraining_overlap_unknown': True, 'production_promoted': False}
    # Migrate the initial pilot's overly broad metadata label. Preference on
    # validation loss alone is NOT a recommendation to deploy this checkpoint.
    metadata_path = output.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text())
    if 'validation_recommends_promotion' in metadata:
        metadata['validation_preferred_adaptation'] = metadata.pop('validation_recommends_promotion')
        metadata_path.write_text(json.dumps(metadata, indent=2))
    report.write_text(json.dumps(result, indent=2))
    render_viewer(viewer, document['dataset'], visuals)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-crops', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--viewer', type=Path)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_crops:
        crop_proposals(args.prepare_crops)
    elif all((args.manifest, args.cache, args.output, args.report, args.viewer)):
        if args.evaluate_only:
            evaluate_stills(args.manifest, args.cache, args.output, args.report, args.viewer)
        else:
            train(args.manifest, args.cache, args.output, args.report, args.viewer, args.epochs)
    else:
        parser.error('Provide crop preparation root or all training/output paths')
