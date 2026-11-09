"""Small, testable safeguards for a reproducible OCR weight experiment."""
from collections import Counter


def character_errors(expected, observed):
    """Levenshtein alignment with explicit missing/extra/substituted counts."""
    cells = [[None] * (len(observed) + 1) for _ in range(len(expected) + 1)]
    cells[0][0] = (0, 0, 0, 0)
    for i in range(len(expected) + 1):
        for j in range(len(observed) + 1):
            if i == j == 0:
                continue
            choices = []
            if i and j:
                cost, deleted, inserted, substituted = cells[i-1][j-1]
                wrong = expected[i-1] != observed[j-1]
                choices.append((cost + wrong, deleted, inserted, substituted + wrong))
            if i:
                cost, deleted, inserted, substituted = cells[i-1][j]
                choices.append((cost + 1, deleted + 1, inserted, substituted))
            if j:
                cost, deleted, inserted, substituted = cells[i][j-1]
                choices.append((cost + 1, deleted, inserted + 1, substituted))
            cells[i][j] = min(choices)
    cost, deleted, inserted, substituted = cells[-1][-1]
    return dict(total=cost, missing=deleted, extra=inserted, substituted=substituted)


def metrics(results):
    totals = Counter()
    for row in results:
        totals.update(character_errors(row['expected'], row['reading']))
    characters = sum(len(row['expected']) for row in results)
    return dict(samples=len(results), exact=sum(row['expected'] == row['reading'] for row in results),
                character_errors=totals['total'], missing=totals['missing'], extra=totals['extra'],
                substituted=totals['substituted'], characters=characters,
                cer=totals['total'] / characters if characters else None)


def quality_key(summary):
    return -summary['exact'], summary['character_errors']


def disjoint_training(training, evaluation):
    """Refuse registration or exact-file overlap across fit and evaluation."""
    registrations = {row['transcription'] for row in training}
    hashes = {row['crop_sha256'] for row in training}
    if any(row['transcription'] in registrations or row['crop_sha256'] in hashes for row in evaluation):
        raise ValueError('Training/evaluation overlap')


def promotion_decision(baseline, candidate):
    """Require validation improvement and no regression on either guardrail set."""
    if quality_key(candidate['validation']) >= quality_key(baseline['validation']):
        return False, 'No validation improvement'
    for name in ('manual', 'external', 'corpus_test'):
        if (candidate[name]['exact'] < baseline[name]['exact']
                or candidate[name]['character_errors'] > baseline[name]['character_errors']):
            return False, f'Regression on {name}'
    return True, 'Validation improved without aggregate guardrail regression'
