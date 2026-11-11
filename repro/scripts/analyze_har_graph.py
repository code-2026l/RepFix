"""Select and independently replay the matched HAR graph-target experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_har_transfer import ROOT, RESULTS, digest, interval, save

PLAN = ROOT / 'docs/plans/2026-11-11-har-graph-targets.md'
CHOICE = RESULTS / 'har_graph_choice_20261111.json'
CONFIRMATION = RESULTS / 'har_graph_confirmation_20261111.json'


def load(stage, records, arrays):
    confirm = stage == 'confirmation'
    prefix, start, count = ('g', 8400, 16) if confirm else ('k', 8300, 8)
    for offset in range(count):
        name = f'{prefix}{offset:03d}'
        row = json.loads((records / f'{name}.json').read_text())
        assert row['complete'] and row['device'] == 'cpu' and row['stage'] == stage
        assert row['seed'] == start + offset and row['plan_sha256'] == digest(PLAN)
        assert row['test_loaded'] == confirm
        assert row['choice_sha256'] == (digest(CHOICE) if confirm else None)
        groups = [set(row[key]) for key in ('fit_subjects', 'monitor_subjects', 'evaluation_subjects')]
        assert not any(a & b for i, a in enumerate(groups) for b in groups[i+1:])
        path = arrays / f'{name}_states/replay.pt'
        assert digest(path) == row['replay_sha256']
        replay = torch.load(path, map_location='cpu', weights_only=True)
        labels = replay['labels']
        hits = {key: logits.argmax(1) == labels for key, logits in replay['outputs'].items()}
        assert {key: val.double().mean().item() for key, val in hits.items()} == row['scores']
        assert all(torch.equal(replay['old'][head], replay['retained'][head]) for head in replay['old'])
        yield row, hits, replay['subjects']


def select(rows):
    candidates = []
    for dose in (.1, .3, 1.):
        means = {key: float(np.mean([100 * (row['scores'][f'graph/{dose}'] - row['scores'][key])
                                   for row, _, _ in rows]))
                 for key in ('control/0.0', f'warm/{dose}')}
        candidates.append(dict(dose=dose, gain_pp=means, qualifies=min(means.values()) >= .5))
    eligible = [row for row in candidates if row['qualifies']]
    selected = max(eligible, key=lambda row: min(row['gain_pp'].values())) if eligible else None
    return dict(passed=bool(eligible), dose=selected['dose'] if selected else None,
                plan_sha256=digest(PLAN), candidates=candidates)


def confirm(rows):
    dose = json.loads(CHOICE.read_text())['dose']
    methods = (f'graph/{dose}', 'control/0.0', f'warm/{dose}')
    scores = {key: interval([100 * row['scores'][key] for row, _, _ in rows]) for key in methods}
    gains = {key: interval([100 * (row['scores'][methods[0]] - row['scores'][key])
                            for row, _, _ in rows]) for key in methods[1:]}
    subject_keys = rows[0][0]['evaluation_subjects']
    subject_gains = {}
    for key in subject_keys:
        subject_gains[str(key)] = {method: float(np.mean([
            100 * (hits[methods[0]][subjects == key].double().mean() -
                   hits[method][subjects == key].double().mean()) for _, hits, subjects in rows]))
            for method in methods[1:]}
    return dict(passed=all(gain['ci95'][0] > 0 for gain in gains.values()), dose=dose,
                plan_sha256=digest(PLAN), choice_sha256=digest(CHOICE), accuracy_pp=scores,
                gains_pp=gains, subject_gains_pp=subject_gains, exact_old_readers=2 * len(rows),
                graph_setup_seconds=interval([row['graph_setup_seconds'] for row, _, _ in rows]),
                labeled_examples=[row['labeled_examples'] for row, _, _ in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('screen', 'confirmation'))
    parser.add_argument('--records', type=Path)
    parser.add_argument('--arrays', type=Path)
    args = parser.parse_args()
    arrays = args.arrays or ROOT / f'tmp/conditional_contrastivehargraph{args.stage}'
    torch.set_num_threads(4)
    rows = list(load(args.stage, args.records or arrays, arrays))
    if args.stage == 'screen':
        result, out = select(rows), CHOICE
    else:
        result, out = confirm(rows), CONFIRMATION
    save(out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
