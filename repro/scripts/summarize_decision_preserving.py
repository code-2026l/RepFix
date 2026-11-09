"""Summarise the pre-registered decision-preserving continuation records.

Reads dp_dev_<seed>_<gamma>.json, dp_conf_<seed>.json and dp_knee_<seed>.json
from the record directory and writes summary.json. No training and no rerun.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def records(folder, pattern):
    return [p for p in sorted(folder.glob(pattern)) if not p.name.endswith('.partial.json')]


def arms(path):
    d = json.loads(path.read_text(encoding='utf-8'))
    return d, {b['mode']: b for b in d['branches']}


def contrast(rows, a, b, key='test'):
    diffs = np.asarray([r[a][key]['own']['acc'] - r[b][key]['own']['acc'] for r in rows])
    n = len(diffs)
    mean, sd = float(diffs.mean()), float(diffs.std(ddof=1))
    half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n))
    return dict(mean=mean, sd=sd, ci95=[mean - half, mean + half],
                n=n, positive=int((diffs > 0).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    dev = {}
    for path in records(args.records, 'dp_dev_*.json'):
        d, b = arms(path)
        dev.setdefault(d['gamma'], []).append(dict(
            seed=d['seed'],
            detach=b['detach']['validation']['own']['acc'],
            joint=b['joint']['validation']['own']['acc'],
            penalty=b['penalty']['validation']['own']['acc'],
            disagreement=b['penalty']['validation']['disagreement'],
            joint_disagreement=b['joint']['validation']['disagreement']))
    frontier = []
    for gamma in sorted(dev):
        rows = dev[gamma]
        gain = np.mean([r['penalty'] - r['detach'] for r in rows]) * 100
        joint = np.mean([r['joint'] - r['detach'] for r in rows]) * 100
        frontier.append(dict(gamma=gamma, seeds=len(rows),
                             gain_pp=float(gain), joint_gain_pp=float(joint),
                             disagreement=float(np.mean([r['disagreement'] for r in rows])),
                             joint_disagreement=float(np.mean([r['joint_disagreement'] for r in rows]))))
    blocks = {}
    for name, pattern in [('primary_gamma_3', 'dp_conf_*.json'), ('knee_gamma_1', 'dp_knee_*.json')]:
        rows = []
        for path in records(args.records, pattern):
            d, b = arms(path)
            assert d['gamma'] == (3.0 if name == 'primary_gamma_3' else 1.0)
            rows.append(dict(seed=d['seed'], detach=b['detach'], joint=b['joint'], penalty=b['penalty']))
        if not rows:
            continue
        blocks[name] = dict(
            seeds=[r['seed'] for r in rows],
            penalty_minus_detach=contrast(rows, 'penalty', 'detach'),
            joint_minus_detach=contrast(rows, 'joint', 'detach'),
            penalty_disagreement=float(np.mean([r['penalty']['test']['disagreement'] for r in rows])),
            penalty_disagreement_sd=float(np.std([r['penalty']['test']['disagreement'] for r in rows], ddof=1)),
            joint_disagreement=float(np.mean([r['joint']['test']['disagreement'] for r in rows])),
            detach_disagreement=float(np.mean([r['detach']['test']['disagreement'] for r in rows])),
            certified=int(sum(1 for r in rows if r['penalty']['test']['consumer_audit']['certified'])),
            max_upper_bound=float(max(r['penalty']['test']['consumer_audit']['upper_remaining_disagreement'] for r in rows)),
            frozen_acc=float(np.mean([r['penalty']['test']['frozen']['acc'] for r in rows])),
            frozen_nll=float(np.mean([r['penalty']['test']['frozen']['nll'] for r in rows])),
            frozen_ece=float(np.mean([r['penalty']['test']['frozen']['ece'] for r in rows])),
            reference_acc=float(np.mean([r['penalty']['test']['reference']['acc'] for r in rows])),
            encoder_at_inference=1)
    summary = dict(
        protocol='predeclared decision-preserving continuation, single shared encoder',
        plan_sha256=json.loads(records(args.records, 'dp_conf_*.json')[0].read_text())['plan_sha256'],
        domain='radioml', dose=1.0, margin=0.0,
        selection_rule='smallest gamma with development disagreement <= 0.05, else the largest',
        gamma_grid=sorted(dev), selected_gamma=3.0,
        development_frontier=frontier,
        confirmation=blocks,
        note='The gamma=1.0 block is a post-hoc knee confirmation run after the fixed rule selected gamma=3.0 and the primary confirmation failed; it is reported separately and does not change the primary decision.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
