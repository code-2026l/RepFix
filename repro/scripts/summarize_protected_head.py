"""Summarise the pre-registered protected-head continuation records.

Reads ph_dev_<seed>.json and ph_conf_<seed>.json and writes one summary. No
training and no rerun; the fixed rule is applied exactly as pre-registered.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def records(folder, pattern):
    return [p for p in sorted(folder.glob(pattern)) if not p.name.endswith('.partial.json')]


def branches(path):
    d = json.loads(path.read_text(encoding='utf-8'))
    return d, {b['mode'] if b['mode'] != 'protected' else 'protected_%g' % b['lam']: b
               for b in d['branches']}, d['branches']


def contrast(rows, key, part='test'):
    diffs = np.asarray([r[key][part]['own']['acc'] - r['detach'][part]['own']['acc'] for r in rows])
    n = len(diffs)
    mean, sd = float(diffs.mean()), float(diffs.std(ddof=1))
    half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n))
    return dict(mean=mean, sd=sd, ci95=[mean - half, mean + half], n=n,
                positive=int((diffs > 0).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prefix', default='ph_')
    parser.add_argument('--domain', default='radioml')
    parser.add_argument('--dose', type=float, default=1.0)
    parser.add_argument('--selected-lambda', type=float, default=None)
    args = parser.parse_args()
    dev = {}
    for path in records(args.records, f'{args.prefix}dev_*.json'):
        d, b, raw = branches(path)
        assert len({x['minibatch_order_sha256'] for x in raw}) == 1
        for key in [k for k in b if k.startswith('protected_')]:
            lam = float(key.split('_')[1])
            dev.setdefault(lam, []).append(dict(
                seed=d['seed'],
                protected=b[key]['validation']['own']['acc'],
                detach=b['detach']['validation']['own']['acc'],
                joint=b['joint']['validation']['own']['acc'],
                disagreement=b[key]['validation']['disagreement'],
                fixed=b[key]['validation']['fixed_weight_disagreement'],
                protected_mse=b[key]['validation']['auxiliary_mse'],
                detach_mse=b['detach']['validation']['auxiliary_mse'],
                joint_mse=b['joint']['validation']['auxiliary_mse']))
    frontier = []
    for lam in sorted(dev):
        rows = dev[lam]
        frontier.append(dict(
            lam=lam, seeds=len(rows),
            gain_pp=float(np.mean([r['protected'] - r['detach'] for r in rows]) * 100),
            joint_gain_pp=float(np.mean([r['joint'] - r['detach'] for r in rows]) * 100),
            disagreement=float(np.mean([r['disagreement'] for r in rows])),
            disagreement_max=float(np.max([r['disagreement'] for r in rows])),
            fixed_weight_disagreement=float(np.mean([r['fixed'] for r in rows])),
            protected_auxiliary_mse_ratio=float(np.mean([r['protected_mse'] / r['detach_mse'] for r in rows])),
            joint_auxiliary_mse_ratio=float(np.mean([r['joint_mse'] / r['detach_mse'] for r in rows]))))
    selected = args.selected_lambda
    if selected is None:
        eligible = [r['lam'] for r in frontier if r['disagreement'] <= .05]
        selected = min(eligible) if eligible else max(r['lam'] for r in frontier)
    key = 'protected_%g' % selected
    rows = []
    for path in records(args.records, f'{args.prefix}conf_*.json'):
        d, b, raw = branches(path)
        assert len({x['minibatch_order_sha256'] for x in raw}) == 1
        rows.append(dict(seed=d['seed'], detach=b['detach'], joint=b['joint'], protected=b[key]))
    block = {}
    if rows:
        block['selected'] = dict(
            lam=selected, seeds=[r['seed'] for r in rows],
            protected_minus_detach=contrast(rows, 'protected'),
            joint_minus_detach=contrast(rows, 'joint'),
            protected_disagreement=float(np.mean([r['protected']['test']['disagreement'] for r in rows])),
            protected_disagreement_sd=float(np.std([r['protected']['test']['disagreement'] for r in rows], ddof=1)),
            fixed_weight_disagreement=float(np.mean([r['protected']['test']['fixed_weight_disagreement'] for r in rows])),
            joint_disagreement=float(np.mean([r['joint']['test']['disagreement'] for r in rows])),
            detach_disagreement=float(np.mean([r['detach']['test']['disagreement'] for r in rows])),
            certified=int(sum(1 for r in rows if r['protected']['test']['consumer_audit']['certified'])),
            max_upper_bound=float(max(r['protected']['test']['consumer_audit']['upper_remaining_disagreement'] for r in rows)),
            frozen_acc=float(np.mean([r['protected']['test']['frozen']['acc'] for r in rows])),
            detach_frozen_acc=float(np.mean([r['detach']['test']['frozen']['acc'] for r in rows])),
            joint_frozen_acc=float(np.mean([r['joint']['test']['frozen']['acc'] for r in rows])),
            reference_acc=float(np.mean([r['protected']['test']['reference']['acc'] for r in rows])),
            max_logit_error=float(max(r['protected']['test']['max_absolute_healthy_logit_error'] for r in rows)),
            protected_auxiliary_mse_ratio=float(np.mean([r['protected']['test']['auxiliary_mse'] / r['detach']['test']['auxiliary_mse'] for r in rows])),
            joint_auxiliary_mse_ratio=float(np.mean([r['joint']['test']['auxiliary_mse'] / r['detach']['test']['auxiliary_mse'] for r in rows])),
            encoder_at_inference=1, forward_passes_at_inference=1)
    first = records(args.records, f'{args.prefix}conf_*.json') or records(args.records, f'{args.prefix}dev_*.json')
    summary = dict(
        protocol='predeclared protected-head continuation, single encoder one pass',
        plan_sha256=json.loads(first[0].read_text())['plan_sha256'],
        domain=args.domain, dose=args.dose,
        selection_rule='smallest lambda with development disagreement <= 0.05, else the largest',
        lambda_grid=sorted(dev), selected_lambda=selected,
        development_frontier=frontier, confirmation=block,
        note='Protected-head disagreement is measured against the healthy consumer on the same '
             'examples; fixed_weight_disagreement applies the original consumer weights to the new feature.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
