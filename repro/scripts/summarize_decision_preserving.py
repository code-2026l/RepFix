"""Summarise the pre-registered decision-preserving continuation records.

Reads dev_<seed>_<gamma>.json, conf_<seed>.json and knee_<seed>.json from the
record directory and writes summary.json. No training and no rerun.

The defaults are the SNR (RadioML) records; ``--prefix hp_ --domain har
--dose 10 --selected-gamma 0.1`` summarises the second corpus. The arm defaults
to the margin hinge; ``--arm anchor --prefix an_ --selected-gamma 10`` summarises
the logit anchor, whose grid lives in the ``anchor_gamma`` field.
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
    parser.add_argument('--prefix', default='dp_', help='record filename prefix before dev/conf/knee')
    parser.add_argument('--domain', default='radioml')
    parser.add_argument('--dose', type=float, default=1.0)
    parser.add_argument('--selected-gamma', type=float, default=3.0)
    parser.add_argument('--knee-gamma', type=float, default=1.0)
    parser.add_argument('--arm', default='penalty', choices=['penalty', 'anchor'])
    args = parser.parse_args()
    key = 'anchor_gamma' if args.arm == 'anchor' else 'gamma'
    dev = {}
    for path in records(args.records, f'{args.prefix}dev_*.json'):
        d, b = arms(path)
        dev.setdefault(d[key], []).append(dict(
            seed=d['seed'],
            detach=b['detach']['validation']['own']['acc'],
            joint=b['joint']['validation']['own']['acc'],
            arm=b[args.arm]['validation']['own']['acc'],
            disagreement=b[args.arm]['validation']['disagreement'],
            joint_disagreement=b['joint']['validation']['disagreement'],
            detach_mse=b['detach']['validation']['auxiliary_mse'],
            joint_mse=b['joint']['validation']['auxiliary_mse'],
            arm_mse=b[args.arm]['validation']['auxiliary_mse']))
    frontier = []
    for gamma in sorted(dev):
        rows = dev[gamma]
        gain = np.mean([r['arm'] - r['detach'] for r in rows]) * 100
        joint = np.mean([r['joint'] - r['detach'] for r in rows]) * 100
        frontier.append(dict(gamma=gamma, seeds=len(rows),
                             gain_pp=float(gain), joint_gain_pp=float(joint),
                             disagreement=float(np.mean([r['disagreement'] for r in rows])),
                             disagreement_max=float(np.max([r['disagreement'] for r in rows])),
                             joint_disagreement=float(np.mean([r['joint_disagreement'] for r in rows])),
                             **{f'{args.arm}_auxiliary_mse_ratio': float(np.mean([r['arm_mse'] / r['detach_mse'] for r in rows])),
               'joint_auxiliary_mse_ratio': float(np.mean([r['joint_mse'] / r['detach_mse'] for r in rows]))}))
    blocks = {}
    patterns = [(f'primary_gamma_{args.selected_gamma:g}', f'{args.prefix}conf_*.json', args.selected_gamma)]
    if args.arm == 'penalty':
        patterns.append((f'knee_gamma_{args.knee_gamma:g}', f'{args.prefix}knee_*.json', args.knee_gamma))
    for name, pattern, gamma in patterns:
        rows = []
        for path in records(args.records, pattern):
            d, b = arms(path)
            assert d[key] == gamma
            rows.append(dict(seed=d['seed'], detach=b['detach'], joint=b['joint'], arm=b[args.arm]))
        if not rows:
            continue
        arm = args.arm
        blocks[name] = dict(
            seeds=[r['seed'] for r in rows],
            **{f'{arm}_minus_detach': contrast(rows, 'arm', 'detach'),
               f'{arm}_minus_joint': contrast(rows, 'arm', 'joint'),
               f'{arm}_disagreement': float(np.mean([r['arm']['test']['disagreement'] for r in rows])),
               f'{arm}_disagreement_sd': float(np.std([r['arm']['test']['disagreement'] for r in rows], ddof=1)),
               f'{arm}_auxiliary_mse_ratio': float(np.mean([r['arm']['test']['auxiliary_mse'] / r['detach']['test']['auxiliary_mse'] for r in rows])),
               'joint_minus_detach': contrast(rows, 'joint', 'detach'),
               'joint_disagreement': float(np.mean([r['joint']['test']['disagreement'] for r in rows])),
               'detach_disagreement': float(np.mean([r['detach']['test']['disagreement'] for r in rows])),
               'certified': int(sum(1 for r in rows if r['arm']['test']['consumer_audit']['certified'])),
               'max_upper_bound': float(max(r['arm']['test']['consumer_audit']['upper_remaining_disagreement'] for r in rows)),
               'frozen_acc': float(np.mean([r['arm']['test']['frozen']['acc'] for r in rows])),
               'detach_frozen_acc': float(np.mean([r['detach']['test']['frozen']['acc'] for r in rows])),
               'joint_frozen_acc': float(np.mean([r['joint']['test']['frozen']['acc'] for r in rows])),
               'frozen_nll': float(np.mean([r['arm']['test']['frozen']['nll'] for r in rows])),
               'frozen_ece': float(np.mean([r['arm']['test']['frozen']['ece'] for r in rows])),
               'reference_acc': float(np.mean([r['arm']['test']['reference']['acc'] for r in rows])),
               'joint_auxiliary_mse_ratio': float(np.mean([r['joint']['test']['auxiliary_mse'] / r['detach']['test']['auxiliary_mse'] for r in rows])),
               'encoder_at_inference': 1})
    first = records(args.records, f'{args.prefix}conf_*.json') or records(args.records, f'{args.prefix}dev_*.json')
    summary = dict(
        protocol='predeclared decision-preserving continuation, single shared encoder',
        plan_sha256=json.loads(first[0].read_text())['plan_sha256'],
        domain=args.domain, dose=args.dose, margin=0.0, arm=args.arm,
        selection_rule='smallest gamma with development disagreement <= 0.05, else the largest',
        gamma_grid=sorted(dev), selected_gamma=args.selected_gamma,
        development_frontier=frontier,
        confirmation=blocks,
        note=('The gamma=1.0 block is a post-hoc knee confirmation run after the fixed rule selected gamma=3.0 and the primary confirmation failed; it is reported separately and does not change the primary decision.'
              if args.arm == 'penalty' else
              'Exploratory origin: the anchor was designed after the margin-hinge and HAR outcomes and pre-registered before either anchor result; the grid, rule and seeds were not retuned.'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
