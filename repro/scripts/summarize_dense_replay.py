"""Fresh-seed summary of the dense low-rank replay confirmation (SNR)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from select_joint_alignment import interval

DIMS = dict(har=(561, 64, 6), radioml=(256, 64, 11))
ENCODERS = dict(har=44096, radioml=24576)


def main(a):
    choice = json.loads(a.choice.read_text())
    paths = sorted(a.records.glob('c???.json'))
    rows = [json.loads(p.read_text()) for p in paths]
    assert len(rows) == 10 and all(r['complete'] for r in rows)
    assert len({r['domain'] for r in rows}) == 1
    domain = rows[0]['domain']
    selected = choice['domains'][domain]['selected']
    assert selected['utility_compatible']
    key = (selected['rank1'], selected['rank2'])
    choice_sha = hashlib.sha256(a.choice.read_bytes()).hexdigest()
    assert all(r['selection_sha256'] == choice_sha for r in rows)
    branches = {}
    for r in rows:
        det = next(b for b in r['branches'] if (b['rank1'], b['rank2']) == key and b['mode'] == 'detach')
        joi = next(b for b in r['branches'] if (b['rank1'], b['rank2']) == key and b['mode'] == 'joint')
        assert det['minibatch_order_sha256'] == joi['minibatch_order_sha256'] == \
               r['full_controls']['detach']['minibatch_order_sha256'] == r['full_controls']['joint']['minibatch_order_sha256']
        branches[r['seed']] = (det, joi, r)
    seeds = sorted(branches)
    det = [branches[s][0]['test'] for s in seeds]
    joi = [branches[s][1]['test'] for s in seeds]
    full = [branches[s][2]['full_controls']['joint']['test'] for s in seeds]
    gain = interval([j['own']['acc'] - d['own']['acc'] for j, d in zip(joi, det)])
    versus_full = interval([j['own']['acc'] - f['own']['acc'] for j, f in zip(joi, full)])
    mse = interval([j['auxiliary_mse'] / d['auxiliary_mse'] for j, d in zip(joi, det)])
    all_rows = [b['test'] for pair in branches.values() for b in pair[:2]]
    raw_rows = [b for pair in branches.values() for b in pair[:2]]
    disagreement = {c: max(b['contracts'][c]['disagreement'] for b in all_rows) for c in all_rows[0]['contracts']}
    certified = all(b['contracts'][c]['certificate']['certified'] for b in all_rows for c in b['contracts'])
    upper = max(b['contracts'][c]['certificate']['upper_remaining_disagreement'] for b in all_rows for c in b['contracts'])
    d, m, nc = DIMS[domain]
    macs = selected['extra_macs']
    out = dict(plan_sha256=rows[0]['plan_sha256'], domain=domain, seeds=seeds,
               configuration=dict(rank1=key[0], rank2=key[1], extra_macs=macs,
                                  encoder_fraction=macs / ENCODERS[domain],
                                  analytic_note='s*d + m*s + m*m + 2*r*m + 2*n_c*m + 2*m'),
               primary_gain_vs_detached=gain, primary_difference_to_full_joint=versus_full,
               auxiliary_mse_ratio=mse,
               detached_accuracy=interval([x['own']['acc'] for x in det]),
               joint_accuracy=interval([x['own']['acc'] for x in joi]),
               full_joint_accuracy=interval([x['own']['acc'] for x in full]),
               maximum_consumer_disagreement=disagreement,
               every_seed_every_consumer_within_one_point=all(v <= .01 for v in disagreement.values()),
               every_certificate_meets_five_point_budget=bool(certified),
               maximum_certificate_upper_bound=float(upper),
               maximum_packed_legacy_logit_error=max(b['packed_serving']['maximum_legacy_logit_error'] for b in all_rows),
               maximum_full_replay_logit_error=max(b['full_reference_maximum_logit_error'] for b in all_rows),
               total_current_merge_prediction_changes=sum(b['current_merge_prediction_changes'] for b in all_rows),
               every_seed_first_and_second_updates_nonzero=all(b['first_delta_norm'] > 0 and b['second_delta_norm'] > 0
                                                               for b in raw_rows),
               source_record_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               utility_gate=dict(positive_gain=gain['ci95'][0] > 0,
                                 noninferiority_to_full_joint=versus_full['mean'] >= -.005))
    out['utility_gate']['passed'] = all(out['utility_gate'].values())
    a.out.write_text(json.dumps(out, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'source_record_sha256'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--choice', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
