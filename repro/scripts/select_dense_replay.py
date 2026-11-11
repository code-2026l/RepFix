"""Frozen development selection for the dense low-rank replay study."""
import argparse
import hashlib
import json
from pathlib import Path

from select_joint_alignment import interval
from summarize_replay_rank import dense_macs

DIMS = dict(har=(561, 64, 6), radioml=(256, 64, 11))
ENCODERS = dict(har=44096, radioml=24576)


def main(a):
    paths = sorted(a.records.glob('d???.json'))
    records = [json.loads(p.read_text()) for p in paths]
    assert len(records) == 6 and all(r['complete'] and r['phase'] == 'development' for r in records)
    out = dict(plan_sha256=records[0]['plan_sha256'], choices_frozen_before_confirmation=True,
               ranking_uses_accuracy=False, cost_acceptance_pending=True,
               source_record_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               domains={})
    for domain in ['har', 'radioml']:
        group = sorted([r for r in records if r['domain'] == domain], key=lambda r: r['seed'])
        assert [r['seed'] for r in group] == [800, 801, 802]
        d, m, nc = DIMS[domain]
        candidates = []
        for s, r in [tuple(c) for c in group[0]['configuration']['configurations']]:
            detached = [next(b for b in rec['branches'] if b['rank1'] == s and b['rank2'] == r and b['mode'] == 'detach')
                        for rec in group]
            joint = [next(b for b in rec['branches'] if b['rank1'] == s and b['rank2'] == r and b['mode'] == 'joint')
                     for rec in group]
            for det, joi in zip(detached, joint):
                assert det['minibatch_order_sha256'] == joi['minibatch_order_sha256']
                assert det['initial_optimizer_sha256'] == joi['initial_optimizer_sha256']
            primary = interval([j['own']['acc'] - x['own']['acc'] for j, x in zip(joint, detached)])
            full = interval([j['own']['acc'] - rec['full_controls']['joint']['own']['acc']
                             for j, rec in zip(joint, group)])
            mse = interval([j['auxiliary_mse'] / x['auxiliary_mse'] for j, x in zip(joint, detached)])
            checks = dict(every_consumer_within_one_point=all(c['disagreement'] <= .01 for b in joint + detached
                                                              for c in b['contracts'].values()),
                          nonlinear_second_updates=all(b['second_delta_norm'] > 0 for b in joint + detached),
                          nonlinear_first_updates=all(b['first_delta_norm'] > 0 for b in joint + detached))
            if domain == 'har':
                checks.update(primary_noninferiority=primary['mean'] >= -.005,
                              reconstruction_progress=mse['mean'] <= .95)
            else:
                checks.update(positive_primary_gain=primary['ci95'][0] > 0,
                              noninferiority_to_full_joint=full['mean'] >= -.005)
            macs = dense_macs(s, r, d, m, nc)
            candidates.append(dict(rank1=s, rank2=r, extra_macs=macs,
                                   encoder_fraction=macs / ENCODERS[domain],
                                   utility_compatible=all(checks.values()), checks=checks,
                                   primary_gain=primary, primary_difference_to_full_joint=full,
                                   auxiliary_mse_ratio=mse,
                                   maximum_compressed_logit_error=max(b['compressed_maximum_logit_error']
                                                                      for b in joint + detached),
                                   maximum_full_replay_logit_error=max(b['full_reference_maximum_logit_error']
                                                                       for b in joint + detached),
                                   maximum_current_merge_logit_error=max(b['current_merge_maximum_logit_error']
                                                                         for b in joint + detached),
                                   total_current_merge_prediction_changes=sum(b['current_merge_prediction_changes']
                                                                              for b in joint + detached),
                                   contracts={c: interval([b['contracts'][c]['disagreement'] for b in joint])
                                              for c in joint[0]['contracts']}))
        eligible = [x for x in candidates if x['utility_compatible']]
        selected = (min(eligible, key=lambda x: (x['extra_macs'], x['rank1'], x['rank2'])) if eligible
                    else max(candidates, key=lambda x: (x['extra_macs'], x['rank1'], x['rank2'])))
        out['domains'][domain] = dict(utility_gate=bool(eligible), selected=selected, all_candidates=candidates)
    a.out.write_text(json.dumps(out, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps({d: dict(utility_gate=v['utility_gate'],
                              selected={k: v['selected'][k] for k in ('rank1', 'rank2', 'extra_macs', 'utility_compatible')})
                      for d, v in out['domains'].items()}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
