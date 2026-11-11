"""Replay the equal-work comparison between the two exact serving students."""
import argparse
import json
from pathlib import Path

import torch

from analyze_har_transfer import ROOT, RESULTS, digest, interval, save
from analyze_rml_one_pass import verify_shared_legacy
from consumer_routing import decisions

PLAN = ROOT / 'docs/plans/2026-11-11-rml-matched-distillation.md'
DEFAULT = ROOT / 'tmp/conditional_contrastivermlmatched'


def audit(records, arrays):
    references = []
    for offset in range(16):
        key = f'c{offset:03d}'
        row = json.loads((records / (key + '.json')).read_text())
        assert row['complete'] and row['device'] == 'cpu' and row['stage'] == 'matched'
        assert row['plan_sha256'] == digest(PLAN) and row['seed'] == 9400 + offset and row['test_loaded']
        folder = arrays / (key + '_states')
        for name in ('replay', 'program'):
            assert digest(folder / (name + '.pt')) == row[name + '_sha256']
        state = torch.load(folder / 'program.pt', map_location='cpu', weights_only=True)
        data = torch.load(folder / 'replay.pt', map_location='cpu', weights_only=True)
        y, snr = data['labels'], data['snr']
        assert y.shape == snr.shape == (4400,)
        old = decisions(data['old'], data['thresholds'])
        teacher_hits = {name: z.argmax(1) == y for name, z in data['teachers'].items()}
        hits, max_logit_error, changed = {}, {}, {}
        for name, z in data['students'].items():
            verify_shared_legacy(dict(reference=state['reference'], chosen=state[name]))
            current, legacy = z.split(11, 1)
            hits[name] = current.argmax(1) == y
            changed[name] = (decisions(legacy, data['thresholds']) != old).sum(0).tolist()
            max_logit_error[name] = (legacy - data['old']).abs().max().item()
        delta = 100 * (hits['joint'].double() - hits['detach'].double())
        references.append(dict(seed=row['seed'], gain_pp=float(delta.mean()),
                               vs_joint_pp=100 * float((hits['joint'].double() - teacher_hits['joint'].double()).mean()),
                               teacher_gain_pp=100 * float((teacher_hits['joint'].double() - teacher_hits['detach'].double()).mean()),
                               low_snr_pp=float(delta[snr < 0].mean()), high_snr_pp=float(delta[snr >= 0].mean()),
                               max_logit_error=max_logit_error, old_decisions_changed=changed))
    metrics = {name: interval([row[name] for row in references])
               for name in ('gain_pp', 'vs_joint_pp', 'teacher_gain_pp', 'low_snr_pp', 'high_snr_pp')}
    old_changes = sum(sum(changes) for row in references for changes in row['old_decisions_changed'].values())
    return dict(plan_sha256=digest(PLAN), metrics=metrics, old_decision_changes=old_changes,
                maximum_old_logit_error=max(max(r['max_logit_error'].values()) for r in references),
                audited_programs=2 * len(references),
                passed=old_changes == 0 and metrics['gain_pp']['ci95'][0] > 0
                and metrics['vs_joint_pp']['ci95'][0] > -.25, references=references)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, default=DEFAULT)
    parser.add_argument('--arrays', type=Path, default=DEFAULT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = audit(args.records, args.arrays)
    save(RESULTS / 'rml_matched_distillation_20261111.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'references'}, indent=2))
