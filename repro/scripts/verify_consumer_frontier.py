"""Local provenance, fixed-family selection, and CNN checkpoint verification."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cnn_reconstruction_continuation import SPEC
from consumer_risk import first_certified, upper_total_disagreements
from paired_readout_continuation import state_hash
from readout_alignment import ARMS

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'repro/results/CPUCONT'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete(path):
    r = json.loads(path.read_text())
    assert r['complete'] and r['device'] == 'cpu'
    assert r['torch_threads'] == 1 and r['cuda_visibility'] == ''
    return r


def main():
    sources = {p.name: p for folder, prefix in [('har', 'h'), ('replication', 'v'), ('projected', 'p'),
        ('real', 'r'), ('mechanism', 'u'), ('mechanism', 'w'), ('frontier', 'c')]
        for p in (BASE / folder).glob(prefix + '???.json')}
    training = sorted((BASE / 'frontier').glob('c???.json'))
    audits = sorted((BASE / 'frontier').glob('b???.json')) + sorted((BASE / 'cnn_consumer').glob('n???.json'))
    assert len(sources) == 35 and len(training) == 5 and len(audits) == 35
    checkpoints = 0
    for path in training:
        r = complete(path)
        assert r['architecture'] == 'waveform_cnn' and r['architecture_spec'] == SPEC
        assert r['source_sha256'] == sha(ROOT / 'repro/scripts/paired_readout_continuation.py')
        assert r['architecture_source_sha256'] == sha(ROOT / 'repro/scripts/cnn_reconstruction_continuation.py')
        assert len(r['branches']) == 7 and r['config']['doses'] == '30,300,3000'
        states = path.parent / (path.stem + '_states')
        ref = torch.load(states / 'reference.pt', map_location='cpu', weights_only=True)
        assert state_hash(ref['model']) == r['reference_model_sha256']
        assert state_hash(ref['optimizer']) == r['reference_optimizer_sha256']
        assert not r['splits']['calibration_used_for_encoder'] and not r['splits']['calibration_used_for_head']
        checkpoints += 1
        primary = None
        order = set()
        for b in r['branches']:
            assert b['initial_model_sha256'] == r['reference_model_sha256']
            assert b['initial_optimizer_sha256'] == r['reference_optimizer_sha256']
            final = torch.load(states / ('%s_%g.pt' % (b['mode'], b['alpha'])),
                               map_location='cpu', weights_only=True)
            shared = {k: v for k, v in final['model'].items() if k.startswith(('encoder.', 'primary.'))}
            assert all(v.device.type == 'cpu' for v in final['model'].values())
            assert state_hash(shared) == b['final_shared_sha256']
            if b['mode'] == 'primary':
                primary = b['final_shared_sha256']
            if b['mode'] == 'detach':
                assert b['matches_primary_only_exactly'] and b['final_shared_sha256'] == primary
            order.add(b['minibatch_order_sha256'])
            checkpoints += 1
        assert len(order) == 1
    candidates = selected = abstentions = bound_violations = 0
    for path in audits:
        r = complete(path)
        assert r['source_sha256'] == sha(ROOT / 'repro/scripts/checkpoint_consumer_audit.py')
        assert r['risk_source_sha256'] == sha(ROOT / 'repro/scripts/consumer_risk.py')
        assert r['alignment_source_sha256'] == sha(ROOT / 'repro/scripts/readout_alignment.py')
        original = sources[r['source_record']]
        assert r['source_record_sha256'] == sha(original)
        states = original.parent / (original.stem + '_states')
        assert r['reference_checkpoint_sha256'] == sha(states / 'reference.pt')
        assert max(r['healthy_reconstruction_error'].values()) <= 1e-7
        assert not r['monitor_labels_used_for_selection'] and not r['monitor_used_for_encoder_training']
        assert not r['forecast_future_updates'] and r['family_size'] == 3015 and r['bank_delta'] == .05
        assert r['monitor_examples'] == 1024 and r['allocated_tail_probability'] == .05 / 3015
        population = r['monitor_examples'] + r['remainder_examples']
        indices = np.random.default_rng(r['monitor_seed']).permutation(population)[:1024].astype(np.int64)
        assert hashlib.sha256(indices.tobytes()).hexdigest() == r['monitor_indices_sha256']
        for b in r['branches']:
            assert b['checkpoint_sha256'] == sha(states / ('%s_%g.pt' % (b['mode'], b['alpha'])))
            assert max(b['frozen_reconstruction_error'].values()) <= 1e-7
            assert tuple(b['candidates']) == ARMS
            for c in b['candidates'].values():
                total = upper_total_disagreements(population, 1024, c['observed_disagreements'], .05 / 3015)
                assert total == c['upper_total_disagreements']
                assert abs(c['upper_remaining_disagreement'] - (total - c['observed_disagreements']) / r['remainder_examples']) < 1e-14
                assert c['loss_increase'] <= c['remainder_disagreement'] + 1e-12
                bound_violations += c['disagreement_bound_violated']
                candidates += 1
            for budget in (.05, .10):
                choice = first_certified(b['candidates'], budget, ARMS)
                decision = b['decisions'][str(budget)]
                assert decision['selected_arm'] == choice and decision['certified'] == (choice is not None)
                selected += choice is not None
                abstentions += choice is None
                if choice is not None:
                    assert decision['remainder_accuracy'] == b['candidates'][choice]['remainder_accuracy']
                    assert decision['loss_increase'] == b['candidates'][choice]['loss_increase']
    assert candidates == 3015
    queues = [json.loads((BASE / group / 'queue.json').read_text()) for group in ('frontier', 'cnn_consumer')]
    for q, expected in zip(queues, (35, 5)):
        assert q['finished'] and not q['failed'] and not q['running'] and not q['pending']
        assert len(q['completed']) == expected and q['max_workers'] <= 4
    report = dict(verified_cnn_training_runs=5, verified_new_cnn_model_optimizer_checkpoints=checkpoints,
        verified_consumer_audits=len(audits), verified_candidate_functions=candidates,
        all_reference_and_frozen_heads_reproduced=True, all_fixed_family_choices_reproduced=True,
        all_monitor_partitions_reproduced=True, monitor_labels_excluded_from_selection=True,
        all_cnn_clone_order_and_detached_identities_pass=True, all_saved_parameters_cpu=True,
        selected_decisions_both_budgets=selected, abstentions_both_budgets=abstentions,
        observed_disagreement_bound_violations=bound_violations, all_queues_finished=True,
        maximum_concurrent_workers=4, monitored_peak_aggregate_rss_bytes=max(q['peak_aggregate_rss_bytes'] for q in queues),
        sum_job_wall_seconds=sum(c['wall_s'] for q in queues for c in q['completed']),
        input_record_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in training + audits})
    (BASE / 'consumer_frontier_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'input_record_sha256'}, indent=2))


if __name__ == '__main__':
    main()
