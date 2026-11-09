"""Verify locally retrieved round-two numerical evidence and checkpoint links."""
import hashlib
import json
from pathlib import Path

import torch

from paired_readout_continuation import state_hash
from readout_alignment import ARMS

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'repro/results/CPUCONT'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_record(path):
    row = json.loads(path.read_text())
    assert row['complete'] and row['device'] == 'cpu'
    assert row['torch_threads'] == 1 and row['cuda_visibility'] == ''
    return row


def main():
    mechanism = BASE / 'mechanism'
    original = {p.name: p for folder, prefix in [('har', 'h'), ('replication', 'v'),
                ('projected', 'p'), ('real', 'r')] for p in (BASE / folder).glob(prefix + '???.json')}
    training = sorted(mechanism.glob('u???.json')) + sorted(mechanism.glob('w???.json'))
    assert len(training) == 10 and len(original) == 20
    checkpoints = endpoints = 0
    for path in training:
        row = numerical_record(path)
        assert row['source_sha256'] == sha(ROOT / 'repro/scripts/paired_readout_continuation.py')
        assert row['config']['aux'] == 'recon'
        assert not row['splits']['calibration_used_for_encoder']
        assert not row['splits']['calibration_used_for_head']
        states = mechanism / (path.stem + '_states')
        reference = torch.load(states / 'reference.pt', map_location='cpu', weights_only=True)
        assert state_hash(reference['model']) == row['reference_model_sha256']
        assert state_hash(reference['optimizer']) == row['reference_optimizer_sha256']
        primary = None
        orders = set()
        checkpoints += 1
        assert len(row['branches']) == 10
        for b in row['branches']:
            assert b['initial_model_sha256'] == row['reference_model_sha256']
            assert b['initial_optimizer_sha256'] == row['reference_optimizer_sha256']
            saved = torch.load(states / ('%s_%g.pt' % (b['mode'], b['alpha'])),
                               map_location='cpu', weights_only=True)
            shared = {k: v for k, v in saved['model'].items() if k.startswith(('encoder.', 'primary.'))}
            assert all(v.device.type == 'cpu' for v in saved['model'].values())
            assert state_hash(shared) == b['final_shared_sha256']
            if b['mode'] == 'primary':
                primary = b['final_shared_sha256']
            if b['mode'] == 'detach':
                assert b['matches_primary_only_exactly'] and state_hash(shared) == primary
            orders.add(b['minibatch_order_sha256'])
            endpoints += 1
            checkpoints += 1
        assert len(orders) == 1

    repaired = sorted(mechanism.glob('d???.json'))
    assert len(repaired) == 20
    repair_endpoints = 0
    for path in repaired:
        row = numerical_record(path)
        assert row['source_sha256'] == sha(ROOT / 'repro/scripts/checkpoint_repair_audit.py')
        assert row['alignment_source_sha256'] == sha(ROOT / 'repro/scripts/readout_alignment.py')
        assert row['paired_source_sha256'] == sha(ROOT / 'repro/scripts/paired_readout_continuation.py')
        record = original[row['source_record']]
        assert row['source_record_sha256'] == sha(record)
        states = record.parent / (record.stem + '_states')
        assert row['reference_checkpoint_sha256'] == sha(states / 'reference.pt')
        assert max(row['healthy_reconstruction_error'].values()) <= 1e-7
        head = row['reference_head']
        rebuilt = {k: torch.tensor(head[k], dtype=torch.float64) for k in ('weight', 'bias')}
        assert state_hash(rebuilt) == head['sha256']
        for b in row['branches']:
            assert b['checkpoint_sha256'] == sha(states / ('%s_%g.pt' % (b['mode'], b['alpha'])))
            assert max(b['frozen_reconstruction_error'].values()) <= 1e-7
            assert max(b['diagonal_reconstruction_error'].values()) <= 1e-7
            assert tuple(b['repairs']) == ARMS
            repair_endpoints += 1

    task_paths = sorted((BASE / 'task_validation').glob('e???.json'))
    assert len(task_paths) == 10
    for path in task_paths:
        row = numerical_record(path)
        assert row['source_sha256'] == sha(ROOT / 'repro/scripts/reconstruction_task_audit.py')
        assert row['reconstruction_uses_declared_forward_filter']
        assert row['evaluation_batch_size'] == 256
        original_path = mechanism / row['source_record']
        assert row['source_record_sha256'] == sha(original_path)
        states = mechanism / (original_path.stem + '_states')
        for b in row['branches']:
            assert b['checkpoint_sha256'] == sha(states / ('%s_%g.pt' % (b['mode'], b['alpha'])))
            assert b['test_mse'] >= 0 and b['calibration_mse'] >= 0

    queues = [json.loads((BASE / name / 'queue.json').read_text())
              for name in ('mechanism', 'task_validation')]
    for q, count in zip(queues, (30, 11)):
        assert q['finished'] and not q['failed'] and not q['running'] and not q['pending']
        assert len(q['completed']) == count and q['max_workers'] <= 4
    split_path = BASE / 'task_validation/f000.json'
    split = numerical_record(split_path)
    assert split['source_sha256'] == sha(ROOT / 'repro/scripts/radioml_split_audit.py')
    assert all(split['cache_tensor_equality'].values())
    paths = training + repaired + task_paths + [split_path]
    report = dict(verified_new_training_runs=10, verified_new_continuation_endpoints=endpoints,
        verified_new_model_optimizer_checkpoints=checkpoints,
        verified_repair_checkpoint_records=len(repaired), verified_repair_endpoints=repair_endpoints,
        fixed_repair_arms=len(ARMS), verified_auxiliary_task_records=len(task_paths),
        all_reference_readouts_reproduced=True, all_frozen_and_diagonal_endpoints_reproduced=True,
        all_new_detached_controls_bitwise_equal=True, all_new_minibatch_orders_and_initial_states_match=True,
        all_saved_parameters_cpu=True, all_queues_finished=True, maximum_concurrent_workers=4,
        radioml_cache_exactly_matches_documented_split=True,
        monitored_peak_aggregate_rss_bytes=max(q['peak_aggregate_rss_bytes'] for q in queues),
        sum_job_wall_seconds=sum(c['wall_s'] for q in queues for c in q['completed']),
        input_record_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths})
    (BASE / 'mechanism_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'input_record_sha256'}, indent=2))


if __name__ == '__main__':
    main()
