"""Verify a pristine supplementary release; run optional checks in a temporary copy.

The default path uses only Python's standard library. It checks files and recorded
provenance, not predictions from tensor checkpoints excluded from this archive.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
TESTS = [
    'test_consumer_risk.py', 'test_readout_alignment.py',
    'test_scale_boundaries.py', 'test_readout_nullspace.py',
    'test_compiled_readout.py', 'test_nonlinear_interface.py',
    'test_nonlinear_selection.py',
    'test_affine_chart.py', 'test_vision_chart.py',
    'test_legacy_head_fit.py',
    'test_alignment_objectives.py',
    'test_consumer_contracts.py',
    'test_consumer_cells.py','test_recovery_readers.py',
    'test_shared_reference_replay.py','test_shared_reference_vision.py',
    'test_dense_replay.py',
    'test_late_replay.py','test_nonlinear_suffix.py','test_suffix_serving.py',
    'test_paired_suffix.py','test_mixed_reference.py',
    'test_consumer_routing.py','test_vision_auxiliary.py',
    'test_vision_auxiliary_selection.py','test_reference_replication.py',
    'test_small_budget_risk.py',
    'test_certified_audit.py', 'test_coupling_control.py',
    'test_contrastive_benefit.py', 'test_ridge_loo.py',
    'test_scripted_current_route.py',
]
EXPECTED_TESTS=145

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_path(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:
        raise ValueError('Unsafe manifest path: ' + name)
    target = root.joinpath(*path.parts)
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Payload path leaves the artifact: ' + name)
    return target


def verify_files(root):
    manifest = json.loads((root / 'ARTIFACT_MANIFEST.json').read_text(encoding='utf-8'))
    files = manifest['sha256']
    if not files:
        raise ValueError('Empty artifact manifest')
    for name, expected in files.items():
        path = payload_path(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Missing or changed payload: ' + name)
    inputs, figures, seen = [], [], set()
    def inspect_tex(name):
        if name in seen:
            return
        seen.add(name)
        tex = payload_path(root, 'paper/' + name).read_text(encoding='utf-8')
        children = re.findall(r'\\input\{([^}]+)\}', tex)
        figures.extend(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex))
        for child in children:
            inputs.append(child)
            inspect_tex(child)
    inspect_tex('RepFix_ICLR2027.tex')
    for name in inputs + figures:
        if 'paper/' + name not in files:
            raise ValueError('Unmanifested manuscript input: ' + name)
    if len(figures) != manifest['publication_figures']:
        raise ValueError('Included manuscript figures differ from the release manifest')
    exported=json.loads((root/'paper/figures/publication_figure_manifest.json').read_text(encoding='utf-8'))['figures']
    included={PurePosixPath(name).stem for name in figures}
    if len(included)!=len(figures) or included!=set(exported):
        raise ValueError('A manuscript figure is duplicated, missing, or unused')
    if manifest['pdf_sha256'] != digest(root / 'paper/RepFix_ICLR2027.pdf'):
        raise ValueError('The manuscript PDF does not match the manifest')

    sources = plans = records = versioned_sources = repr_sources = 0
    python_hashes = {value for name, value in files.items() if name.endswith('.py')}
    # Some historical vision replay records called state_hash(bytes), which
    # hashes repr(bytes). Preserve that convention and the exact source bytes.
    legacy_repr_names = {'vision_scale_replay.py', 'vision_boundary_cpu.py'}
    for name in files:
        if PurePosixPath(name).name in legacy_repr_names:
            python_hashes.add(hashlib.sha256(repr((root/name).read_bytes()).encode()).hexdigest())
    plan_hashes = {value for name, value in files.items()
                   if name.startswith('docs/plans/') and name.endswith('.md')}
    base = 'repro/results/CPUCONT/'
    for name in sorted(files):
        if not name.startswith(base) or not name.endswith('.json'):
            continue
        row = json.loads((root / name).read_text(encoding='utf-8'))
        if not isinstance(row, dict):
            continue
        if row.get('complete'):
            records += 1
        source_map = row.get('source_sha256', {})
        if isinstance(source_map, str):
            if source_map not in python_hashes:
                raise ValueError('Recorded driver digest is unavailable: ' + name)
            sources += 1
            source_map = {}
        for field, expected in row.items():
            if field.endswith('_source_sha256') and isinstance(expected, str):
                if expected not in python_hashes:
                    raise ValueError('Recorded source digest is unavailable: ' + name + '/' + field)
                sources += 1
        for source, expected in source_map.items():
            if not isinstance(expected, str) or len(expected) != 64:
                continue
            path = payload_path(root, source)
            if not path.is_file():
                path = payload_path(root, 'repro/scripts/' + source)
            candidates=[path,payload_path(root,'repro/source_versions/'+expected+'/'+PurePosixPath(source).name)]
            matched=None
            for candidate in candidates:
                if not candidate.is_file():continue
                if digest(candidate)==expected:
                    matched=candidate
                    break
                if candidate.name in legacy_repr_names and hashlib.sha256(repr(candidate.read_bytes()).encode()).hexdigest()==expected:
                    matched=candidate
                    repr_sources+=1
                    break
            if matched is None:
                raise ValueError('Recorded source mismatch in ' + name + ': ' + source)
            versioned_sources+=matched!=path
            sources += 1
        for field,expected in row.items():
            if field.endswith('plan_sha256') and isinstance(expected,str):
                if expected not in plan_hashes:
                    raise ValueError('Recorded plan unavailable: '+name+'/'+field)
                plans+=1
        if isinstance(row.get('source'),dict) and 'script_sha256' in row['source']:
            if row['source']['script_sha256'] not in python_hashes:
                raise ValueError('Nested recorded script unavailable: '+name)
            sources+=1

    choice = json.loads((root / base / 'nonlinear_development_choice.json').read_text(encoding='utf-8'))
    if any(v['confirm'] for v in choice['selected'].values()):
        raise ValueError('The nonlinear decision differs from the released failed gate')
    development = [name for name in files if re.fullmatch(base + r'nonlinear_development/b\d{3}\.json', name)]
    if len(development) != 6 or any(name.startswith(base + 'nonlinear_confirmation/') for name in files):
        raise ValueError('Unexpected nonlinear development/confirmation family')
    for name in development:
        row = json.loads((root / name).read_text(encoding='utf-8'))
        if row['splits']['test_metrics_computed'] or any('test' in b for b in row['branches']):
            raise ValueError('Development record contains test outcomes: ' + name)

    chart_dev = [name for name in files if re.fullmatch(base+r'chart_development/a\d{3}\.json', name)]
    chart_confirm = [name for name in files if re.fullmatch(base+r'chart_confirmation/c\d{3}\.json', name)]
    chart_replay = [name for name in files if re.fullmatch(base+r'chart_replay/e\d{3}\.json', name)]
    if len(chart_dev)!=6 or len(chart_confirm)!=10 or len(chart_replay)!=10:
        raise ValueError('The fixed chart evidence family is incomplete')
    selection_name=base+'chart_development_choice.json'
    selection=json.loads((root/selection_name).read_text())
    if selection['tasks']['har']['confirm'] or selection['tasks']['radioml']['selected']!='paired_ridge':
        raise ValueError('The frozen chart choice changed')
    for name in chart_confirm:
        row=json.loads((root/name).read_text())
        if row['selection_sha256']!=digest(root/selection_name):
            raise ValueError('Confirmation choice hash differs: '+name)
    for name in chart_dev:
        row=json.loads((root/name).read_text())
        if row['splits']['test_metrics_computed']:
            raise ValueError('Chart development opened test outcomes')
    chart_summary=json.loads((root/base/'chart_confirmation_summary.json').read_text())
    if chart_summary['tasks']['radioml']['passes'] or chart_summary['tasks']['radioml']['metrics']['certified']!=0:
        raise ValueError('The failed chart compatibility criterion was changed')

    protected_original = [name for name in files if re.fullmatch(base+r'protected_head/ph_(?:har_)?conf_30\d\.json', name)]
    protected_audits = [name for name in files if re.fullmatch(base+r'protected_head_audit/r\d{3}\.json', name)]
    protected_replays = [name for name in files if re.fullmatch(base+r'protected_head_reader_replay/v\d{3}\.json', name)]
    if any(len(family)!=20 for family in [protected_original,protected_audits,protected_replays]):
        raise ValueError('The protected-head audit family is incomplete')
    original_hashes={digest(root/name) for name in protected_original}
    audit_hashes={digest(root/name) for name in protected_audits}
    for name in protected_audits:
        row=json.loads((root/name).read_text())
        if not row['complete'] or row['original_record_sha256'] not in original_hashes or row['maximum_original_metric_error']!=0:
            raise ValueError('Original protected-head replay changed: '+name)
        if row['new_family_size']!=140 or not row['no_encoder_updates'] or row['labels_used_for_fitting']:
            raise ValueError('Fixed retrospective fit protocol changed: '+name)
    for name in protected_replays:
        row=json.loads((root/name).read_text())
        if (not row['complete'] or row['audit_sha256'] not in audit_hashes or
                len(row['branches'])!=7 or row['maximum_metric_error']!=0 or
                row['maximum_stationarity_residual']>1e-6 or row['refitting_performed']):
            raise ValueError('Frozen reader verification changed: '+name)
    protected_summary=json.loads((root/base/'protected_head_audit_summary.json').read_text())
    if protected_summary['original_gate_replaced']:
        raise ValueError('Retrospective results replaced original decisions')
    for domain,original_count,ols_count in [('har',9,10),('radioml',0,0)]:
        row=protected_summary['domains'][domain]
        if row['original']['passes'] or row['original']['certified']!=original_count or row['frozen_controls']['protected_ols']['certified']!=ols_count:
            raise ValueError('Protected-head verdict changed: '+domain)

    alignment_dev=[name for name in files if re.fullmatch(base+r'alignment_development/a\d{3}\.json',name)]
    alignment_conf=[name for name in files if re.fullmatch(base+r'alignment_confirmation/c\d{3}\.json',name)]
    alignment_replay=[name for name in files if re.fullmatch(base+r'alignment_replay/v\d{3}\.json',name)]
    if (len(alignment_dev),len(alignment_conf),len(alignment_replay))!=(6,10,10):
        raise ValueError('The joint-alignment evidence family is incomplete')
    original_development_hashes={value for name,value in files.items() if re.fullmatch(base+r'protected_head/ph_(?:har_)?dev_80\d\.json',name)}
    for name in alignment_dev:
        row=json.loads((root/name).read_text())
        if (not row['complete'] or row['smoke'] or row['splits']['test_metrics_computed'] or
                row['source_record_sha256'] not in original_development_hashes or any('test' in b for b in row['branches'])):
            raise ValueError('Alignment development provenance changed: '+name)
    alignment_choice_name=base+'alignment_development_choice.json'
    alignment_choice=json.loads((root/alignment_choice_name).read_text())
    if not alignment_choice['choices_frozen_before_confirmation'] or alignment_choice['selection_uses_accuracy']:
        raise ValueError('Alignment selection rule changed')
    if any(v['confirm'] for v in alignment_choice['domains']['radioml'].values()):
        raise ValueError('A failed SNR alignment gate opened confirmation')
    if alignment_choice['domains']['har']['feature_map']['confirm']:
        raise ValueError('The failed HAR feature-map policy was changed')
    confirmation_hashes={digest(root/name) for name in alignment_conf}
    for name in alignment_conf:
        row=json.loads((root/name).read_text())
        if (not row['complete'] or row['domain']!='har' or row['seed'] not in range(1300,1310) or
                row['selection_sha256']!=digest(root/alignment_choice_name)):
            raise ValueError('Fresh confirmation differs from the frozen choice: '+name)
        if row['configuration']['consumer_bank']!=160 or row['configuration']['primary_monitor']!=1024:
            raise ValueError('Fresh confirmation acceptance budget changed')
        if {b['mode'] for b in row['branches']}!={'detach','joint','protected','logit_map','direct_ols'}:
            raise ValueError('Fresh confirmation candidate set changed')
    for name in alignment_replay:
        row=json.loads((root/name).read_text())
        if (not row['complete'] or row['confirmation_record_sha256'] not in confirmation_hashes or
                row['maximum_metric_error']!=0 or row['maximum_ols_stationarity_residual']>1e-6 or
                len(row['branches'])!=5 or row['refitting_performed']):
            raise ValueError('Fresh-consumer replay changed: '+name)
    fresh=json.loads((root/base/'alignment_confirmation_summary.json').read_text())
    if not fresh['prior_study_decisions_unchanged'] or fresh['primary_monitor']!=1024:
        raise ValueError('Fresh study replaced an earlier verdict or its primary budget')
    for mode,passes,count in [('protected',True,10),('direct_ols',True,10),('logit_map',False,9)]:
        row=fresh['domains']['har'][mode]
        if row['passes']!=passes or row['certificates']['1024']['count']!=count:
            raise ValueError('Fresh primary verdict changed: '+mode)
    native=json.loads((root/base/'alignment_native_timing.json').read_text())
    if (not native['complete'] or native['primary_runtime_dtype']!='float32' or
            len(native['branches'])!=30 or len(native['native_primary_metrics'])!=50):
        raise ValueError('Native serving audit is incomplete')

    completion=[name for name in files if re.fullmatch(base+r'completion_har/h\d{3}\.json',name)]
    completion_replay=[name for name in files if re.fullmatch(base+r'completion_replay/r\d{3}\.json',name)]
    if (len(completion),len(completion_replay))!=(10,10):
        raise ValueError('The matched completion or replay is incomplete')
    completion_hashes={digest(root/name) for name in completion}
    for name in completion:
        row=json.loads((root/name).read_text())
        if (not row['complete'] or row['phase']!='retrospective_completion' or row['new_bank']!=1000 or
                row['source_record_sha256'] not in confirmation_hashes or len(row['branches'])!=10 or
                row['feature_training']['lam']!=1 or not all(row['feature_training']['nonlinear_layers_changed'].values())):
            raise ValueError('Matched completion scope changed: '+name)
        for b in row['branches']:
            contracts=b['test']['contracts']
            if set(contracts)!={'classification','coverage80','coverage50'}:
                raise ValueError('A versioned consumer is missing: '+name)
            flows=contracts['classification']['flows']
            if (flows['changed']!=sum(flows[k] for k in ['correct_to_wrong','wrong_to_correct','wrong_to_wrong']) or
                    flows['new_correct']-flows['old_correct']!=flows['wrong_to_correct']-flows['correct_to_wrong']):
                raise ValueError('Decision flows do not conserve changes/correctness: '+name)
            for c in contracts.values():
                cert=c['certificate']
                if cert['monitor_examples']!=1024 or cert['allocated_tail_probability']!=.05/1000:
                    raise ValueError('The retrospective monitor allocation changed')
    for name in completion_replay:
        row=json.loads((root/name).read_text())
        if (not row['complete'] or row['completion_record_sha256'] not in completion_hashes or
                row['maximum_metric_error']!=0 or row['maximum_ols_stationarity_residual']>1e-6 or len(row['branches'])!=10):
            raise ValueError('Expanded contract replay is incomplete: '+name)
    completion_summary=json.loads((root/base/'completion_summary.json').read_text())
    if not completion_summary['old_confirmation_verdicts_unchanged'] or completion_summary['distinct_endpoints']!=90:
        raise ValueError('Retrospective evidence replaced earlier decisions or duplicated endpoints')
    for mode,b in completion_summary['branches'].items():
        if mode in ['detach','detach_ols']:continue
        if any(b['contracts'][c]['certified'] for c in ['coverage80','coverage50']):
            raise ValueError('A selective-consumer negative result was changed')
    timing=json.loads((root/base/'completion_timing.json').read_text())
    if not timing['complete'] or len(timing['branches'])!=40 or any(b['encoder_calls']!=1 for b in timing['branches']):
        raise ValueError('Strong-control native serving evidence is incomplete')
    vision_new=[name for name in files if re.fullmatch(base+r'vision_consumer_screen/v\d{3}\.json',name)]
    vision_replay=[name for name in files if re.fullmatch(base+r'vision_consumer_replay/w\d{3}\.json',name)]
    if (len(vision_new),len(vision_replay))!=(3,3):
        raise ValueError('The stopped vision screen or its replay is incomplete')
    vision_hashes={digest(root/name) for name in vision_new}
    for name in vision_new:
        row=json.loads((root/name).read_text())
        if not row['complete'] or row['smoke'] or row['splits']['official_test_loaded'] or row['splits']['test_metrics_computed']:
            raise ValueError('The vision screen opened test or included smoke outcomes')
    for name in vision_replay:
        row=json.loads((root/name).read_text())
        if (not row['complete'] or row['source_record_sha256'] not in vision_hashes or
                row['maximum_metric_error']!=0 or row['maximum_feature_error']>1e-4 or row['maximum_ols_stationarity_residual']>1e-6):
            raise ValueError('The vision consumer replay is incomplete')
    vision_choice=json.loads((root/base/'vision_consumer_choice.json').read_text())
    if (vision_choice['continue_development'] or vision_choice['official_test_evaluated'] or vision_choice['full_grid_completed'] or
            vision_choice['execution']['failed']!=2 or vision_choice['execution']['interrupted']!=1 or
            vision_choice['execution']['unlaunched']!=2 or vision_choice['resource_scientific_gate']['primary_gain']):
        raise ValueError('The stopped vision gate or unmeasured branches were rewritten')
    execution=json.loads((root/base/'completion_execution_audit.json').read_text())
    if (not execution['complete'] or not execution['all_owned_queues_stopped_or_finished'] or
            execution['gpu_compute_used'] or execution['total_new_remote_bytes']>=4*2**30 or
            any(q['running'] or not q['finished'] or q['max_workers']>2 for q in execution['queues'].values())):
        raise ValueError('Completion resource/transport audit is incomplete')

    # Loss-only routes remain stopped before test, including the formally
    # eligible class-only HAR candidate that was dominated by standard KL.
    for folder,pattern,count,endpoints in [('cell_stage_a','a',6,19),('cell_stage_b','b',6,36),('cell_class_secondary','s',6,16)]:
        family=[name for name in files if re.fullmatch(base+folder+'/'+pattern+r'\d{3}\.json',name)]
        if len(family)!=count:raise ValueError('Incomplete cell development: '+folder)
        for name in family:
            row=json.loads((root/name).read_text())
            if (not row['complete'] or row['phase']!='development' or row.get('smoke',False) or len(row['branches'])!=endpoints or
                    row['splits']['test_metrics_computed'] or any('test' in b for b in row['branches'])):
                raise ValueError('Cell development scope changed: '+name)
    stop=json.loads((root/base/'cell_development_stop.json').read_text())
    if (any(stop['primary_three_consumer_gate'].values()) or not stop['secondary_har_gate_passed'] or
            stop['test_confirmation_started'] or not stop['futility_stop_is_additional_development_judgment']):
        raise ValueError('Cell development verdict or stopping scope changed')
    replay_choice=base+'shared_replay_development_choice.json';choice_sha=digest(root/replay_choice)
    replay_dev=[name for name in files if re.fullmatch(base+r'shared_replay_development/d\d{3}\.json',name)]
    replay_confirm=[name for name in files if re.fullmatch(base+r'shared_replay_confirmation/c\d{3}\.json',name)]
    if (len(replay_dev),len(replay_confirm))!=(6,20):raise ValueError('Incomplete shared-replay evidence')
    for name in replay_dev:
        row=json.loads((root/name).read_text())
        if not row['complete'] or row['phase']!='development' or row['smoke'] or len(row['branches'])!=8 or row['splits']['test_metrics_computed']:
            raise ValueError('Shared-replay development scope changed')
    replay_records=[]
    for name in replay_confirm:
        row=json.loads((root/name).read_text());replay_records.append(row)
        if (not row['complete'] or row['phase']!='confirmation' or row['selection_sha256']!=choice_sha or
                not 2500<=row['seed']<=2509 or len(row['branches'])!=2 or set(row['full_controls'])!={'detach','joint'}):
            raise ValueError('Shared-replay confirmation changed: '+name)
        if len({b['minibatch_order_sha256'] for b in [*row['branches'],*row['full_controls'].values()]})!=1:
            raise ValueError('Shared-replay future orders differ')
        for b in row['branches']:
            if b['test']['changed_first_relu_gates']<=0 or b['test']['changed_second_relu_gates']<=0:
                raise ValueError('A nonlinear update did not change its gates')
            for c in b['test']['contracts'].values():
                cert=c['certificate']
                if (c['disagreement']!=0 or not cert['certified'] or cert['monitor_examples']!=1024 or
                        cert['allocated_tail_probability']!=.05/600):
                    raise ValueError('Shared-replay consumer evidence changed')
    summary=json.loads((root/base/'shared_replay_confirmation_summary.json').read_text())
    if summary['cost_gate_passed'] or not summary['domains']['har']['utility_compatibility_confirmed'] or summary['domains']['radioml']['utility_compatibility_confirmed']:
        raise ValueError('The partial frontier was upgraded to a complete target')
    for domain in ['har','radioml']:
        rows=[r for r in replay_records if r['domain']==domain]
        if {r['seed'] for r in rows}!=set(range(2500,2510)):raise ValueError('Fresh replay seeds differ')
        pairs=[{b['mode']:b for b in r['branches']} for r in rows]
        values={
            'primary_gain':[p['joint']['test']['own']['acc']-p['detach']['test']['own']['acc'] for p in pairs],
            'primary_difference_to_full_joint':[p['joint']['test']['own']['acc']-r['full_controls']['joint']['test']['own']['acc'] for p,r in zip(pairs,rows)],
            'auxiliary_mse_ratio':[p['joint']['test']['auxiliary_mse']/p['detach']['test']['auxiliary_mse'] for p in pairs]}
        import statistics,math
        for key,v in values.items():
            item=summary['domains'][domain][key];m=statistics.mean(v);half=2.2621571628540993*statistics.stdev(v)/math.sqrt(10)
            if max(abs(m-item['mean']),abs(m-half-item['ci95'][0]),abs(m+half-item['ci95'][1]))>1e-10:
                raise ValueError('Shared-replay paired interval mismatch')
    audit=json.loads((root/base/'shared_replay_independent_audit.json').read_text())
    if not audit['complete'] or len(audit['branches'])!=40 or not all(v['contracts_match'] and v['current_metrics_match'] and v['full_reference_bitwise_equal'] for v in audit['branches']):
        raise ValueError('Independent shared-replay audit is incomplete')

    # Dense low-rank companion: the utility gap is a rank gap, confirmed on SNR
    # and stopped at development on HAR.  The cost failure is retained.
    dense_plan=digest(root/'docs/plans/2026-11-10-dense-lowrank-replay.md')
    dense_choice_name=base+'dense_replay_development_choice.json'
    dense_choice=json.loads((root/dense_choice_name).read_text())
    dense_choice_sha=digest(root/dense_choice_name)
    if (not dense_choice['choices_frozen_before_confirmation'] or dense_choice['ranking_uses_accuracy'] or
            dense_choice['domains']['har']['utility_gate'] or not dense_choice['domains']['radioml']['utility_gate'] or
            (dense_choice['domains']['radioml']['selected']['rank1'],dense_choice['domains']['radioml']['selected']['rank2'])!=(16,12)):
        raise ValueError('The dense development verdict or selection changed')
    dense_dev=[name for name in files if re.fullmatch(base+r'dense_replay_development/d\d{3}\.json',name)]
    dense_conf=[name for name in files if re.fullmatch(base+r'dense_replay_confirmation/c\d{3}\.json',name)]
    if (len(dense_dev),len(dense_conf))!=(6,10):
        raise ValueError('Incomplete dense-replay evidence')
    for name in dense_dev:
        row=json.loads((root/name).read_text())
        if (not row['complete'] or row['phase']!='development' or row['plan_sha256']!=dense_plan or
                row['splits']['test_metrics_computed'] or len(row['branches'])!=6 or
                any('test' in b for b in row['branches'])):
            raise ValueError('Dense development scope changed: '+name)
    dense_rows=[]
    for name in dense_conf:
        row=json.loads((root/name).read_text());dense_rows.append(row)
        if (not row['complete'] or row['phase']!='confirmation' or row['domain']!='radioml' or
                not 2520<=row['seed']<=2529 or row['selection_sha256']!=dense_choice_sha or
                row['plan_sha256']!=dense_plan or len(row['branches'])!=2 or
                set(row['full_controls'])!={'detach','joint'}):
            raise ValueError('Dense confirmation changed: '+name)
        if len({b['minibatch_order_sha256'] for b in [*row['branches'],*row['full_controls'].values()]})!=1:
            raise ValueError('Dense future orders differ')
        for b in row['branches']:
            if (b['rank1'],b['rank2'])!=(16,12) or b['test']['changed_first_relu_gates']<=0 or b['test']['changed_second_relu_gates']<=0:
                raise ValueError('A dense nonlinear update did not change its gates')
            if b['test']['packed_serving']['maximum_legacy_logit_error']>1e-3 or b['test']['full_reference_maximum_logit_error']!=0:
                raise ValueError('Dense replay identity changed')
            for c in b['test']['contracts'].values():
                cert=c['certificate']
                if (c['disagreement']!=0 or not cert['certified'] or cert['monitor_examples']!=1024 or
                        cert['allocated_tail_probability']!=.05/600):
                    raise ValueError('Dense consumer evidence changed')
    dense_summary=json.loads((root/base/'dense_replay_confirmation_summary.json').read_text())
    if (not dense_summary['utility_gate']['passed'] or dense_summary['configuration']['extra_macs']!=12288 or
            any(v!=0 for v in dense_summary['maximum_consumer_disagreement'].values()) or
            dense_summary['total_current_merge_prediction_changes']!=0):
        raise ValueError('The dense SNR utility verdict changed')
    dense_pairs=[{b['mode']:b for b in r['branches']} for r in dense_rows]
    dense_values={
        'primary_gain_vs_detached':[p['joint']['test']['own']['acc']-p['detach']['test']['own']['acc'] for p in dense_pairs],
        'primary_difference_to_full_joint':[p['joint']['test']['own']['acc']-r['full_controls']['joint']['test']['own']['acc'] for p,r in zip(dense_pairs,dense_rows)],
        'auxiliary_mse_ratio':[p['joint']['test']['auxiliary_mse']/p['detach']['test']['auxiliary_mse'] for p in dense_pairs]}
    for key,v in dense_values.items():
        item=dense_summary[key];m=statistics.mean(v);half=2.2621571628540993*statistics.stdev(v)/math.sqrt(10)
        if max(abs(m-item['mean']),abs(m-half-item['ci95'][0]),abs(m+half-item['ci95'][1]))>1e-10:
            raise ValueError('Dense paired interval mismatch')
    dense_native=json.loads((root/base/'dense_replay_native.json').read_text())
    if (not dense_native['complete'] or len(dense_native['branches'])!=20 or
            any(b['random_input_maximum_errors']['packed_replay']>1e-3 for b in dense_native['branches'])):
        raise ValueError('Dense native serving evidence is incomplete')
    dense_medians={backend+'/'+batch:statistics.median(
        [b['backends'][backend]['timings'][batch]['ratios_to_current']['packed_replay'] for b in dense_native['branches']])
        for backend in ['eager','torchscript'] for batch in ['1','1024']}
    if dense_medians['torchscript/1']<=1.2 or dense_medians['torchscript/1024']<=1.1:
        raise ValueError('The dense cost failure was overwritten')

    visual=json.loads((root/base/'vision_shared_replay_choice.json').read_text())
    if visual['continue_development'] or visual['official_test_evaluated'] or len(visual['all_candidates'])!=6 or any(v['passes'] for v in visual['all_candidates']):
        raise ValueError('The visual capacity stop was changed')
    va=json.loads((root/base/'vision_shared_replay_independent_audit.json').read_text())
    if not va['complete'] or len(va['branches'])!=10 or any(v['maximum_metric_error']!=0 or not v['contracts_match'] for v in va['branches']):
        raise ValueError('Visual replay is incomplete')
    transport=json.loads((root/base/'recovery_replay_execution_audit.json').read_text())
    if (transport['unfinished_campaigns_not_included'] or transport['gpu_compute_used'] or transport['new_state_files_sha256_verified']!=630 or
            transport['total_new_remote_bytes']>=4*2**30 or any(q['running'] or not q['finished'] for q in transport['queues'].values())):
        raise ValueError('New CPU transport or resource checks are incomplete')
    finite=json.loads((root/base/'recovery_replay_finite_state_audit.json').read_text())
    if (not finite['complete'] or not finite['all_floating_tensors_finite'] or finite['states']!=630 or
            finite['transport_audit_sha256']!=digest(root/base/'recovery_replay_execution_audit.json')):
        raise ValueError('Transferred floating tensor check is incomplete')

    from verify_frontier_records import verify as verify_frontier
    frontier=verify_frontier(root)
    from verify_followup_records import verify as verify_followup
    followup=verify_followup(root)

    return manifest, dict(payload_files=len(files), manuscript_figures=len(figures),
        numerical_inputs=len(inputs), completed_cpu_records=records,
        recorded_source_hashes_checked=sources, recorded_plan_hashes_checked=plans,
        historical_source_versions_checked=versioned_sources,
        historical_repr_bytes_digests_checked=repr_sources,
        recorded_protected_original_replays_checked=60,recorded_protected_control_replays_checked=140,
        alignment_development_records=6,fresh_training_confirmation_records=10,
        recorded_fresh_consumer_replays_checked=50,native_serving_consumers_measured=30,
        matched_completion_records=10,expanded_consumer_endpoints_replayed=100,distinct_expanded_endpoints=90,
        strong_control_native_serving_records=40,new_vision_completed_endpoints=3,new_vision_consumers_replayed=7,
        new_vision_test_evaluated=False,
        cell_development_endpoints=426,shared_replay_development_records=6,shared_replay_confirmation_records=20,
        shared_replay_independently_replayed_endpoints=40,shared_replay_monitored_contracts=120,
        shared_replay_cost_gate_passed=False,residual_block_candidates=6,residual_block_endpoints_replayed=10,
        new_scientific_states_sha256_verified=630,
        dense_replay_development_records=len(dense_dev),dense_replay_confirmation_records=len(dense_conf),
        dense_replay_monitored_contracts=60,dense_replay_cost_gate_passed=False,
        dense_packed_ratio_batch1=float(dense_medians['torchscript/1']),
        dense_packed_ratio_batch1024=float(dense_medians['torchscript/1024']),
        nonlinear_development_records=len(development), nonlinear_confirmation_tasks=0,
        frontier_extension=frontier,research_followup=followup,manuscript_pdf_sha256=manifest['pdf_sha256'])


def execute(command, root):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONUTF8='1')
    start = time.perf_counter()
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=300)
    if result.returncode:
        raise RuntimeError('Check failed: ' + ' '.join(command[1:]) + '\n' +
                           result.stdout[-4000:] + '\n' + result.stderr[-2000:])
    return result.stdout, round(time.perf_counter() - start, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests', action='store_true', help='Run the focused structural checks on CPU')
    parser.add_argument('--figures', action='store_true', help='Rebuild all included figures and compare numerical artists')
    parser.add_argument('--paper', action='store_true', help='Rebuild the manuscript and compare page text and pixels')
    args = parser.parse_args()
    manifest, report = verify_files(ROOT)
    report['checks'] = {}
    report['trained_tensor_states_replayed'] = False
    if not (args.tests or args.figures or args.paper):
        print(json.dumps(report, indent=2))
        return

    with tempfile.TemporaryDirectory(prefix='repfix-review-') as temporary:
        copy = Path(temporary)
        for name in manifest['sha256']:
            target = payload_path(copy, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        shutil.copyfile(ROOT / 'ARTIFACT_MANIFEST.json', copy / 'ARTIFACT_MANIFEST.json')
        if args.tests:
            print('Running structural checks on CPU...', flush=True)
            stdout, elapsed = execute([sys.executable, '-m', 'pytest'] +
                ['repro/scripts/' + name for name in TESTS] + ['-q'], copy)
            passed = re.search(r'(\d+) passed', stdout)
            if not passed or int(passed.group(1)) != EXPECTED_TESTS:
                raise ValueError('Unexpected focused-test result: ' + stdout[-1000:])
            report['checks']['structural_tests'] = dict(passed=EXPECTED_TESTS, elapsed_seconds=elapsed)
        if args.figures:
            print('Rebuilding all included figures from recorded results...', flush=True)
            reference = json.loads((ROOT / 'paper/figures/publication_figure_manifest.json').read_text())
            _, elapsed = execute([sys.executable, 'repro/scripts/render_paper_figures.py'], copy)
            rebuilt = json.loads((copy / 'paper/figures/publication_figure_manifest.json').read_text())
            if set(reference['figures']) != set(rebuilt['figures']):
                raise ValueError('The figure set changed')
            for name, original in reference['figures'].items():
                for field in ['width_inches', 'height_inches', 'numerical_artists_sha256']:
                    if original[field] != rebuilt['figures'][name][field]:
                        raise ValueError('Figure evidence differs: ' + name + '/' + field)
            report['checks']['figures'] = dict(rebuilt=len(reference['figures']), matching_numerical_artists=len(reference['figures']),
                elapsed_seconds=elapsed)
        if args.paper:
            print('Building and comparing the manuscript...', flush=True)
            expected_pages=manifest['manuscript_snapshot']['total_pages']
            expected_main=manifest['manuscript_snapshot']['main_text_pages']
            _, elapsed = execute([sys.executable, 'repro/scripts/build_cpu_paper.py'], copy)
            build = json.loads((copy / 'tmp/cpu_build/build_report.json').read_text())
            if (build['warnings'] or not 1<=expected_main<=9 or
                    build['main_text_pages'] != expected_main or build['total_pages'] != expected_pages):
                raise ValueError('Unexpected manuscript build: ' + json.dumps(build))
            import fitz
            with fitz.open(ROOT / 'paper/RepFix_ICLR2027.pdf') as old, \
                    fitz.open(copy / 'paper/RepFix_ICLR2027.pdf') as new:
                if len(old) != len(new) or len(new) != expected_pages:
                    raise ValueError('Manuscript page count changed')
                text_matches = pixel_matches = 0
                for a, b in zip(old, new):
                    text_matches += a.get_text() == b.get_text()
                    pixel_matches += a.get_pixmap(alpha=False).samples == b.get_pixmap(alpha=False).samples
            if text_matches != expected_pages or pixel_matches != expected_pages:
                raise ValueError('Rendered manuscript differs: text %s, pixels %s of %s. '
                    'Check the LaTeX and figure font environment.' % (text_matches, pixel_matches,expected_pages))
            report['checks']['paper'] = dict(main_text_pages=expected_main,total_pages=expected_pages,warnings=[],
                matching_page_text=text_matches,matching_page_pixels_at_72dpi=pixel_matches,elapsed_seconds=elapsed)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
