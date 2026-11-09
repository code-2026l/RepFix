"""Local state/replay audit of the gated nonlinear-channel development study."""
import hashlib
import json
from pathlib import Path

import torch

from paired_readout_continuation import state_hash
from select_nonlinear_configuration import exact_accuracy_mean, endpoint
from summarize_cpu_evidence import describe

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'repro/results/CPUCONT'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    plan = ROOT/'docs/plans/2026-11-09-nonlinear-interface.md'
    choice_path = BASE/'nonlinear_development_choice.json'
    choice = json.loads(choice_path.read_text())
    assert choice['plan_sha256']==sha(plan)
    assert not any(item['confirm'] for item in choice['selected'].values())
    assert not list((BASE/'nonlinear_confirmation').glob('c???.json'))
    paths = sorted((BASE/'nonlinear_development').glob('b???.json'))
    replay_paths = sorted((BASE/'nonlinear_validation_replay').glob('e???.json'))
    assert len(paths)==len(replay_paths)==6
    records = [json.loads(path.read_text()) for path in paths]
    states_verified = endpoints = protected = 0
    max_logit = max_fusion = max_replay = 0.
    for index,(path,record) in enumerate(zip(paths,records)):
        assert record['complete'] and record['phase']=='development'
        assert record['device']=='cpu' and record['cuda_visibility']=='' and record['torch_threads']==1
        assert record['plan_sha256']==sha(plan)
        assert record['seed']==500+index%3 and record['domain']==('har' if index<3 else 'radioml')
        assert len(record['branches'])==(19 if index<3 else 10)
        assert record['configuration']['encoder_dtype']=='float64'
        assert record['splits']['primary_label_fraction']==(.01 if index<3 else .2)
        assert not record['splits']['test_metrics_computed']
        assert not record['splits']['validation_used_for_encoder']
        assert not record['splits']['validation_used_for_healthy_head']
        assert choice['development_record_sha256'][path.relative_to(ROOT).as_posix()]==sha(path)
        for name,digest in record['source_sha256'].items():
            assert sha(ROOT/'repro/scripts'/name)==digest,name
        states = path.parent/(path.stem+'_states')
        ref = torch.load(states/'reference.pt',map_location='cpu',weights_only=True)
        assert state_hash(ref['model'])==record['reference_model_sha256']
        assert state_hash(ref['optimizer'])==record['reference_optimizer_sha256']
        a,p = ref['head']['weight']/ref['sd'].reshape(-1),ref['projector']
        assert float((a@p).norm()/a.norm())<1e-10 and float((p@p-p).norm())<1e-10
        assert len({row['minibatch_order_sha256'] for row in record['branches']})==1
        side_initializations = {}
        states_verified += 1
        for row in record['branches']:
            assert 'test' not in row
            assert row['initial_shared_model_sha256']==record['reference_model_sha256']
            assert row['initial_shared_optimizer_state_sha256']==state_hash(ref['optimizer']['state'])
            raw = torch.load(states/(row['key']+'.pt'),map_location='cpu',weights_only=True)
            assert state_hash(raw['model'])==row['final_model_sha256']
            assert state_hash(raw['optimizer'])==row['final_optimizer_sha256']
            assert all(value.device.type=='cpu' for value in raw['model'].values())
            for name,value in raw['model'].items():
                if name.startswith(('base_trunk.','base_final.','side_trunk.','side_final.','trunk.','final.','projector')):
                    assert value.dtype==torch.float64
            states_verified += 1;endpoints += 1
            max_fusion = max(max_fusion,row['fusion_max_absolute_feature_error'])
            assert row['fusion_max_absolute_feature_error']<1e-10
            if row['channel_width']:
                key = (row['channel_width'],row['alpha'])
                side_initializations.setdefault(key,set()).add(row['initial_side_sha256'])
                for name,value in raw['model'].items():
                    if name.startswith(('base_trunk.','base_final.')):
                        original = name.replace('base_trunk.','trunk.').replace('base_final.','final.')
                        assert torch.equal(value,ref['model'][original])
                for param_id in range(6):
                    assert state_hash(raw['optimizer']['state'][param_id])==state_hash(ref['optimizer']['state'][param_id])
                deployed = torch.load(states/(row['key']+'_deployed.pt'),map_location='cpu',weights_only=True)['model']
                assert state_hash(deployed)==row['deployed_model_sha256']
                assert not any('projector' in name or 'base_' in name or 'side_' in name for name in deployed)
                base_width = ref['model']['final.weight'].shape[1]
                assert deployed['trunk.0.weight'].shape[0]==base_width+row['channel_width']
                assert torch.equal(deployed['trunk.0.weight'][:base_width],ref['model']['trunk.0.weight'])
                assert torch.equal(deployed['trunk.2.weight'][:base_width,:base_width],ref['model']['trunk.2.weight'])
                assert torch.count_nonzero(deployed['trunk.2.weight'][:base_width,base_width:])==0
                assert torch.count_nonzero(deployed['trunk.2.weight'][base_width:,:base_width])==0
                states_verified += 1
            if row['mode'].startswith('T'):
                metrics = row['validation'];protected += 1
                assert metrics['prediction_changes']==0 and metrics['max_absolute_healthy_logit_error']<1e-8
                assert all(trace['prediction_changes']==0 for trace in row['diagnostic_trace'])
                assert row['nonlinear_prefix_changed']
                for name in ['acc','nll','ece']:
                    assert abs(metrics['frozen'][name]-metrics['reference'][name])<1e-9
                max_logit = max(max_logit,metrics['max_absolute_healthy_logit_error'])
        assert all(len(values)==1 for values in side_initializations.values())
        replay = json.loads(replay_paths[index].read_text())
        assert replay['complete'] and replay['scored_corpus']=='validation'
        assert replay['record_sha256']==sha(path)
        assert replay['verifier_source_sha256']==sha(ROOT/'repro/scripts/verify_nonlinear_consumers_cpu.py')
        assert len(replay['branches'])==len(record['branches'])
        assert all(row['all_scores_replayed'] and not row['bounds_replayed'] and
                   row['tensor_only_feature_and_fusion_reconstruction'] for row in replay['branches'])
        max_replay = max(max_replay,max(row['maximum_metric_error'] for row in replay['branches']))
    assert (states_verified,endpoints,protected)==(165,87,36)
    candidates = []
    for candidate in choice['candidates']:
        domain,width,dose = candidate['domain'],candidate['width'],candidate['dose']
        rows = [record for record in records if record['domain']==domain]
        comparisons = dict(candidate['comparisons'])
        for name,left,lwidth,right,rwidth in [
            ('unconstrained_side_auxiliary_gain','UJ',width,'UD',width),
            ('constraint_cost_at_matched_capacity','TJ',width,'UJ',width)]:
            pairs = [(endpoint(r,left,lwidth,dose)['validation']['own']['acc'],
                      endpoint(r,right,rwidth,dose)['validation']['own']['acc'],r['splits']['validation_examples']) for r in rows]
            comparisons[name] = describe([left-right for left,right,_ in pairs])
            comparisons[name]['mean'] = float(exact_accuracy_mean(pairs))
        mse_difference = describe([endpoint(r,'TJ',width,dose)['validation']['auxiliary_mse']-
                                   endpoint(r,'TD',width,dose)['validation']['auxiliary_mse'] for r in rows])
        mse_ratio = describe([endpoint(r,'TJ',width,dose)['validation']['auxiliary_mse']/
                              endpoint(r,'TD',width,dose)['validation']['auxiliary_mse'] for r in rows])
        state_path = paths[records.index(rows[0])].parent/(paths[records.index(rows[0])].stem+'_states')
        first = endpoint(rows[0],'TJ',width,dose)
        deployed = torch.load(state_path/(first['key']+'_deployed.pt'),weights_only=True)['model']
        original = torch.load(state_path/'reference.pt',weights_only=True)['model']
        count = lambda tensors:sum(value.numel() for name,value in tensors.items() if name.startswith(('trunk.','final.')))
        fused_count,ordinary_count = count(deployed),count(original)
        candidates.append(dict(candidate,comparisons=comparisons,validation_auxiliary_mse_difference=mse_difference,
            validation_auxiliary_mse_ratio=mse_ratio,encoder_parameter_ratio=fused_count/ordinary_count,
            fused_encoder_parameters=fused_count,ordinary_encoder_parameters=ordinary_count,
            fused_encoder_parameter_bytes=fused_count*8,empirical_full_data_latency_measured=False))
    queues = [json.loads((BASE/folder/'queue.json').read_text()) for folder in ['nonlinear_development','nonlinear_validation_replay']]
    assert all(q['finished'] and not q['failed'] and not q['running'] for q in queues)
    verification = dict(development_records=6,states_verified=states_verified,saved_validation_consumers_replayed=endpoints,
        protected_validation_consumers=protected,maximum_replay_metric_error=max_replay,
        max_guarded_logit_error=max_logit,max_fusion_feature_error=max_fusion,
        source_plan_shared_states_masks_orders_and_frozen_base_Adam_verified=True,
        test_outcomes_evaluated=False,confirmation_tasks=0,exact_count_gate_verified=True,
        peak_aggregate_rss_bytes=max(q['peak_aggregate_rss_bytes'] for q in queues),
        sum_job_wall_seconds=sum(job['wall_s'] for q in queues for job in q['completed']))
    summary = dict(plan_sha256=sha(plan),choice_sha256=sha(choice_path),candidates=candidates,
        selected=choice['selected'],verification=verification,
        interpretation='Development-only conditional seed effects; neither task qualifies; no independent test confirmation or full-data latency claim.',
        input_record_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in paths})
    (BASE/'nonlinear_interface_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
    (BASE/'nonlinear_interface_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8',newline='\n')
    macros = {}
    for domain,label in [('radioml','Rml'),('har','Har')]:
        selected = choice['selected'][domain]
        macros['Nonlinear'+label+'Width'] = str(selected['width'])
        for name,suffix in [('ordinary_auxiliary_gain','JointGain'),('compatible_auxiliary_gain','AuxGain'),
                            ('compatible_total_gain','TotalGain'),('compatible_vs_joint','JointCost')]:
            value = selected['comparisons'][name]
            for part,number in [('',value['mean']),('Low',value['mean_ci95'][0]),('High',value['mean_ci95'][1])]:
                macros['Nonlinear'+label+suffix+part] = '%.2f'%(100*number)
    for name,value in [('NonlinearLogitError',max_logit),('NonlinearFusionError',max_fusion)]:
        coefficient,exponent=('%.1e'%value).split('e')
        macros[name] = coefficient+r'\times10^{'+str(int(exponent))+'}'
    (ROOT/'paper/nonlinear_numbers.tex').write_text('% Generated from the complete validation grid; no test confirmation.\n'+
        '\n'.join('\\newcommand{\\'+name+'}{'+value+'}' for name,value in macros.items())+'\n',encoding='utf-8',newline='\n')
    table = [r'\begin{table}[t]',r'\centering',
        r'\caption{Complete nonlinear-channel development grid: three seeds per task, validation only. D/J are full-network branches; TD/TJ are constrained side branches. All entries are percentage-point means. The score is $\min\{\overline{\mathrm{J-D}},\overline{\mathrm{TJ-TD}},\overline{\mathrm{TJ-D}},\overline{\mathrm{TJ-J}}+0.5\}$. Confirmation requires strictly positive score, evaluated from integer correct counts. No candidate qualifies. HAR uses a new 1\% fitting-label regime, SNR 20\%.}',
        r'\label{tab:nonlinear_development}',r'\small',r'\begin{tabular}{lrrrrrrrr}',r'\toprule',
        r'Task & $k$ & $\alpha$ & J--D & TJ--TD & TJ--D & TJ--J & score & confirm \\',r'\midrule']
    for item in candidates:
        values = [100*item['comparisons'][name]['mean'] for name in
                  ['ordinary_auxiliary_gain','compatible_auxiliary_gain','compatible_total_gain','compatible_vs_joint']]
        table.append(('%s & %d & %g & '%('SNR' if item['domain']=='radioml' else 'HAR',item['width'],item['dose']))+
                     ' & '.join('$%+.2f$'%value for value in values)+' & $%.2f$ & no'% (100*item['score'])+r' \\')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/'paper/nonlinear_development_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(verification=verification,selected=choice['selected'],
        parameter_ratios=[dict(domain=c['domain'],width=c['width'],ratio=c['encoder_parameter_ratio']) for c in candidates]),indent=2))


if __name__=='__main__':main()
