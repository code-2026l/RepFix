"""Verify the complete frozen-recipe study and evaluate its declared criteria."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from paired_readout_continuation import state_hash
from consumer_risk import upper_total_disagreements
from summarize_cpu_evidence import describe

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'repro/results/CPUCONT'
MODES=['detach','joint','guarded_detach','guarded_joint']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan=ROOT/'docs/plans/2026-11-09-prospective-benefit-compatibility.md'
    choice_path=BASE/'benefit_development_choice.json'
    choice=json.loads(choice_path.read_text())
    raw_choice=choice_path.read_bytes().replace(b'\r\n',b'\n')
    choice_digests={hashlib.sha256(raw_choice).hexdigest(),
                    hashlib.sha256(raw_choice.replace(b'\n',b'\r\n')).hexdigest()}
    selected=choice['selected'];plan_sha=sha(plan)
    dev_paths=sorted((BASE/'benefit_development').glob('l???.json'))
    conf_paths=sorted((BASE/'benefit_confirmation').glob('s???.json'))
    assert len(dev_paths)==6 and len(conf_paths)==10
    rows=[json.loads(p.read_text()) for p in conf_paths]
    assert [r['seed'] for r in rows]==list(range(300,310))
    state_count=0
    for path in dev_paths+conf_paths:
        r=json.loads(path.read_text())
        assert r['complete'] and r['device']=='cpu' and r['cuda_visibility']=='' and r['torch_threads']==1
        assert r['plan_sha256']==plan_sha==choice['plan_sha256']
        for name,digest in r['source_sha256'].items():
            assert sha(ROOT/'repro/scripts'/name)==digest
        states=path.parent/(path.stem+'_states')
        reference=torch.load(states/'reference.pt',map_location='cpu',weights_only=True)
        assert state_hash(reference['model'])==r['reference_model_sha256']
        assert state_hash(reference['optimizer'])==r['reference_optimizer_sha256']
        w,sd=reference['head']['weight'],reference['sd'].reshape(-1)
        a=w/sd;p=reference['projector']
        assert float((a@p).norm()/a.norm())<1e-10 and float((p@p-p).norm())<1e-10
        assert len({b['minibatch_order_sha256'] for b in r['branches']})==1
        state_count+=1
        for b in r['branches']:
            assert b['initial_model_sha256']==r['reference_model_sha256']
            assert b['initial_optimizer_sha256']==r['reference_optimizer_sha256']
            saved=torch.load(states/('%s_%g.pt'%(b['mode'],b['alpha'])),map_location='cpu',weights_only=True)
            assert state_hash(saved['model'])==b['final_model_sha256']
            assert state_hash(saved['optimizer'])==b['final_optimizer_sha256']
            assert all(t.device.type=='cpu' for t in saved['model'].values())
            if b['mode'].startswith('guarded'):
                for split in ['validation','test']:
                    if split not in b:continue
                    metric=b[split]
                    assert metric['max_absolute_healthy_logit_error']<1e-8 and metric['prediction_changes']==0
                    for key in ['acc','nll','ece']:
                        assert abs(metric['frozen'][key]-metric['reference'][key])<1e-9
                assert all(t['prediction_changes']==0 for t in b['diagnostic_trace'])
            state_count+=1
        if r['phase']=='development':
            assert not r['splits']['test_metrics_computed'] and all('test' not in b for b in r['branches'])
        else:
            assert r['selection_sha256'] in choice_digests
            assert r['domain']==selected['domain'] and len(r['branches'])==4
            assert all(b['alpha']==selected['dose'] for b in r['branches'])
            assert r['consumer_bank']['candidates']==40 and not r['consumer_bank']['used_for_training']
            for b in r['branches']:
                c=b['test']['consumer_audit']
                total=upper_total_disagreements(c['population_examples'],c['monitor_examples'],
                                               c['observed_disagreements'],.05/40)
                assert total==c['upper_total_disagreements']
                assert abs(c['upper_remaining_disagreement']-(total-c['observed_disagreements'])/
                           (c['population_examples']-c['monitor_examples']))<1e-12
    assert state_count==104
    def row(r,mode):return next(b for b in r['branches'] if b['mode']==mode)
    comparisons={name:describe([row(r,left)['test']['own']['acc']-row(r,right)['test']['own']['acc'] for r in rows])
        for name,left,right in [('ordinary_auxiliary_gain','joint','detach'),
            ('guarded_auxiliary_gain','guarded_joint','guarded_detach'),
            ('guarded_gain_over_ordinary_detach','guarded_joint','detach'),
            ('guard_cost_in_accuracy','guarded_joint','joint')]}
    criteria={name:comparisons[name]['mean_ci95'][0]>0 for name in
        ['ordinary_auxiliary_gain','guarded_auxiliary_gain','guarded_gain_over_ordinary_detach']}
    criteria['noninferior_within_half_percentage_point']=comparisons['guard_cost_in_accuracy']['mean_ci95'][0]>-.005
    guarded=[row(r,m)['test'] for r in rows for m in ['guarded_detach','guarded_joint']]
    criteria['guarded_consumer_preserved']=max(t['max_absolute_healthy_logit_error'] for t in guarded)<1e-8 and all(t['prediction_changes']==0 for t in guarded)
    arms={}
    for mode in MODES:
        endpoints=[row(r,mode)['test'] for r in rows]
        arms[mode]={k:describe([e[k]['acc'] for e in endpoints]) for k in ['own','frozen','reference']}
        arms[mode]['frozen_nll']=describe([e['frozen']['nll'] for e in endpoints])
        arms[mode]['frozen_ece']=describe([e['frozen']['ece'] for e in endpoints])
        arms[mode]['reference_nll']=describe([e['reference']['nll'] for e in endpoints])
        arms[mode]['reference_ece']=describe([e['reference']['ece'] for e in endpoints])
        arms[mode].update({k:describe([e[k] for e in endpoints]) for k in ['auxiliary_mse','relative_trace','disagreement','max_absolute_healthy_logit_error']})
        arms[mode]['max_logit_error']=max(e['max_absolute_healthy_logit_error'] for e in endpoints)
        arms[mode]['certified_count']=sum(e['consumer_audit']['certified'] for e in endpoints)
        arms[mode]['above_budget_certifications']=sum(e['consumer_audit']['certified'] and e['consumer_audit']['remainder_loss_increase']>.05 for e in endpoints)
        arms[mode]['bound']=describe([e['consumer_audit']['upper_remaining_disagreement'] for e in endpoints])
        triggers=[next((t['epoch'] for t in row(r,mode)['diagnostic_trace'] if t['triggered']),None) for r in rows]
        arms[mode]['first_anchor_trigger_epochs']=triggers
    latency=describe([r['inference_latency']['guarded_two_encoders']['median_seconds']/
                      r['inference_latency']['ordinary']['median_seconds'] for r in rows])
    mse_ratio=describe([row(r,'guarded_joint')['test']['auxiliary_mse']/
                        row(r,'guarded_detach')['test']['auxiliary_mse'] for r in rows])
    replay_paths=sorted((BASE/'benefit_replay').glob('z???.json'))
    assert len(replay_paths)==10
    replays=[json.loads(p.read_text()) for p in replay_paths]
    for index,replay in enumerate(replays):
        assert replay['complete'] and replay['device']=='cpu' and len(replay['branches'])==4
        assert replay['record_sha256']==sha(conf_paths[index])
        assert replay['verifier_source_sha256']==sha(ROOT/'repro/scripts/verify_benefit_consumers_cpu.py')
        assert all(b['all_scores_replayed'] and b['bounds_replayed'] and b['maximum_metric_error']<1e-7 for b in replay['branches'])
    summary=dict(protocol='independent ten-seed confirmation of validation-frozen real auxiliary recipe',
        selected=selected,seeds=[r['seed'] for r in rows],plan_sha256=plan_sha,
        selection_sha256_observed=rows[0]['selection_sha256'],comparisons=comparisons,criteria=criteria,
        all_predeclared_criteria_pass=all(criteria.values()),arms=arms,inference_time_ratio=latency,
        guarded_auxiliary_mse_ratio=mse_ratio,independent_saved_consumers_replayed=40,
        protected_rank=describe([r['projector']['protected_rank'] for r in rows]),
        free_dimensions=describe([r['projector']['free_dimensions'] for r in rows]),
        uncertainty='Paired two-sided descriptive 95% t intervals over ten seeds at fixed data/split/recipe.',
        input_record_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in dev_paths+conf_paths})
    (BASE/'benefit_confirmation_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
    queues=[json.loads((BASE/f/'queue.json').read_text()) for f in ['benefit_development','benefit_confirmation','benefit_replay']]
    assert all(q['finished'] and not q['failed'] and not q['running'] for q in queues)
    verification=dict(development_records=6,confirmation_records=10,model_optimizer_states_verified=state_count,
        sources_plan_choice_and_pairing_verified=True,development_test_outcomes_excluded=True,
        guarded_outputs_preserved=True,independent_consumer_bank_candidates=40,
        independent_saved_consumers_replayed=40,
        criteria=criteria,all_predeclared_criteria_pass=all(criteria.values()),
        aggregate_peak_rss_bytes=max(q['peak_aggregate_rss_bytes'] for q in queues),
        sum_job_wall_seconds=sum(j['wall_s'] for q in queues for j in q['completed']))
    (BASE/'benefit_confirmation_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8',newline='\n')
    macros={}
    for mode,suffix in [('detach','Detach'),('joint','Joint'),('guarded_detach','GuardDetach'),('guarded_joint','GuardJoint')]:
        macros['BenefitOwn'+suffix]='%.3f'%arms[mode]['own']['mean']
        macros['BenefitFrozen'+suffix]='%.3f'%arms[mode]['frozen']['mean']
        macros['BenefitCertified'+suffix]=str(arms[mode]['certified_count'])
    for name,suffix in [('ordinary_auxiliary_gain','JointGain'),('guarded_auxiliary_gain','GuardGain'),
                        ('guarded_gain_over_ordinary_detach','TotalGain'),('guard_cost_in_accuracy','Cost')]:
        value=comparisons[name]
        macros['Benefit'+suffix]='%.2f'%(100*value['mean'])
        macros['Benefit'+suffix+'Low']='%.2f'%(100*value['mean_ci95'][0])
        macros['Benefit'+suffix+'High']='%.2f'%(100*value['mean_ci95'][1])
    macros['BenefitLatencyRatio']='%.2f'%latency['mean']
    value=max(t['max_absolute_healthy_logit_error'] for t in guarded)
    coefficient,exponent=('%.1e'%value).split('e')
    macros['BenefitLogitError']=coefficient+r'\times10^{'+str(int(exponent))+'}'
    macros['BenefitHealthyECE']='%.3f'%arms['joint']['reference_ece']['mean']
    macros['BenefitJointECE']='%.3f'%arms['joint']['frozen_ece']['mean']
    macros['BenefitMseRatio']='%.3f'%mse_ratio['mean']
    (ROOT/'paper/benefit_numbers.tex').write_text('% Generated from all ten fixed confirmation seeds.\n'+
        '\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n',encoding='utf-8',newline='\n')
    lines=[r'\begin{table}[t]',r'\centering',
        r'\caption{Independent confirmation of the validation-selected SNR auxiliary: 20\% of fitting primary labels, dose $1$, ten new seeds. D/J denote detachment/joint training; G denotes the reference-anchored readout-nullspace constraint. Own accuracy uses the current task head. The same healthy consumer is retained for frozen accuracy. Means with population seed SD; bounds use a separate fixed 40-consumer bank and a five-point relative-error budget. The guarded representation requires both encoder evaluations at inference.}',
        r'\label{tab:benefit_confirmation}',r'\small',r'\begin{tabular}{lrrrrr}',r'\toprule',
        r'Arm & own accuracy & frozen accuracy & frozen NLL & ECE & certified \\',r'\midrule']
    for mode,label in [('detach','D'),('joint','J'),('guarded_detach','GD'),('guarded_joint','GJ')]:
        arm=arms[mode]
        cells=[r'$%.3f\pm%.3f$'%(arm[k]['mean'],arm[k]['sd']) for k in ['own','frozen']]
        cells += [r'$%.3f$'%arm['frozen_nll']['mean'],r'$%.3f$'%arm['frozen_ece']['mean'],str(arm['certified_count'])+'/10']
        lines.append(label+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/'paper/benefit_confirmation_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    contrast=[r'\begin{table}[t]',r'\centering',
        r'\caption{All predeclared primary-accuracy contrasts in percentage points, ten independent confirmation seeds. Paired two-sided 95\% descriptive t intervals are conditional on the fixed dataset/split/recipe. GD controls the guarded architecture itself; GJ--GD isolates its auxiliary benefit. Noninferiority was declared at $-0.5$ point, not selected from the outcomes.}',
        r'\label{tab:benefit_contrasts}',r'\small',r'\begin{tabular}{lrrr}',r'\toprule',
        r'Contrast & mean (pp) & paired 95\% interval & required lower limit \\',r'\midrule']
    for name,label,limit in [('ordinary_auxiliary_gain','J--D','0'),('guarded_auxiliary_gain','GJ--GD','0'),
                            ('guarded_gain_over_ordinary_detach','GJ--D','0'),('guard_cost_in_accuracy','GJ--J','$-0.50$')]:
        value=comparisons[name]
        contrast.append(r'%s & $%.2f$ & $[%.2f,\,%.2f]$ & %s \\'%(label,100*value['mean'],
                        100*value['mean_ci95'][0],100*value['mean_ci95'][1],limit))
    contrast += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/'paper/benefit_contrasts_table.tex').write_text('\n'.join(contrast)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(criteria=criteria,all_pass=all(criteria.values()),comparisons=comparisons,
        arms={m:dict(own=arms[m]['own']['mean'],frozen=arms[m]['frozen']['mean'],certified=arms[m]['certified_count']) for m in MODES},
        max_guarded_logit_error=max(t['max_absolute_healthy_logit_error'] for t in guarded),latency_ratio=latency['mean']),indent=2))


if __name__=='__main__':main()
