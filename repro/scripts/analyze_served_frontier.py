"""Local paired statistics for actual CPU implementations, without selection."""
import hashlib,json
from pathlib import Path
import numpy as np
from analyze_frontier_extension import interval

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'repro/results/CPUCONT'
def load(folder):return [json.loads(p.read_text()) for p in sorted((BASE/folder).glob('c???.json'))]
def fmt(q,scale=1,digits=2):return f"{scale*q['mean']:.{digits}f} [{scale*q['ci95'][0]:.{digits}f}, {scale*q['ci95'][1]:.{digits}f}]"

def main():
    native=load('precision_native_confirmation');timing=load('precision_serving_confirmation')
    assert len(native)==len(timing)==20 and all(r['complete'] for r in native+timing)
    choice=json.loads((BASE/'reference_precision_choice.json').read_text());choice_sha=hashlib.sha256((BASE/'reference_precision_choice.json').read_bytes()).hexdigest()
    assert all(r['choice_sha256']==choice_sha for r in native+timing)
    out=dict(complete=True,analysis='local',source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},backbones={})
    for arch in ['resnet18','vit_b32']:
        rr=[r for r in native if r['arch']==arch];tt=[r for r in timing if r['arch']==arch]
        assert {r['seed'] for r in rr}=={r['seed'] for r in tt}==set(range(3810,3820))
        modes=list(rr[0]['audits']);utility={}
        for kind in ['chosen','full_precision']:
            assert all(r['serving']['U'][kind]['1']['own']['acc']==r['serving']['U'][kind]['128']['own']['acc'] for r in rr)
            values=[r['serving']['U'][kind]['128']['own']['acc'] for r in rr]
            controls={m:[r['healthy']['acc'] if m=='healthy' else r['audits'][m]['own']['acc'] for r in rr] for m in ['healthy',*[m for m in modes if m!='U']]}
            diffs={m:interval(np.asarray(values)-np.asarray(c)) for m,c in controls.items()}
            contracts=[v for r in rr for b in ['1','128'] for v in r['serving']['U'][kind][b]['contracts'].values()]
            utility[kind]=dict(absolute=interval(values),differences=diffs,
                primary_gate=all(diffs[m]['ci95'][0]>=.02 for m in ['healthy','F']) and all(diffs[m]['ci95'][0]>=-.005 for m in ['N','H']),
                consumer_checks=len(contracts),maximum_disagreement=max(v['disagreement'] for v in contracts),
                certified=sum(v['certificate']['certified'] for v in contracts),maximum_remaining_bound=max(v['certificate']['upper_remaining_disagreement'] for v in contracts),
                maximum_logit_error=max(r['serving']['U'][kind][b]['maximum_legacy_logit_error'] for r in rr for b in ['1','128']),
                maximum_absolute_accuracy_change_from_cached=max(abs(r['serving']['U'][kind][b]['cached_prefix_accuracy_difference']) for r in rr for b in ['1','128']))
        costs={}
        for batch,limit in [('1',1.20),('128',1.10)]:
            fixed=choice['backbones'][arch]['modes']['U']['batches'][batch];values=[r['serving']['U']['timings'][batch]['medians_seconds'] for r in tt]
            current=np.asarray([v[fixed['current']] for v in values]);rows={}
            exports=[r['serving']['U']['serialized_program_bytes'] for r in tt]
            for kind,key in [('chosen','protected'),('full_precision','full_precision_protected')]:
                protected=np.asarray([v[fixed[key]] for v in values]);ratio=interval(protected/current)
                rows[kind]=dict(backend=fixed[key],protected_seconds=interval(protected),ratio=ratio,
                    exported_program_bytes=interval([e[fixed[key]] for e in exports]),
                    mean_within_budget=ratio['mean']<=limit,upper_ci_within_budget=ratio['ci95'][1]<=limit)
            costs[batch]=dict(current_backend=fixed['current'],current_seconds=interval(current),exported_current_program_bytes=interval([e[fixed['current']] for e in exports]),budget=limit,**rows)
        out['backbones'][arch]=dict(utility=utility,cost=costs,scope='one healthy reference, one label mask and split; ten fresh continuation seeds')
    small=json.loads((BASE/'native_frontier_confirmation/c000.json').read_text());out['small_networks']={}
    for domain in ['har','radioml']:
        rows=[r for r in small['branches'] if r['domain']==domain and r['mode']=='joint'];assert len(rows)==10
        cc={}
        for batch,prefix,limit in [('1','native',1.20),('1024','torchscript',1.10)]:
            values=[r['timings'][batch]['medians_seconds'] for r in rows];current=np.asarray([v[prefix+'_current'] for v in values]);protect=np.asarray([v[prefix+'_replay'] for v in values])
            cc[batch]=dict(current_seconds=interval(current),protected_seconds=interval(protect),ratio=interval(protect/current),budget=limit)
        contracts=[v for r in rows for audit in r['audits'].values() for v in audit['contracts'].values()]
        out['small_networks'][domain]=dict(cost=cc,maximum_disagreement=max(v['disagreement'] for v in contracts),consumer_checks=len(contracts))
    families=['precision_native_confirmation','precision_serving_confirmation','native_frontier_confirmation','suffix_repair_development','precision_serving_development_final','whole_amp_utility_development']
    out['evidence_sha256']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for f in families for p in sorted((BASE/f).glob('*.json'))}
    (BASE/'served_frontier_summary.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n')
    macros={}
    for domain,code in [('har','Har'),('radioml','Snr')]:
        for batch,label in [('1','One'),('1024','Batch')]:macros['Small'+code+'Cost'+label]=f"{out['small_networks'][domain]['cost'][batch]['ratio']['mean']:.3f}"
    for arch,code in [('resnet18','RN'),('vit_b32','ViT')]:
        x=out['backbones'][arch];u=x['utility']['chosen']
        for control,label in [('healthy','Healthy'),('F','Head'),('N','NonlinearHead'),('H','RichHead')]:
            q=u['differences'][control]
            for suffix,value in [('',q['mean']),('Low',q['ci95'][0]),('High',q['ci95'][1])]:macros['Served'+code+'Gain'+label+suffix]=f'{100*value:.2f}'
        macros['Served'+code+'Acc']=f"{100*u['absolute']['mean']:.2f}"
        macros['Served'+code+'Disagreement']=f"{100*u['maximum_disagreement']:.2f}"
        macros['Served'+code+'Bound']=f"{u['maximum_remaining_bound']:.4f}"
        for b,label in [('1','One'),('128','Batch')]:
            for kind,tag in [('chosen',''),('full_precision','Full')]:macros['Served'+code+tag+'Cost'+label]=f"{x['cost'][b][kind]['ratio']['mean']:.3f}"
    (ROOT/'paper/served_frontier_numbers.tex').write_text('% Generated locally from actual serving outputs and isolated timing.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()),encoding='utf-8',newline='\n')
    lines=['\\begin{table}[t]','\\centering','\\caption{Actual selected CPU serving, ten fresh continuation seeds. Ratios use an independently optimized fixed FP32 current-only function. Intervals describe seed-paired timing ratios, not independent timing repeats. Mixed precision changes only the old suffix; full precision is a separate fixed control. Forward latency excludes preprocessing and compilation.}','\\label{tab:served_frontier}','\\begin{tabular}{llrrr}','\\toprule','Backbone / batch & Reference & Current (ms) & Protected (ms) & Ratio [95\\%] \\\\','\\midrule']
    for arch,label in [('resnet18','ResNet-18'),('vit_b32','ViT-B/32')]:
        for batch in ['1','128']:
            c=out['backbones'][arch]['cost'][batch]
            for kind,title in [('full_precision','FP32'),('chosen','Selected')]:lines.append(f"{label} / {batch} & {title} & {1000*c['current_seconds']['mean']:.3f} & {1000*c[kind]['protected_seconds']['mean']:.3f} & {fmt(c[kind]['ratio'],1,3)} \\\\")
    lines+=['\\bottomrule','\\end{tabular}','\\end{table}']
    (ROOT/'paper/served_frontier_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({a:dict(utility={m:fmt(q,100) for m,q in x['utility']['chosen']['differences'].items()},cost={b:{k:fmt(v[k]['ratio'],1,3) for k in ['chosen','full_precision']} for b,v in x['cost'].items()}) for a,x in out['backbones'].items()},indent=2))

if __name__=='__main__':main()
