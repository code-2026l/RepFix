"""Local paired analysis of support, scope, and matched serving evidence."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'repro/results/CPUCONT'

def interval(values):
    a=np.asarray(values,float);n=len(a);m=float(a.mean())
    half=0. if n==1 else float(t.ppf(.975,n-1)*a.std(ddof=1)/np.sqrt(n))
    return dict(mean=m,ci95=[m-half,m+half],values=a.tolist(),n=n)

def records(folder,pattern='*.json'):
    return [json.loads(p.read_text()) for p in sorted((BASE/folder).glob(pattern)) if not p.name.endswith('.partial.json')]

def utility(folder,choice_file):
    rows=records(folder);selected=json.loads((BASE/choice_file).read_text())['backbones'];out={}
    for arch,c in selected.items():
        if c['selected'] is None:continue
        rr=[r for r in rows if r['arch']==arch];assert len(rr)==10 and {r['seed'] for r in rr}==set(range(3810,3820))
        assert all(r['complete'] and r['selection_sha256']==hashlib.sha256((BASE/choice_file).read_bytes()).hexdigest() for r in rr)
        by=[{b['mode']:b for b in r['branches']} for r in rr];mode=c['selected'];modes=list(by[0])
        absolute={m:interval([b[m]['test']['own']['acc'] for b in by]) for m in modes}
        differences={m:interval([b[mode]['test']['own']['acc']-(r['healthy']['acc'] if m=='healthy' else b[m]['test']['own']['acc']) for b,r in zip(by,rr)]) for m in ['healthy',*[k for k in modes if k!=mode]]}
        contracts=[v for b in by for v in b[mode]['test']['contracts'].values()]
        out[arch]=dict(selected=mode,absolute=absolute,differences=differences,healthy=rr[0]['healthy'],seeds=[r['seed'] for r in rr],
            primary_gate=all(differences[m]['ci95'][0]>=.02 for m in ['healthy','F']) and all(differences[m]['ci95'][0]>=-.005 for m in ['N','H']),
            consumer_checks=len(contracts),maximum_old_disagreement=max(v['disagreement'] for v in contracts),
            certified=sum(v['certificate']['certified'] for v in contracts),
            maximum_remaining_bound=max(v['certificate']['upper_remaining_disagreement'] for v in contracts),
            maximum_legacy_logit_error=max(b[mode]['test']['maximum_legacy_logit_error'] for b in by),
            interval_scope='continuation variability on one fixed reference, label mask and split')
    return out

def main():
    factorial=records('support_rank_factorial');assert len(factorial)==3
    cells={}
    for family in ['sparse','dense']:
        for rank in [8,12]:
            pairs=[{b['mode']:b for b in r['branches'] if b['family']==family and b['second_rank']==rank} for r in factorial]
            cells[family+'/'+str(rank)]=dict(detach=interval([p['detach']['validation']['own']['acc'] for p in pairs]),
                joint=interval([p['joint']['validation']['own']['acc'] for p in pairs]),
                paired_gain=interval([p['joint']['validation']['own']['acc']-p['detach']['validation']['own']['acc'] for p in pairs]),
                update_parameters=pairs[0]['joint']['update_parameters'])
    contrasts={}
    for r in [8,12]:contrasts['dense_minus_sparse_r'+str(r)]=interval(np.array(cells['dense/'+str(r)]['joint']['values'])-cells['sparse/'+str(r)]['joint']['values'])
    for family in ['sparse','dense']:contrasts[family+'_r12_minus_r8']=interval(np.array(cells[family+'/12']['joint']['values'])-cells[family+'/8']['joint']['values'])
    summary=dict(complete=True,analysis='local',source_sha256={'analyze_frontier_extension.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        factorial=dict(seeds=[r['seed'] for r in factorial],first_rank_ceiling=16,cells=cells,contrasts=contrasts,
            interpretation='first-map family also changes orientation, bias support and degrees of freedom; rank alone is not identified'),
        matched_dense_macs=dict(current=256*64+64*64+11*64,extra=16*256+64*16+64*64+2*12*64+11*64))
    summary['matched_dense_macs']['extra_fraction']=summary['matched_dense_macs']['extra']/summary['matched_dense_macs']['current']
    summary['nonlinear_suffix']=utility('nonlinear_suffix_confirmation','nonlinear_suffix_choice.json')
    if (BASE/'two_block_choice.json').exists() and len(records('two_block_confirmation'))==10:summary['two_block']=utility('two_block_confirmation','two_block_choice.json')
    source_names=['support_rank_factorial','late_vision_pilot','late_primary_audit','nonlinear_suffix_pilot','nonlinear_suffix_development','nonlinear_suffix_confirmation','token_suffix_pilot','two_block_pilot','two_block_development','two_block_confirmation']
    summary['evidence_sha256']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for name in source_names for p in sorted((BASE/name).glob('*.json'))}
    (BASE/'frontier_extension_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
    macros={};r=summary['nonlinear_suffix']['resnet18']
    for key,label in [('healthy','Healthy'),('F','Head'),('N','NonlinearHead'),('H','RichHead'),('L','LastMap')]:
        q=r['differences'][key]
        for suffix,value in [('',q['mean']),('Low',q['ci95'][0]),('High',q['ci95'][1])]:macros['SuffixRNGain'+label+suffix]=f'{100*value:.2f}'
    macros.update(SuffixRNAcc=f"{100*r['absolute']['U']['mean']:.2f}",SuffixRNBound=f"{r['maximum_remaining_bound']:.4f}")
    if 'two_block' in summary:
        v=summary['two_block']['vit_b32']
        for key,label in [('healthy','Healthy'),('F','Head'),('N','NonlinearHead'),('H','RichHead'),('M','FinalMLP')]:
            q=v['differences'][key]
            for suffix,value in [('',q['mean']),('Low',q['ci95'][0]),('High',q['ci95'][1])]:macros['SuffixViTGain'+label+suffix]=f'{100*value:.2f}'
        macros.update(SuffixViTAcc=f"{100*v['absolute']['U']['mean']:.2f}",SuffixViTBound=f"{v['maximum_remaining_bound']:.4f}")
    (ROOT/'paper/frontier_numbers.tex').write_text('% Generated locally from recorded endpoint metrics.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()),encoding='utf-8',newline='\n')
    lines=['\\begin{table}[t]','\\centering','\\caption{Matched SNR support/rank factorial, three development seeds. Both first-map families have rank ceiling 16; dense updates also have more parameters and a larger bias support. Accuracy is in percent; the gain is paired J--D, in points. These are validation outcomes, not fresh confirmation.}','\\label{tab:support_factorial}','\\begin{tabular}{lrrrr}','\\toprule','First map / second rank & Update params & D & J & J--D \\\\','\\midrule']
    for key,c in cells.items():lines.append(f"{key} & {c['update_parameters']:,} & {100*c['detach']['mean']:.2f} & {100*c['joint']['mean']:.2f} & {100*c['paired_gain']['mean']:+.2f} \\\\")
    lines+=['\\bottomrule','\\end{tabular}','\\end{table}']
    (ROOT/'paper/support_factorial_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    lines=['\\begin{table}[t]','\\centering','\\caption{ResNet-18/CIFAR-100 test utility after the publicly frozen three-seed development choice. Ten fresh continuation seeds share one healthy reference and split. Accuracies are in percent; U--control intervals are paired 95\\% intervals in points. H is a stronger frozen-feature readout, so its improvement does not establish encoder learning.}','\\label{tab:suffix_confirmation}','\\begin{tabular}{lrr}','\\toprule','Arm & Test accuracy & U--control [95\\% interval] \\\\','\\midrule']
    labels=dict(F='Head only (F)',N='Residual head (N)',H='Richer frozen probe (H)',L='Final map (L)',U='Complete nonlinear suffix (U)')
    for mode,q in r['absolute'].items():
        difference='--'
        if mode!='U':
            v=r['differences'][mode];difference=f"{100*v['mean']:+.2f} [{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}]"
        lines.append(f"{labels[mode]} & {100*q['mean']:.2f} & {difference} \\\\")
    lines+=['\\bottomrule','\\end{tabular}','\\end{table}']
    (ROOT/'paper/suffix_confirmation_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    if 'two_block' in summary:
        v=summary['two_block']['vit_b32']
        lines=['\\begin{table}[t]','\\centering','\\caption{ViT-B/32/CIFAR-100 test utility after the public two-block choice. Ten fresh continuation seeds share one reference and split. Accuracy is in percent; U--control intervals are paired 95\\% intervals in points.}','\\label{tab:two_block_confirmation}','\\begin{tabular}{lrr}','\\toprule','Arm & Test accuracy & U--control [95\\% interval] \\\\','\\midrule']
        labels=dict(F='Head only (F)',N='Residual head (N)',H='Richer frozen probe (H)',M='Final MLP (M)',U='Two complete blocks (U)')
        for mode,q in v['absolute'].items():
            difference='--'
            if mode!='U':
                d=v['differences'][mode];difference=f"{100*d['mean']:+.2f} [{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}]"
            lines.append(f"{labels[mode]} & {100*q['mean']:.2f} & {difference} \\\\")
        lines+=['\\bottomrule','\\end{tabular}','\\end{table}']
        (ROOT/'paper/two_block_confirmation_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(factorial=contrasts,resnet18=r['differences'],primary_gate=r['primary_gate'],two_block=summary.get('two_block')),indent=2))

if __name__=='__main__':main()
