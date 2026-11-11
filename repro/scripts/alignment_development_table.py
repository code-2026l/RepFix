"""Render every fixed development candidate, including failed policy choices."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'repro/results/CPUCONT'
rows=[json.loads(p.read_text()) for p in sorted((base/'alignment_development').glob('a???.json'))]
choice=json.loads((base/'alignment_development_choice.json').read_text())
out=[r'\begin{table}[H]',r'\centering',
    r'\caption{Complete fixed development grid, three seeds per corpus. Primary differences are percentage points relative to detachment; MSE is auxiliary error relative to detachment. Means use validation only. A dagger marks the coefficient selected by mean disagreement before utility is checked; all other coefficients remain visible.}',
    r'\label{tab:alignment_development}',r'\footnotesize',r'\begin{tabular}{llrrrr}',r'\toprule',
    r'Corpus & target / reader & $\lambda$ & primary & MSE / D & changed \\',r'\midrule']
for domain in ['har','radioml']:
    group=[r for r in rows if r['domain']==domain]
    assert len(group)==3
    for mode,label in [('feature_map','Feature map'),('logit_map','Logit map'),('direct_ols','Direct + OLS-fp32')]:
        lams=sorted({b['lam'] for b in group[0]['branches'] if b['mode']==mode})
        for lam in lams:
            candidates=[next(b for b in r['branches'] if b['mode']==mode and b['lam']==lam) for r in group]
            primary=np.mean([b['validation']['own']['acc']-r['original_controls']['detach']['own']['acc'] for b,r in zip(candidates,group)])*100
            ratio=np.mean([b['validation']['auxiliary_mse']/r['original_controls']['detach']['auxiliary_mse'] for b,r in zip(candidates,group)])
            changed=np.mean([b['validation']['disagreement'] for b in candidates])*100
            mark=r'^{\dagger}' if lam==choice['domains'][domain][mode]['selected_lambda'] else ''
            out.append(('HAR' if domain=='har' else 'SNR')+' & '+label+f' & ${lam:g}{mark}$ & ${primary:+.2f}$ & ${ratio:.3f}$ & ${changed:.2f}\\%$'+r' \\')
    out.append(r'\midrule' if domain=='har' else r'\bottomrule')
out.extend([r'\end{tabular}',r'\end{table}'])
(ROOT/'paper/alignment_development_table.tex').write_text('\n'.join(out)+'\n',encoding='utf-8',newline='\n')
