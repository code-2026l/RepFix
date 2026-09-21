"""Aggregate seeds within temporal folds before descriptive paired comparisons.

Ten temporal folds can still be dependent. These intervals are sensitivity
summaries, not population guarantees or tests based on 100 independent units.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from paired_stats import wilcoxon_exact, holm


def main():
    root=Path(__file__).resolve().parents[2]
    matrices={};sources=[];summaries={};comparisons={}
    for mode in ('joint','stf0','stfhard','stfsoft'):
        matrix=[]
        for seed in range(10):
            path=root/'results/R5fin'/f'R5fin_{mode}_s{seed}.json'
            blob=path.read_bytes();rows=sorted(json.loads(blob),key=lambda r:r['fold'])
            assert [r['fold'] for r in rows]==list(range(10))
            assert all(r['fixed_aux']==0 for r in rows)
            matrix.append([r['oos_ic'] for r in rows])
            sources.append(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(blob).hexdigest()))
        arr=np.array(matrix);matrices[mode]=arr
        summaries[mode]=dict(mean=float(arr.mean()),std_fold_cells=float(arr.std()))
    for mode in ('stf0','stfhard','stfsoft'):
        d=(matrices[mode]-matrices['joint']).mean(axis=0)
        half=2.2621571628*d.std(ddof=1)/np.sqrt(10)
        comparisons[mode]=dict(fold_mean_differences=d.tolist(),mean=float(d.mean()),
                              ci95_t=[float(d.mean()-half),float(d.mean()+half)],p_wilcoxon=wilcoxon_exact(d.tolist())[0])
    adjusted=holm([x['p_wilcoxon'] for x in comparisons.values()])
    for row,p in zip(comparisons.values(),adjusted):row['p_holm_three_methods']=p
    out=root/'results/OPERATOR_AUDIT/finance_fold_evidence.json'
    out.write_text(json.dumps(dict(scope=__doc__,sources=sources,summary=summaries,comparisons=comparisons),indent=2))
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
