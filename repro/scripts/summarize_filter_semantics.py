"""Combine disjoint seed shards and report all matched-operator outcomes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from paired_stats import wilcoxon_exact,holm


def main():
    root=Path(__file__).resolve().parents[2]
    folder=root/'results/FILTER_SEMANTICS'
    sources=[];raw=[]
    for name in ('har_recon_a100_aux_3s60.json','har_recon_a100_aux_seeds3to9_60.json'):
        blob=(folder/name).read_bytes();d=json.loads(blob)
        assert d['complete'] and d['configuration']['alpha']==100
        assert d['configuration']['epochs']==60 and d['configuration']['basis']=='aux'
        sources.append(dict(path=name,sha256=hashlib.sha256(blob).hexdigest()))
        raw.extend(d['raw'])
    rows={m:sorted([r for r in raw if r['mode']==m],key=lambda r:r['seed']) for m in ('joint','stf0','stfcalrank','stfbackward')}
    assert len(raw)==40
    assert all([r['seed'] for r in rs]==list(range(10)) for rs in rows.values())
    for a,b in zip(rows['stfcalrank'],rows['stfbackward']):
        assert a['gate']==b['gate']==1
        assert a['rank']==b['rank'] and a['rho_cal']==b['rho_cal']
    fields=('metric','aux_metric','rel_scatter')
    summary={m:{k:dict(mean=float(np.mean([r[k] for r in rs])),std=float(np.std([r[k] for r in rs]))) for k in fields} for m,rs in rows.items()}
    differences={}
    for k in fields:
        diff=np.array([b[k]-a[k] for a,b in zip(rows['stfcalrank'],rows['stfbackward'])])
        half=2.2621571628*diff.std(ddof=1)/np.sqrt(10)
        differences[k]=dict(mean=float(diff.mean()),ci95_t=[float(diff.mean()-half),float(diff.mean()+half)],
                            p_wilcoxon=wilcoxon_exact(diff.tolist())[0],positive_seeds=int((diff>0).sum()))
    for row,p in zip(differences.values(),holm([r['p_wilcoxon'] for r in differences.values()])):
        row['p_holm_three_outcomes']=p
    out=dict(sources=sources,scope='exploratory matched forward/backward comparison on one HAR split; ten initialization seeds; differences are backward minus forward',summary=summary,differences=differences,raw=raw)
    (folder/'har_recon_a100_aux_10s_summary.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(dict(summary=summary,differences=differences),indent=2))


if __name__=='__main__':main()
