"""Local reference-level inference, keeping continuation seeds nested."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import t


def reference_interval(values):
    values=np.asarray(values,dtype=float)
    assert values.ndim==2 and values.shape[0]>=2 and np.isfinite(values).all()
    means=values.mean(1);mean=float(means.mean());radius=float(t.ppf(.975,len(means)-1)*means.std(ddof=1)/np.sqrt(len(means)))
    return dict(mean=mean,ci95=[mean-radius,mean+radius],reference_means=means.tolist(),
        references=len(means),continuations_per_reference=values.shape[1],df=len(means)-1)


def main(root,choice,out):
    selected=json.loads(choice.read_text());assert selected['complete']
    assert selected['backbones']['vit_b32']['confirm_ce_only']
    paths=sorted(root.glob('c???.json'));rows=[json.loads(p.read_text()) for p in paths]
    assert len(rows)==12 and all(r['complete'] and r['phase']=='confirmation' and not r['splits']['official_test_loaded'] for r in rows)
    expected={(r,s) for r in range(6201,6205) for s in range(6310,6313)}
    assert {(r['reference_seed'],r['seed']) for r in rows}==expected
    by_reference=[];contrasts={k:[] for k in ['F','N','H','strongest_frozen']}
    for reference in range(6201,6205):
        own=sorted([r for r in rows if r['reference_seed']==reference],key=lambda r:r['seed'])
        assert len({r['splits']['fitting_sha256'] for r in own})==1
        assert len({r['splits']['validation_sha256'] for r in own})==1
        assert len({r['splits']['mask_sha256'] for r in own})==1
        scores={k:[next(b['validation']['own']['acc'] for b in r['branches'] if b['mode']==k) for r in own] for k in ['F','N','H','D']}
        assert all(set(b['mode'] for b in r['branches'])=={'F','N','H','D'} for r in own)
        best=max(['F','N','H'],key=lambda k:np.mean(scores[k]))
        for name in contrasts:
            control=best if name=='strongest_frozen' else name
            contrasts[name].append((np.array(scores['D'])-np.array(scores[control])).tolist())
        by_reference.append(dict(reference_seed=reference,continuation_seeds=[r['seed'] for r in own],
            label_count=own[0]['splits']['labeled'],splits=own[0]['splits'],accuracies=scores,
            strongest_frozen_mean=best,mean_gain_over_H_pp=100*float(np.mean(contrasts['H'][-1]))))
    statistics={k:reference_interval(v) for k,v in contrasts.items()}
    result=dict(complete=True,analysis='local',arch='vit_b32',protocol='CE-only independent fitted-reference confirmation',
        official_test_loaded=False,auxiliary_benefit_confirmed=False,
        shared_imagenet_pretraining=True,independent_pretraining_seeds=False,
        reference_level_statistics=statistics,primary_transfer_gate=statistics['H']['ci95'][0]>0,
        reference_groups=by_reference,selection_sha256=hashlib.sha256(choice.read_bytes()).hexdigest(),
        record_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(transfer_gate=result['primary_transfer_gate'],statistics=statistics),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--choice',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.records,a.choice,a.out)
