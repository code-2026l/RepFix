"""Exploratory anchor-only forecast versus observed validation disagreement."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from conditional_reference_cpu import ridge
from consumer_routing import decisions,decision_margin,oscillation


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    paths=sorted(a.states.rglob('seed_*.pt'));assert len(paths)==6
    rows=[];order=torch.randperm(2048,generator=torch.Generator().manual_seed(6500));fit,hold=order[:1024],order[1024:]
    for path in paths:
        raw=torch.load(path,map_location='cpu',weights_only=True);is_vit=raw['anchor_current'].shape[1]==768
        refpath=a.references/('next_visiontwoprepare' if is_vit else 'next_visionfullprepare')/('p002_states' if is_vit else 'p000_states')/'reference.pt'
        ref=torch.load(refpath,map_location='cpu',weights_only=True);old=ref['heads']['coarse'];thresholds=list(ref['thresholds'].values())
        x=raw['anchor_current'].float();target=F.linear(raw['anchor_target'].float(),old['weight'],old['bias']);target-=target.mean(1,keepdim=True)
        reader,_,_=ridge(x[fit],target[fit],.001)
        pred=F.linear(x,reader['weight'],reader['bias']);vpred=F.linear(raw['validation_features'].float(),reader['weight'],reader['bias'])
        errors=(decisions(pred,thresholds)!=decisions(target,thresholds)).double()
        actual=(decisions(vpred,thresholds)!=decisions(raw['reference_logits'],thresholds)).double().mean(0)
        margin=decision_margin(target,thresholds);residual=oscillation(pred-target)
        geometric=float((margin[hold]<=residual[hold]).double().mean())
        assert not errors[margin>residual].any()
        sd=raw['anchor_target'][fit].std(0,unbiased=False).clamp_min(1e-6)
        rows.append(dict(arch='vit_b32' if is_vit else 'resnet18',seed=int(path.stem.split('_')[-1]),state_sha256=digest(path),
            reference_sha256=digest(refpath),actual_max_disagreement=float(actual.max()),actual_by_consumer=actual.tolist(),
            predictions=dict(in_sample=float(errors[fit].mean(0).max()),holdout=float(errors[hold].mean(0).max()),geometry=geometric),
            holdout_by_consumer=errors[hold].mean(0).tolist(),standardized_feature_mse=float(((x[hold]-raw['anchor_target'][hold])/sd).square().mean()),
            centered_logit_mse=float((pred[hold]-target[hold]).square().mean())))
    mae={k:float(np.mean([abs(r['predictions'][k]-r['actual_max_disagreement']) for r in rows])) for k in ['in_sample','holdout','geometry']}
    passed=mae['geometry']<=.02 and mae['geometry']<=.75*mae['holdout']
    result=dict(complete=True,phase='exploratory_retrospective_development',official_test_loaded=False,
        plan_sha256=digest(a.plan),source_sha256={Path(__file__).name:digest(Path(__file__))},rows=rows,mae=mae,
        geometry_advance=passed,confirmation_runs=0,labels_used_for_fitting_or_prediction=False,
        interpretation='Margin condition is a sufficient event bound, not a calibrated probability prediction or information-loss test')
    a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(mae=mae,advance=passed,rows=[{k:r[k] for k in ['arch','seed','actual_max_disagreement','predictions']} for r in rows]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--states',type=Path,required=True);p.add_argument('--references',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);main(a)
