"""Local tensor/logit verification of routing and saved-model replay evidence."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def outputs(z,thresholds):
    probabilities=z.double().softmax(1);conf,label=probabilities.max(1)
    return torch.stack([label,*[label.masked_fill(conf<t,-1) for t in thresholds]],1)


def finite(value):
    if isinstance(value,torch.Tensor):
        assert not value.is_floating_point() or torch.isfinite(value).all()
        return 1
    if isinstance(value,dict):return sum(finite(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(finite(v) for v in value)
    return 0


def routing(root):
    audited=[]
    for path in sorted(root.glob('p???.json')):
        row=json.loads(path.read_text());state=root/(path.stem+'_states')/'routing.pt'
        assert row['complete'] and sha(state)==row['state_sha256']
        raw=torch.load(state,map_location='cpu',weights_only=True);tensors=finite(raw)
        reference=outputs(raw['reference'],raw['thresholds']);policies=0
        for kind,reader in raw['readers'].items():
            errors=outputs(reader['cheap'],raw['thresholds'])!=reference
            for branch in [b for b in row['branches'] if b['reader']==kind]:
                cutoff=branch['cutoff'];fraction=branch['target_referral'];score=reader['scores'][branch['gate']]
                selected=torch.full((len(score),),fraction==1.,dtype=torch.bool) if cutoff is None else score<=cutoff
                rates=(errors & ~selected[:,None]).double().mean(0)
                assert np.allclose(rates.numpy(),branch['disagreements'],rtol=0,atol=1e-12)
                assert abs(float(selected.double().mean())-branch['referral'])<1e-12
                policies+=1
        assert policies==80
        audited.append(dict(record=path.name,arch=row['arch'],seed=row['seed'],state_sha256=sha(state),
            finite_tensors=tensors,replayed_policies=policies,current_model_modified=False))
    assert len(audited)==6
    return audited


def auxiliary(root):
    audited=[]
    for path in sorted(root.glob('e???.json')):
        row=json.loads(path.read_text());state=root/(path.stem+'_states')/'replay.pt'
        assert row['complete'] and sha(state)==row['state_sha256']
        raw=torch.load(state,map_location='cpu',weights_only=True);tensors=finite(raw)
        thresholds=list(raw['thresholds'].values())
        changes=(outputs(raw['old_logits'],thresholds)!=outputs(raw['cached_old_logits'],thresholds)).sum(0).tolist()
        assert row['all_old_decisions_unchanged']==(sum(changes)==0)
        for b in row['branches']:
            value=raw['branches'][b['mode']];correct=int((value['own_logits'].argmax(1)==raw['labels']).sum())
            assert correct==b['correct'] and abs(correct/len(raw['labels'])-b['accuracy'])<1e-12
            for dose,errors in value.get('squared_errors',{}).items():
                assert abs(float(errors.mean())-b['mse_by_dose'][dose])<2e-7
        audited.append(dict(record=path.name,arch=row['arch'],reference_seed=row['reference_seed'],seed=row['seed'],
            state_sha256=sha(state),finite_tensors=tensors,replayed_branches=len(row['branches']),
            old_decision_change_counts=changes,source_model_execution='separate CPU saved-state process',
            local_verification='all returned logits and per-image reconstruction errors'))
    assert audited
    return audited


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--routing',type=Path);p.add_argument('--auxiliary',type=Path,nargs='*',default=[])
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    result=dict(complete=True,source_sha256={Path(__file__).name:sha(Path(__file__))},
        routing=[] if a.routing is None else routing(a.routing),
        auxiliary=[r for root in a.auxiliary for r in auxiliary(root)])
    a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(routing=len(result['routing']),auxiliary=len(result['auxiliary']),complete=True)))
