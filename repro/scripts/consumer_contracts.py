"""Versioned decisions, prediction regressions and selective coverage."""
import torch
import numpy as np
from consumer_risk import certificate
import paired_readout_continuation as paired


def confidence_thresholds(training_logits):
    if training_logits.ndim!=2 or not torch.isfinite(training_logits).all():raise ValueError('Finite training logits required')
    confidence=training_logits.double().softmax(1).max(1).values
    return {'coverage80':float(torch.quantile(confidence,.2)),'coverage50':float(torch.quantile(confidence,.5))}


def decision_flows(old,new,labels):
    if old.shape!=new.shape or old.shape!=labels.shape:raise ValueError('Paired decisions and labels required')
    changed=old!=new;right=old==labels;now=new==labels
    counts=dict(examples=len(old),changed=int(changed.sum()),correct_to_wrong=int((right&~now).sum()),
        wrong_to_correct=int((~right&now).sum()),wrong_to_wrong=int((changed&~right&~now).sum()),old_correct=int(right.sum()),new_correct=int(now.sum()))
    assert counts['changed']==sum(counts[k] for k in ['correct_to_wrong','wrong_to_correct','wrong_to_wrong'])
    assert counts['new_correct']-counts['old_correct']==counts['wrong_to_correct']-counts['correct_to_wrong']
    counts['negative_flip_rate']=counts['correct_to_wrong']/len(old)
    counts['negative_flip_given_old_correct']=counts['correct_to_wrong']/max(1,counts['old_correct'])
    return counts


def evaluate_contracts(current,reference,labels,thresholds,monitor=None,alpha=.05/1000):
    if current.shape!=reference.shape or not torch.isfinite(current).all() or not torch.isfinite(reference).all():
        raise ValueError('Paired finite logits required')
    new_conf,new=current.double().softmax(1).max(1);old_conf,old=reference.double().softmax(1).max(1)
    result={}
    for name,threshold in [('classification',None),*thresholds.items()]:
        p=new.clone();q=old.clone()
        if threshold is not None:p[new_conf<threshold]=-1;q[old_conf<threshold]=-1
        accepted=p>=0;old_accepted=q>=0
        row=dict(disagreement=float((p!=q).double().mean()),threshold=threshold,
            coverage=float(accepted.double().mean()),reference_coverage=float(old_accepted.double().mean()),
            accepted_accuracy=float((p[accepted]==labels[accepted]).double().mean()) if accepted.any() else None,
            reference_accepted_accuracy=float((q[old_accepted]==labels[old_accepted]).double().mean()) if old_accepted.any() else None)
        if threshold is None:
            row.update(metrics=paired.metrics(current,labels),reference_metrics=paired.metrics(reference,labels),flows=decision_flows(q,p,labels))
        if monitor is not None:
            bound=certificate(p.numpy(),q.numpy(),np.asarray(monitor),alpha);bound['certified']=bound['upper_remaining_disagreement']<=.05
            row['certificate']=bound
        result[name]=row
    return result
