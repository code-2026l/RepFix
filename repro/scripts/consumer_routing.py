"""Consumer-boundary margins and the oracle mass of exact-reference repair.

These quantities concern agreement with a fixed reference, not label accuracy.
An estimated residual is a routing score, not a certified pointwise bound.
"""
import numpy as np
import torch
from scipy.optimize import linprog


def validate(z,thresholds):
    if z.ndim!=2 or z.shape[1]<2 or not torch.isfinite(z).all():
        raise ValueError('Finite categorical logits with at least two classes required')
    if any(not 0<float(t)<1 for t in thresholds):raise ValueError('Thresholds must be probabilities in (0,1)')


def decisions(z,thresholds):
    validate(z,thresholds);p=z.double().softmax(1);confidence,label=p.max(1)
    return torch.stack([label,*[torch.where(confidence>=t,label,torch.full_like(label,-1)) for t in thresholds]],1)


def decision_margin(z,thresholds):
    validate(z,thresholds);z=z.double();top=z.topk(2,dim=1)
    winner=top.indices[:,0];other=z.clone();other.scatter_(1,winner[:,None],-torch.inf)
    odds=top.values[:,0]-torch.logsumexp(other,dim=1)
    margins=[top.values[:,0]-top.values[:,1]]
    for threshold in thresholds:
        t=torch.tensor(float(threshold),dtype=torch.double,device=z.device)
        margins.append((odds-t.logit()).abs())
    return torch.stack(margins,1).min(1).values


def oscillation(delta):
    if delta.ndim!=2 or not torch.isfinite(delta).all():raise ValueError('Finite logit differences required')
    return delta.max(1).values-delta.min(1).values


def route_logits(cheap,reference,referred):
    if cheap.shape!=reference.shape or referred.dtype!=torch.bool or referred.shape!=(len(cheap),):
        raise ValueError('Matched logits and a boolean referral vector required')
    result=cheap.clone();result[referred]=reference[referred].to(result);return result


def oracle_referral(errors,budget):
    """Fractional exact-fallback mass under each consumer's disagreement budget.

    The seven error patterns encode overlap. This lower-bounds any integral
    routing policy with unit reference cost; it does not lower-bound latency.
    """
    errors=np.asarray(errors,dtype=bool)
    if errors.ndim!=2 or not len(errors) or not 1<=errors.shape[1]<=8:raise ValueError('Nonempty consumer errors required')
    budgets=np.broadcast_to(np.asarray(budget,dtype=float),(errors.shape[1],))
    if np.any((budgets<0)|(budgets>1)):raise ValueError('Budgets must be fractions')
    masks=np.arange(1,2**errors.shape[1]);patterns=((masks[:,None]>>np.arange(errors.shape[1]))&1).astype(float)
    ids=(errors*(1<<np.arange(errors.shape[1]))).sum(1)
    mass=np.asarray([(ids==mask).mean() for mask in masks])
    result=linprog(np.ones(len(mass)),A_ub=-patterns.T,b_ub=budgets-errors.mean(0),bounds=list(zip(np.zeros_like(mass),mass)),method='highs')
    if not result.success:raise RuntimeError(result.message)
    assert np.all(errors.mean(0)-patterns.T@result.x<=budgets+1e-8)
    return dict(minimum_fraction=max(0.,float(result.fun)),pattern_mass=mass.tolist(),referred_pattern_mass=result.x.tolist(),
                initial_disagreements=errors.mean(0).tolist(),budgets=budgets.tolist(),fractional_unit_cost=True)
