"""KL projection onto a class/threshold cell, with a feasible interior buffer.

Set-valued KL targets and envelope differentiation are established operations.
This module implements the specific star-order and confidence-bin intersection
needed to retain a class reader and versioned selective readers together.
"""
import math
import torch
from torch.nn import functional as F


@torch.no_grad()
def cell_spec(teacher_logits,thresholds,buffer=.1):
    z=teacher_logits.detach().double()
    if z.ndim!=2 or z.shape[1]<2 or not torch.isfinite(z).all():
        raise ValueError('A finite multiclass teacher is required')
    if not 0<buffer<1 or any(not math.isfinite(float(t)) or not 0<=t<=1 for t in thresholds):
        raise ValueError('Valid confidence thresholds and an interior buffer required')
    logp=z.log_softmax(1);k=logp.argmax(1);chosen=logp.gather(1,k[:,None]).squeeze(1)
    other=logp.scatter(1,k[:,None],-torch.inf).max(1).values
    confidence=chosen.exp();ratio=(buffer*(chosen-other).clamp(0,1)).exp()
    lower=torch.zeros_like(confidence);upper=torch.ones_like(confidence)
    has_upper=torch.zeros_like(k,dtype=torch.bool)
    for t in sorted(set(float(t) for t in thresholds)):
        accepted=confidence>=t
        lower=torch.where(accepted,torch.maximum(lower,torch.full_like(lower,t)),lower)
        upper=torch.where(~accepted,torch.minimum(upper,torch.full_like(upper,t)),upper)
        has_upper|=~accepted
    lo=torch.where(lower>0,(1-buffer)*lower+buffer*confidence,lower)
    hi=torch.where(has_upper,(1-buffer)*upper+buffer*confidence,upper)
    # A threshold equal to one is different from an unconstrained upper endpoint.
    hi=torch.where(has_upper,torch.minimum(hi,torch.nextafter(upper,torch.full_like(upper,-torch.inf))),hi)
    minimum=ratio/(ratio+z.shape[1]-1)
    lo=torch.maximum(lo,minimum)
    if ((lo>confidence+2e-14)|(hi<confidence-2e-14)|(lo>hi+2e-14)).any():
        raise ArithmeticError('Healthy teacher does not lie in its cell')
    return dict(**{'class':k},ratio=ratio,lower=lo,upper=hi,teacher_log_probability=logp,
                teacher_ties=int((chosen==other).sum()),threshold_ties=int(sum((confidence==float(t)).sum() for t in thresholds)))


def slice_spec(spec,index):
    return {k:(v[index] if isinstance(v,torch.Tensor) else v) for k,v in spec.items()}


@torch.no_grad()
def project_cell(logq,spec):
    """Unique minimizer of KL(p||q); weighted pooling then capped allocation."""
    logq=logq.detach().double();n,c=logq.shape;k=spec['class'];r=spec['ratio']
    if c<2 or len(k)!=n or not torch.isfinite(logq).all():raise ValueError('Finite paired log probabilities required')
    chosen=logq.gather(1,k[:,None]).squeeze(1)
    other=logq[torch.arange(c).expand(n,c)!=k[:,None]].reshape(n,c-1)
    ordered,indices=other.sort(1,descending=True)
    prefix=torch.cat([torch.zeros(n,1,dtype=logq.dtype),ordered.cumsum(1)],1)
    m=torch.arange(c-1,dtype=logq.dtype)
    previous=(r[:,None]*(chosen-r.log())[:,None]+prefix[:,:-1])/(r[:,None]+m)
    pooled=(ordered>previous).sum(1)
    logg=(r*(chosen-r.log())+prefix.gather(1,pooled[:,None]).squeeze(1))/(r+pooled)
    tails=torch.cat([ordered.flip(1).logcumsumexp(1).flip(1),torch.full((n,1),-torch.inf,dtype=logq.dtype)],1)
    remaining=tails.gather(1,pooled[:,None]).squeeze(1)
    normalizer=torch.logaddexp((r+pooled).log()+logg,remaining)
    unconstrained=(r.log()+logg-normalizer).exp()
    t=torch.maximum(spec['lower'],torch.minimum(unconstrained,spec['upper']))
    cap=t/r;logcap=cap.log()
    mass=1-t[:,None]-cap[:,None]*m
    logmass=torch.where(mass>0,mass.clamp_min(1e-300).log(),torch.full_like(mass,-torch.inf))
    logscale=logmass-tails[:,:-1]
    possible=(mass>=-2e-14)&(logscale+ordered<=logcap[:,None]+2e-13)
    tight=t<=r/(r+c-1)+2e-14
    if (~possible.any(1)&~tight).any():raise ArithmeticError('No capped allocation found')
    capped=possible.to(torch.int64).argmax(1)
    scale=logscale.gather(1,capped[:,None]).squeeze(1)
    values=(ordered+scale[:,None]).exp()
    values=torch.where(torch.arange(c-1)[None,:]<capped[:,None],cap[:,None],values)
    values=torch.where(tight[:,None],cap[:,None],values)
    unsorted=torch.empty_like(values).scatter_(1,indices,values)
    result=torch.empty_like(logq)
    result[torch.arange(c).expand(n,c)!=k[:,None]]=unsorted.reshape(-1)
    result.scatter_(1,k[:,None],t[:,None])
    # Return the exact input probabilities when already inside the protected cell.
    q=logq.exp();qk=q.gather(1,k[:,None]).squeeze(1)
    inside=(qk>=spec['lower'])&(qk<=spec['upper'])&(qk[:,None]>=r[:,None]*q.scatter(1,k[:,None],0.)).all(1)
    return torch.where(inside[:,None],q,result)


def cell_loss(logits,spec,reduction='mean'):
    logq=F.log_softmax(logits.double(),dim=1)
    target=project_cell(logq,spec)
    values=(torch.special.xlogy(target,target)-target*logq).sum(1).clamp_min(0)
    return values.mean() if reduction=='mean' else values


def cell_logodds_loss(logits,spec,reduction='mean'):
    """Largest squared cell violation, measured in logit/log-odds units."""
    z=logits.double();k=spec['class'];chosen=z.gather(1,k[:,None]).squeeze(1)
    other=z.scatter(1,k[:,None],-torch.inf)
    rank=(other.max(1).values-chosen+spec['ratio'].log()).clamp_min(0)
    odds=chosen-torch.logsumexp(other,dim=1)
    lo,hi=spec['lower'],spec['upper']
    lower=torch.where(lo>0,lo.log()-torch.log1p(-lo),torch.full_like(lo,-torch.inf))
    upper=torch.where(hi<1,hi.log()-torch.log1p(-hi),torch.full_like(hi,torch.inf))
    violation=torch.maximum(rank,torch.maximum((lower-odds).clamp_min(0),(odds-upper).clamp_min(0)))
    values=violation.square()
    return values.mean() if reduction=='mean' else values
