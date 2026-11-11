"""Two fixed affine teacher fits; inputs contain features/logits, never labels."""
import torch


@torch.no_grad()
def fit_legacy_heads(features, teacher_logits):
    h,z=features.double(),teacher_logits.double()
    if h.ndim!=2 or z.ndim!=2 or len(h)!=len(z) or len(h)<2:
        raise ValueError('Paired two-dimensional features and teacher logits required')
    mu=h.mean(0,keepdim=True)
    sd=h.std(0,keepdim=True).clamp_min(1e-6)
    x=(h-mu)/sd
    mean=z.mean(0,keepdim=True);target=z-mean
    u,s,vh=torch.linalg.svd(x,full_matrices=False)
    tolerance=max(x.shape)*torch.finfo(torch.float64).eps*s.max()
    inverse=torch.where(s>tolerance,1/s,torch.zeros_like(s))
    ols=vh.T@((u.T@target)*inverse[:,None])
    ridge=torch.linalg.solve(x.T@x/len(x)+.001*torch.eye(x.shape[1],dtype=torch.float64),
                             x.T@target/len(x))
    out={}
    for name,coef in [('ols',ols),('ridge',ridge)]:
        weight=(coef/sd.reshape(-1,1)).T
        bias=(mean-mu@weight.T).reshape(-1)
        residual=(h@weight.T+bias-z).square().mean()
        assert torch.isfinite(weight).all() and torch.isfinite(bias).all()
        out[name]=dict(weight=weight,bias=bias,fitting_logit_mse=float(residual),
                       feature_rank=int((s>tolerance).sum()),svd_tolerance=float(tolerance),
                       ridge_coefficient=0. if name=='ols' else .001)
    return out
