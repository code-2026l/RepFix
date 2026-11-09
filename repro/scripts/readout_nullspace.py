"""Elementary output-preserving feature projection, not a new optimizer."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def readout_projector(weight, sd, relative_tolerance=1e-12):
    w, s=weight.double(),sd.double().reshape(-1)
    if w.ndim!=2 or w.shape[1]!=len(s) or torch.any(s<=0):
        raise ValueError('An affine readout and positive feature scales are required')
    a=w/s
    _,singular,vh=torch.linalg.svd(a,full_matrices=True)
    cutoff=relative_tolerance*float(singular.max()) if len(singular) else 0.
    rank=int((singular>cutoff).sum())
    basis=vh[:rank].T
    p=torch.eye(w.shape[1],dtype=torch.float64,device=w.device)-basis@basis.T
    residual=float((a@p).norm()/a.norm().clamp_min(1e-30))
    if residual>1e-10 or float((p@p-p).norm())>1e-10:
        raise ArithmeticError('Readout-nullspace projection failed')
    return p,dict(protected_rank=rank,free_dimensions=w.shape[1]-rank,
                  relative_annihilation_error=residual,singular_values=singular.tolist())


def guarded_features(current,reference,projector):
    if current.shape!=reference.shape or projector.shape!=(current.shape[1],)*2:
        raise ValueError('Current/reference feature and projector dimensions differ')
    return reference.double()+(current.double()-reference.double())@projector


def masked_primary_loss(logits,labels,labeled):
    if labeled.dtype!=torch.bool or labeled.shape!=labels.shape:
        raise ValueError('A Boolean labeled mask is required')
    return F.cross_entropy(logits[labeled],labels[labeled]) if labeled.any() else logits.sum()*0.
