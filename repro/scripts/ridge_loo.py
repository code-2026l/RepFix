"""Standard fixed-design ridge leave-one-out identity, with free intercept."""
import torch


def fixed_design_loo(design,target,penalty):
    x=design.double();y=target.double();n,d=x.shape
    mean=x.mean(0);a=x-mean;ym=y.mean(0)
    solved=torch.linalg.solve(a.T@a+penalty*torch.eye(d,dtype=torch.double),a.T)
    fitted=ym+a@(solved@(y-ym));leverage=(a*solved.T).sum(1)+1/n
    assert bool(((leverage>=0)&(leverage<1)).all())
    return fitted,y-(y-fitted)/(1-leverage[:,None]),leverage
