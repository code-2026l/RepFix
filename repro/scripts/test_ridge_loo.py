import torch
from ridge_loo import fixed_design_loo


def test_identity_equals_independent_row_deleted_ridge_refits_with_intercept():
    torch.manual_seed(940);x=torch.randn(13,4,dtype=torch.double);y=torch.randn(13,3,dtype=torch.double);penalty=.7
    _,predicted,_=fixed_design_loo(x,y,penalty);aug=torch.cat([torch.ones(13,1,dtype=torch.double),x],1)
    p=torch.diag(torch.tensor([0.,penalty,penalty,penalty,penalty],dtype=torch.double))
    for i in range(13):
        keep=torch.arange(13)!=i;coef=torch.linalg.solve(aug[keep].T@aug[keep]+p,aug[keep].T@y[keep])
        assert torch.allclose(predicted[i],aug[i]@coef,atol=1e-11,rtol=0)
