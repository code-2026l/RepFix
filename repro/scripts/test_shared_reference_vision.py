import copy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from shared_reference_vision import SharedReferenceBlock,NormalizedHead,dense_block


@pytest.mark.parametrize('k',[0,2])
def test_cached_residual_update_matches_merge_and_preserves_reference(k):
    torch.manual_seed(711);c=6;f=nn.Conv2d(c,c,3,padding=1);s=nn.Conv2d(c,c,3,padding=1)
    head=NormalizedHead(torch.randn(3,c),torch.randn(3),torch.randn(c),torch.rand(c)+.3)
    model=SharedReferenceBlock(f,s,torch.arange(k),torch.randn(2,c*9)*.2,head)
    with torch.no_grad():model.delta1.normal_(0,.1);model.delta1_bias.normal_(0,.2);model.u.normal_(0,.2)
    x=torch.randn(17,c,3,3);cache=(f(x).detach(),s(F.relu(f(x))).detach())
    current,old=model.forward_states(x,cache);fm,sm=model.merged()
    assert torch.equal(old,dense_block(x,f,s))
    assert torch.allclose(current,dense_block(x,fm,sm),atol=2e-6,rtol=2e-6)
    assert torch.allclose(model(x,cache),model.head(dense_block(x,fm,sm).mean((-2,-1))),atol=1e-5)
    assert int(((current>0)!=(old>0)).sum())>0
    parameters=[p for p in [model.delta1,model.delta1_bias,model.u,model.v,model.delta2_bias] if p.numel()]
    gradients=torch.autograd.grad(model(x,cache).square().mean(),parameters)
    assert all(torch.isfinite(g).all() for g in gradients)
    affine=head.affine();h=torch.randn(8,c)
    assert torch.allclose(head(h),F.linear(h,affine['weight'],affine['bias']),atol=3e-6)
