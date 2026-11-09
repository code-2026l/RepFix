"""Tests the protected output, its metric, and primary-label isolation."""
import pytest
import torch
from readout_nullspace import guarded_features,masked_primary_loss,readout_projector


@pytest.mark.parametrize('classes,width',[(3,9),(7,4),(1,7)])
def test_arbitrary_feature_drift_preserves_full_affine_logits(classes,width):
    g=torch.Generator().manual_seed(351+width)
    w=torch.randn(classes,width,generator=g,dtype=torch.float64)
    mu=torch.randn(width,generator=g,dtype=torch.float64)
    sd=torch.exp(torch.randn(width,generator=g,dtype=torch.float64)*2)
    b=torch.randn(classes,generator=g,dtype=torch.float64)
    h0=torch.randn(31,width,generator=g,dtype=torch.float64)
    h=torch.randn(31,width,generator=g,dtype=torch.float64)*15
    p,info=readout_projector(w,sd)
    actual=((guarded_features(h,h0,p)-mu)/sd)@w.T+b
    expected=((h0-mu)/sd)@w.T+b
    torch.testing.assert_close(actual,expected,atol=2e-10,rtol=2e-12)
    assert info['protected_rank']==min(classes,width)


def test_raw_weight_nullspace_is_wrong_with_nonuniform_normalization():
    w=torch.tensor([[1.,1.,0.]],dtype=torch.float64)
    sd=torch.tensor([1.,10.,2.],dtype=torch.float64)
    correct,_=readout_projector(w,sd)
    wrong,_=readout_projector(w,torch.ones(3))
    a=w/sd
    assert float((a@correct).norm())<1e-12
    assert float((a@wrong).norm())>.1


def test_hidden_primary_labels_cannot_change_training_gradients():
    z=torch.randn(8,3,dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0,1,2,0,1,2,1,0])
    mask=torch.tensor([True,False,True,False,False,True,False,False])
    altered=y.clone();altered[~mask]=(altered[~mask]+1)%3
    first=torch.autograd.grad(masked_primary_loss(z,y,mask),z)[0]
    second=torch.autograd.grad(masked_primary_loss(z,altered,mask),z)[0]
    assert torch.equal(first,second)
    assert torch.count_nonzero(first[~mask])==0


def test_gradient_can_learn_only_in_readout_nullspace():
    w=torch.tensor([[1.,2.,3.,4.]],dtype=torch.float64)
    p,_=readout_projector(w,torch.ones(4))
    h=torch.randn(10,4,dtype=torch.float64,requires_grad=True)
    ref=torch.zeros_like(h)
    target=torch.randn_like(h)
    loss=(guarded_features(h,ref,p)-target).square().sum()
    grad=torch.autograd.grad(loss,h)[0]
    assert float(grad.norm())>0
    torch.testing.assert_close(grad@w.T,torch.zeros(10,1,dtype=torch.float64),atol=1e-12,rtol=0)
