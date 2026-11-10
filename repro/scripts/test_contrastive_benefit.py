import torch
from contrastive_benefit_cpu import contrastive
from contrastive_serving_cpu import paired_retrieval


def test_paired_view_order_and_common_batch_permutation_preserve_the_objective():
    torch.manual_seed(2);a=torch.randn(12,8);b=torch.randn(12,8);order=torch.randperm(12)
    x=contrastive(a,b,torch.zeros(8),torch.ones(8))
    assert torch.allclose(x,contrastive(b,a,torch.zeros(8),torch.ones(8)))
    assert torch.allclose(x,contrastive(a[order],b[order],torch.zeros(8),torch.ones(8)))


def test_correct_positives_improve_loss_and_both_views_have_gradients():
    a=torch.eye(12,requires_grad=True);b=torch.eye(12,requires_grad=True);mu=torch.zeros(12);sd=torch.ones(12)
    loss=contrastive(a,b,mu,sd)
    assert loss<contrastive(a,b.roll(1,0),mu,sd)
    loss.backward();assert a.grad is not None and b.grad is not None
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_bidirectional_retrieval_checks_instance_pair_identity():
    from torch import nn
    features=torch.eye(8);head=dict(mean=torch.zeros(8),sd=torch.ones(8))
    assert paired_retrieval(nn.Identity(),torch.stack([features,features]),head).all()
    assert not paired_retrieval(nn.Identity(),torch.stack([features,features.roll(1,0)]),head).any()
