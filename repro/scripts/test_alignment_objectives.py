import torch
import pytest
from alignment_objectives import alignment_loss,folded_reader,reader_logits


def test_feature_loss_matches_bc_single_step_norm_with_explicit_width_factor():
    x=torch.tensor([[1.,2.,3.],[-1.,0.,2.]],dtype=torch.float64)
    y=torch.zeros_like(x)
    actual=alignment_loss(x,y,torch.eye(3),torch.zeros(3),'feature_map')
    torch.testing.assert_close(actual,(x-y).square().sum(1).mean()/3)


def test_logit_alignment_ignores_a_consumer_null_direction():
    ref=torch.zeros(2,3,dtype=torch.float64)
    current=torch.tensor([[0.,0.,1.],[0.,0.,-2.]],dtype=torch.float64,requires_grad=True)
    a=torch.tensor([[1.,0.,0.],[0.,1.,0.]],dtype=torch.float64)
    b=torch.tensor([.3,-.2],dtype=torch.float64)
    logit=alignment_loss(current,ref,a,b,'logit_map')
    feature=alignment_loss(current,ref,a,b,'feature_map')
    assert logit==0 and feature>0
    assert torch.autograd.grad(logit,current,retain_graph=True)[0].abs().max()==0
    assert torch.autograd.grad(feature,current)[0].abs().max()>0


def test_folding_preserves_affine_outputs_in_double_precision():
    g=torch.Generator().manual_seed(88)
    x=torch.randn(71,9,generator=g,dtype=torch.float64)
    m=torch.randn(9,9,generator=g,dtype=torch.float64)
    a=torch.randn(5,9,generator=g,dtype=torch.float64)
    b=torch.randn(5,generator=g,dtype=torch.float64)
    reader=folded_reader(m,a,b,torch.float64)
    torch.testing.assert_close(reader_logits(x,reader),(x@m.T)@a.T+b,atol=1e-12,rtol=1e-12)


def test_teacher_is_frozen_while_student_gets_alignment_gradient():
    x=torch.ones(3,4,requires_grad=True)
    teacher=torch.zeros_like(x,requires_grad=True)
    loss=alignment_loss(x,teacher,torch.eye(4),torch.zeros(4),'logit_map')
    loss.backward()
    assert teacher.grad is None and x.grad.abs().sum()>0


def test_unknown_alignment_does_not_silently_change_objective():
    with pytest.raises(ValueError):
        alignment_loss(torch.zeros(2,3),torch.zeros(2,3),torch.eye(3),torch.zeros(3),'other')
