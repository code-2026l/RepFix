import copy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from shared_reference_replay import SharedReferenceMLP,linear_delta_states,convolution_delta_states,serve_mlp,merged_serving,serve_merged_mlp,packed_serving,serve_packed_mlp,serve_prefix_mlp


class Healthy(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(9,7),nn.ReLU(),nn.Linear(7,7),nn.ReLU(),nn.Linear(7,7))
        self.primary=nn.Linear(7,4);self.aux=nn.Linear(7,1)
    def forward(self,x):return self.encoder(x)


@pytest.mark.parametrize('k',[0,3])
def test_nonzero_linear_update_matches_merged_network_and_old_function(k):
    torch.manual_seed(121);healthy=Healthy().double();A=torch.randn(4,7,dtype=torch.float64);B=torch.randn(4,dtype=torch.float64)
    update=SharedReferenceMLP(healthy,torch.arange(k),torch.randn(2,7,dtype=torch.float64),A,B)
    with torch.no_grad():
        update.delta1.normal_(0,.7);update.delta1_bias.normal_(0,.3);update.u.normal_(0,.4);update.delta2_bias.normal_(0,.2)
    x=torch.randn(23,9,dtype=torch.float64);current,old=update.forward_states(x)
    assert torch.allclose(update(x),update.merged()(x),atol=1e-12)
    assert torch.equal(update.legacy_logits(old),F.linear(healthy(x),A,B))
    assert torch.allclose(update.legacy_logits(old,True),update.legacy_logits(old),atol=1e-12)
    assert int(((current>0)!=(old>0)).sum())>0


def test_replay_and_independent_dense_merge_have_equal_parameter_gradients():
    torch.manual_seed(920);x=torch.randn(11,9,dtype=torch.float64);healthy=Healthy().double()
    update=SharedReferenceMLP(healthy,torch.tensor([1,4]),torch.randn(2,7,dtype=torch.float64),torch.randn(4,7,dtype=torch.float64),torch.zeros(4,dtype=torch.float64))
    with torch.no_grad():update.delta1.normal_(0,.1);update.u.normal_(0,.2)
    p=update.update_parameters();y=update(x);g=torch.autograd.grad(y.square().mean(),p,retain_graph=True)
    w1=healthy.encoder[0].weight.index_add(0,update.support,update.delta1)
    b1=healthy.encoder[0].bias.index_add(0,update.support,update.delta1_bias)
    h1=F.relu(F.linear(x,w1,b1));h2=F.relu(F.linear(h1,healthy.encoder[2].weight+update.u@update.v,healthy.encoder[2].bias+update.delta2_bias))
    direct=update.final(h2);g2=torch.autograd.grad(direct.square().mean(),p)
    assert all(torch.allclose(a,b,atol=1e-11,rtol=1e-9) for a,b in zip(g,g2))


def test_convolutional_identity_residual_update_matches_dense_merge():
    torch.manual_seed(820);n,c,r=4,6,2;x=torch.randn(n,c,4,4,dtype=torch.float64)
    w1=torch.randn(c,c,3,3,dtype=torch.float64)*.2;w2=torch.randn(c,c,3,3,dtype=torch.float64)*.2
    b1=torch.randn(c,dtype=torch.float64)*.1;b2=torch.randn(c,dtype=torch.float64)*.1;s=torch.tensor([0,3])
    d=torch.randn(len(s),c,3,3,dtype=torch.float64)*.15;db=torch.randn(len(s),dtype=torch.float64)*.1
    u=torch.randn(c,r,1,1,dtype=torch.float64)*.1;v=torch.randn(r,c,3,3,dtype=torch.float64)*.1;db2=torch.randn(c,dtype=torch.float64)*.1
    pre1=F.conv2d(x,w1,b1,padding=1);pre2=F.conv2d(F.relu(pre1),w2,b2,padding=1)
    current,old=convolution_delta_states(x,pre1,pre2,w2[:,s],s,d,db,u,v,db2)
    merged1=w1.index_add(0,s,d);mergedb=b1.index_add(0,s,db);merged2=w2+torch.einsum('or,rijk->oijk',u[:,:,0,0],v)
    direct=F.relu(F.conv2d(F.relu(F.conv2d(x,merged1,mergedb,padding=1)),merged2,b2+db2,padding=1)+x)
    assert torch.allclose(current,direct,atol=2e-14)
    assert torch.equal(old,F.relu(pre2+x))


def test_float32_serving_uses_no_second_full_encoder_and_matches_folded_current():
    torch.manual_seed(77);h=Healthy();A=torch.randn(4,7,dtype=torch.float64);B=torch.randn(4,dtype=torch.float64)
    u=SharedReferenceMLP(h,torch.tensor([1,2]),torch.randn(2,7),A,B)
    with torch.no_grad():u.u.normal_(0,.03);u.delta1.normal_(0,.1)
    p=u.serving();assert set(p)=={'first_weight','first_bias','second_weight','second_bias','support','delta1','delta1_bias','columns','u','v','delta2_bias','current_weight','current_bias','old_weight','old_bias'}
    x=torch.randn(19,9);now,old=serve_mlp(x,p)
    assert torch.allclose(now,serve_merged_mlp(x,merged_serving(p)),atol=1e-6)
    assert torch.allclose(old,F.linear(h(x).double(),A,B).float(),atol=2e-6)


@pytest.mark.parametrize('k',[0,3])
def test_packed_and_prefix_serving_preserve_both_functions(k):
    torch.manual_seed(202);h=Healthy();u=SharedReferenceMLP(h,torch.arange(k),torch.randn(2,7),torch.randn(4,7,dtype=torch.float64),torch.randn(4,dtype=torch.float64))
    with torch.no_grad():u.u.normal_(0,.2);u.delta1.normal_(0,.3);u.delta1_bias.normal_(0,.1)
    p=u.serving();x=torch.randn(29,9);expected=serve_mlp(x,p)
    for actual in [serve_packed_mlp(x,packed_serving(p)),serve_prefix_mlp(x,p,merged_serving(p))]:
        assert all(torch.allclose(a,b,atol=3e-6,rtol=2e-6) for a,b in zip(actual,expected))
