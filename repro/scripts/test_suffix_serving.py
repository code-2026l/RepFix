import copy
import torch
from torch import nn
from torch.nn import functional as F
from nonlinear_suffix import AffineMap,FullSuffix
from suffix_serving import SuffixReadout,WholeNetwork

def objects():
    torch.manual_seed(824)
    prefix=nn.Conv2d(3,6,1)
    old=FullSuffix([AffineMap(nn.Conv2d(6,6,3,padding=1)),AffineMap(nn.Conv2d(6,6,3,padding=1))])
    current=copy.deepcopy(old)
    with torch.no_grad():current.maps[0].weight.add_(.13*torch.randn_like(current.maps[0].weight))
    legacy=dict(weight=torch.randn(3,6),bias=torch.randn(3));reader=dict(weight=torch.randn(4,6),bias=torch.randn(4))
    return prefix,old,current,legacy,reader

def test_separate_suffixes_keep_the_old_function_and_dynamic_batch():
    prefix,old,current,legacy,reader=objects();x=torch.randn(13,3,4,4)
    one=WholeNetwork(prefix,SuffixReadout(old,current,reader,legacy)).eval()
    both=WholeNetwork(prefix,SuffixReadout(old,current,reader,legacy,both=True)).eval()
    with torch.no_grad():
        traced=torch.jit.freeze(torch.jit.trace(both,x[:1]));z,healthy=traced(x)
        assert torch.allclose(z,one(x),atol=5e-6)
        assert torch.allclose(healthy,F.linear(old(prefix(x)),**legacy),atol=5e-6)
        assert not torch.allclose(z,F.linear(old(prefix(x)),**reader))

def test_richer_frozen_feature_control_uses_the_same_healthy_trace_for_both_heads():
    prefix,old,_,legacy,_=objects();x=prefix(torch.randn(17,3,4,4));reader=dict(weight=torch.randn(4,12),bias=torch.randn(4))
    with torch.no_grad():
        own,healthy=SuffixReadout(old,None,reader,legacy,'H',True)(x)
        h=old.hidden(x);expected=F.linear(torch.cat([old(x),h.mean((-2,-1))],1),**reader)
        assert torch.allclose(own,expected,atol=2e-6)
        assert torch.allclose(healthy,F.linear(old(x),**legacy),atol=2e-6)
        same=SuffixReadout(old,None,legacy,legacy,'F',False).eval()
        traced=torch.jit.freeze(torch.jit.trace(same,x[:1]))
        assert torch.allclose(traced(x),healthy,atol=2e-6)
