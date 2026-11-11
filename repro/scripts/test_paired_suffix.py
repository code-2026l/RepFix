import copy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from nonlinear_suffix import AffineMap,FullSuffix
from token_suffix import TokenSuffix
from paired_suffix import pack_suffix

@pytest.mark.parametrize('both',[False,True])
def test_grouped_dense_convolutions_match_independently_evaluated_versions(both):
    torch.manual_seed(601);old=FullSuffix([AffineMap(nn.Conv2d(7,7,3,padding=1)),AffineMap(nn.Conv2d(7,7,3,padding=1))]).double();current=copy.deepcopy(old)
    with torch.no_grad():
        for m in current.maps:m.weight.add_(torch.randn_like(m.weight)*.1)
    reader=dict(weight=torch.randn(5,7,dtype=torch.float64),bias=torch.randn(5,dtype=torch.float64));legacy=dict(weight=torch.randn(3,7,dtype=torch.float64),bias=torch.randn(3,dtype=torch.float64));x=torch.randn(9,7,4,4,dtype=torch.float64)
    model=pack_suffix(old,current,reader,legacy,both)
    with torch.no_grad():
        actual=model(x);expected=F.linear(current(x),**reader)
        assert torch.allclose(actual[0] if both else actual,expected,atol=2e-13)
        if both:assert torch.allclose(actual[1],F.linear(old(x),**legacy),atol=2e-13)
        traced=torch.jit.freeze(torch.jit.trace(model,x[:1],check_trace=False));result=traced(x)
        assert torch.allclose(result[0] if both else result,expected,atol=2e-13)

@pytest.mark.parametrize('both',[False,True])
def test_grouped_attention_and_mlp_match_both_full_token_suffix_functions(both):
    from torchvision.models.vision_transformer import VisionTransformer
    torch.manual_seed(602);m=VisionTransformer(image_size=32,patch_size=16,num_layers=2,num_heads=2,hidden_dim=12,mlp_dim=24).double().eval()
    old=TokenSuffix(m.encoder.layers[-1],m.encoder.ln,[m.encoder.layers[0]]);current=copy.deepcopy(old)
    with torch.no_grad():
        current.preceding[0].self_attention.in_proj_weight.add_(.03*torch.randn_like(current.preceding[0].self_attention.in_proj_weight))
        for layer in current.maps:layer.weight.add_(.03*torch.randn_like(layer.weight))
    reader=dict(weight=torch.randn(5,12,dtype=torch.float64),bias=torch.randn(5,dtype=torch.float64));legacy=dict(weight=torch.randn(3,12,dtype=torch.float64),bias=torch.randn(3,dtype=torch.float64));x=torch.randn(9,5,12,dtype=torch.float64)
    model=pack_suffix(old,current,reader,legacy,both)
    with torch.no_grad():
        actual=model(x);expected=F.linear(current(x),**reader)
        assert torch.allclose(actual[0] if both else actual,expected,atol=3e-13)
        if both:assert torch.allclose(actual[1],F.linear(old(x),**legacy),atol=3e-13)
        traced=torch.jit.freeze(torch.jit.trace(model,x[:1],check_trace=False));result=traced(x)
        assert torch.allclose(result[0] if both else result,expected,atol=3e-13)
