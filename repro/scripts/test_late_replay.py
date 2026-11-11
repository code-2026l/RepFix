import copy
import pytest
import torch
from torch import nn
from late_replay import LateDelta,finish,ViTPrefix

@pytest.mark.parametrize('convolution',[False,True])
def test_late_update_equals_independent_merge_and_parameter_gradients(convolution):
    torch.manual_seed(810);d,m,r=7,9,3
    tail=(nn.Conv2d(d,m,3,padding=1) if convolution else nn.Linear(d,m)).double()
    norm=nn.Identity() if convolution else nn.LayerNorm(m).double()
    delta=LateDelta(tail,norm,torch.randn(r,tail.weight[0].numel(),dtype=torch.float64))
    with torch.no_grad():delta.u.normal_(0,.2);delta.bias.normal_(0,.1)
    x=torch.randn((11,d,3,3) if convolution else (11,d),dtype=torch.float64)
    residual=torch.randn((11,m,3,3) if convolution else (11,m),dtype=torch.float64)
    current=delta(x,residual);merged=finish(delta.merged()(x),residual,norm)
    assert torch.allclose(current,merged,atol=5e-14,rtol=1e-12)
    weight=tail.weight+(delta.u@delta.v).reshape_as(tail.weight);bias=tail.bias+delta.bias
    direct=nn.functional.conv2d(x,weight,bias,padding=1) if convolution else nn.functional.linear(x,weight,bias)
    target=finish(direct,residual,norm)
    p=[delta.u,delta.v,delta.bias]
    g=torch.autograd.grad(current.square().mean(),p,retain_graph=True);g2=torch.autograd.grad(target.square().mean(),p)
    assert all(torch.allclose(a,b,atol=1e-12,rtol=1e-10) for a,b in zip(g,g2))

def test_last_vit_cls_pruning_matches_full_model_before_any_adaptation():
    from torchvision.models.vision_transformer import VisionTransformer
    torch.manual_seed(912);model=VisionTransformer(image_size=32,patch_size=16,num_layers=2,num_heads=2,hidden_dim=12,mlp_dim=24).double().eval()
    model.heads=nn.Identity();prefix=ViTPrefix(copy.deepcopy(model));tail=model.encoder.layers[-1].mlp[3]
    x=torch.randn(5,3,32,32,dtype=torch.float64);h,residual=prefix(x)
    assert torch.allclose(finish(tail(h),residual,model.encoder.ln),model(x),atol=1e-13)
