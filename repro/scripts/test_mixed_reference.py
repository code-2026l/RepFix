import copy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from nonlinear_suffix import AffineMap,FullSuffix
from token_suffix import TokenSuffix
from mixed_reference import MixedReference

@pytest.mark.parametrize('transformer',[False,True])
def test_reference_only_quantization_does_not_change_current_fp32_function(transformer):
    torch.manual_seed(931)
    if transformer:
        from torchvision.models.vision_transformer import VisionTransformer
        m=VisionTransformer(image_size=32,patch_size=16,num_layers=2,num_heads=2,hidden_dim=12,mlp_dim=24).eval()
        old=TokenSuffix(m.encoder.layers[-1],m.encoder.ln,[m.encoder.layers[0]]);x=torch.randn(7,5,12);d=12
    else:
        old=FullSuffix([AffineMap(nn.Conv2d(7,7,3,padding=1)),AffineMap(nn.Conv2d(7,7,3,padding=1))]);x=torch.randn(7,7,4,4);d=7
    current=copy.deepcopy(old)
    with torch.no_grad():current.maps[-1].weight.add_(.03*torch.randn_like(current.maps[-1].weight))
    reader=dict(weight=torch.randn(5,d),bias=torch.randn(5));legacy=dict(weight=torch.randn(3,d),bias=torch.randn(3))
    model=MixedReference(old,current,reader,legacy,True).eval().requires_grad_(False)
    with torch.no_grad():
        expected=F.linear(current(x),**reader);own,protected=model(x)
        assert torch.equal(own,expected) and own.dtype==torch.float32 and protected.dtype==torch.float32
        assert all(p.dtype==torch.float32 for p in current.parameters())
        assert any(t.dtype==torch.bfloat16 for t in model.old.buffers())
        traced=torch.jit.freeze(torch.jit.trace(model,x[:1],check_trace=False));z,old_z=traced(x)
        assert torch.allclose(z,expected,atol=6e-6,rtol=2e-6) and torch.isfinite(old_z).all()
