import copy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from nonlinear_suffix import AffineMap,FullSuffix
from token_suffix import TokenSuffix
from mixed_reference import MixedReference,AutocastReference

@pytest.mark.parametrize('transformer',[False,True])
@pytest.mark.parametrize('autocast_ref',[False,True])
def test_reference_only_quantization_does_not_change_current_fp32_function(transformer,autocast_ref):
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
    model=(AutocastReference(old,current,reader,legacy) if autocast_ref else MixedReference(old,current,reader,legacy,True)).eval().requires_grad_(False)
    with torch.no_grad():
        expected=F.linear(current(x),**reader);own,protected=model(x)
        assert torch.equal(own,expected) and own.dtype==torch.float32 and protected.dtype==torch.float32
        assert all(p.dtype==torch.float32 for p in current.parameters())
        if not autocast_ref:assert any(t.dtype==torch.bfloat16 for t in model.old.buffers())
        if autocast_ref:
            from benchmark_suffix_cpu import trace_amp
            traced=trace_amp(model,x[:1])
        else:traced=torch.jit.freeze(torch.jit.trace(model,x[:1],check_trace=False))
        z,old_z=traced(x)
        assert torch.allclose(z,expected,atol=6e-6,rtol=2e-6) and torch.isfinite(old_z).all()

def test_standard_amp_control_keeps_fp32_consumer_and_restores_execution_settings():
    from whole_amp import WholeAMPReadout
    from benchmark_suffix_cpu import trace_amp
    torch.manual_seed(943);body=FullSuffix([AffineMap(nn.Conv2d(4,4,3,padding=1)),AffineMap(nn.Conv2d(4,4,3,padding=1))]).eval()
    reader=dict(weight=torch.randn(3,4),bias=torch.randn(3));model=WholeAMPReadout(nn.Identity(),body,reader).eval().requires_grad_(False);x=torch.randn(7,4,4,4)
    original=torch.backends.mha.get_fastpath_enabled()
    with torch.no_grad():
        traced=trace_amp(model,x[:1]);z=traced(x)
        assert z.dtype==torch.float32 and torch.isfinite(z).all()
        assert torch.allclose(z,model(x),atol=.01,rtol=.01)
    assert torch.backends.mha.get_fastpath_enabled()==original and not torch.is_autocast_enabled('cpu')
    assert model.weight.dtype==torch.float32 and all(p.dtype==torch.float32 for p in body.parameters())
