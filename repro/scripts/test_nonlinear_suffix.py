import copy
import torch
from torch import nn
from nonlinear_suffix import AffineMap,FullSuffix,AttentionPrefix

def test_complete_vit_mlp_matches_original_with_cls_pruning():
    from torchvision.models.vision_transformer import VisionTransformer
    torch.manual_seed(332);m=VisionTransformer(image_size=32,patch_size=16,num_layers=2,num_heads=2,hidden_dim=12,mlp_dim=24).double().eval();m.heads=nn.Identity()
    prefix=AttentionPrefix(copy.deepcopy(m));last=m.encoder.layers[-1]
    suffix=FullSuffix([AffineMap(last.mlp[0]),AffineMap(last.mlp[3])],last.ln_2,m.encoder.ln)
    x=torch.randn(7,3,32,32,dtype=torch.float64)
    assert torch.allclose(suffix(prefix(x)),m(x),atol=1e-13)

def test_full_convolutional_lowrank_suffix_matches_all_merged_layers():
    torch.manual_seed(418);base=FullSuffix([AffineMap(nn.Conv2d(6,6,3,padding=1)),AffineMap(nn.Conv2d(6,6,3,padding=1))])
    update=base.lowrank([torch.randn(2,54),torch.randn(2,54)],2)
    with torch.no_grad():
        for layer in update.maps:layer.u.normal_(0,.1);layer.delta_bias.normal_(0,.03)
    x=torch.randn(17,6,3,3)
    assert torch.allclose(update(x),update.dense()(x),atol=2e-6,rtol=1e-5)
    assert not torch.allclose(update(x),base(x))

def test_head_rebasing_retains_the_initial_affine_function():
    from shared_reference_vision import NormalizedHead
    torch.manual_seed(513);x=torch.randn(27,13);w=torch.randn(5,13);b=torch.randn(5);mean=torch.randn(13);sd=torch.rand(13)+.2
    head=NormalizedHead(w*sd,b+w@mean,mean,sd)
    assert torch.allclose(head(x),nn.functional.linear(x,w,b),atol=4e-6,rtol=2e-6)
