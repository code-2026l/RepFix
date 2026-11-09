"""Structural folding, optimization, normalization and scope tests."""
import copy

import pytest
import torch
from torch import nn

from compiled_readout import TrainableInterface, interface_shapes
from readout_nullspace import readout_projector


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(7, 13), nn.ReLU(), nn.Linear(13, 8))
        self.primary = nn.Linear(8, 3)
        self.aux = nn.Linear(8, 1)


def fixture():
    torch.manual_seed(71)
    model = TrainableInterface(Base())
    reference = copy.deepcopy(model)
    aweight = torch.randn(3, 8, dtype=torch.float64)
    sd = torch.linspace(.2, 2.1, 8, dtype=torch.float64)
    p, _ = readout_projector(aweight, sd)
    a = aweight / sd
    model.constrain_last(p)
    return model, reference, a


@pytest.mark.parametrize('scale', [.01, 1., 100.])
def test_arbitrary_weight_and_bias_update_preserve_normalized_logits(scale):
    model, reference, a = fixture()
    with torch.no_grad():
        model.final.weight.add_(scale * torch.randn_like(model.final.weight))
        model.final.bias.add_(scale * torch.randn_like(model.final.bias))
    x = torch.randn(29, 7)
    assert torch.max(torch.abs((model(x) - reference(x)) @ a.T)) < 1e-10
    assert (model(x) - reference(x)).norm() > 1e-3


def test_adam_and_folding_keep_the_identity_without_inference_extras():
    model, reference, a = fixture()
    x = torch.randn(31, 7)
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    initial = model.final.weight.detach().clone()
    for _ in range(8):
        optimizer.zero_grad()
        loss = (model(x) - torch.randn(31, 8, dtype=torch.float64)).square().mean()
        loss.backward()
        assert model.final.weight.grad.norm() > 1e-8
        assert all(p.grad is None for p in model.trunk.parameters())
        optimizer.step()
    assert not torch.equal(initial, model.final.weight)
    deployed = model.fold()
    assert torch.allclose(deployed(x), model(x), atol=1e-12, rtol=0)
    assert torch.max(torch.abs((deployed(x) - reference(x)) @ a.T)) < 1e-10
    assert interface_shapes(deployed) == interface_shapes(reference)
    assert not any('anchor' in key or 'projector' in key for key in deployed.state_dict())
    calls = []
    handles = [module.register_forward_hook(lambda *_: calls.append(1))
               for module in [*deployed.trunk.modules(), deployed.final]
               if isinstance(module, nn.Linear)]
    deployed(x)
    for handle in handles:
        handle.remove()
    assert len(calls) == 2


def test_trunk_drift_is_outside_the_exact_preservation_contract():
    model, reference, a = fixture()
    with torch.no_grad():
        model.trunk[0].bias.add_(.5)
    x = torch.randn(19, 7)
    assert torch.max(torch.abs((model(x) - reference(x)) @ a.T)) > 1e-3


def test_unconstrained_fold_is_equivalent_and_trunk_remains_trainable():
    torch.manual_seed(89)
    model = TrainableInterface(Base())
    x = torch.randn(23, 7)
    model(x).square().mean().backward()
    assert model.trunk[0].weight.grad.norm() > 1e-8
    assert torch.equal(model.fold()(x), model(x))


def test_full_rank_consumer_leaves_no_affine_feature_update():
    model = TrainableInterface(Base())
    reference = copy.deepcopy(model)
    p, info = readout_projector(torch.eye(8), torch.ones(8))
    assert info['free_dimensions'] == 0
    model.constrain_last(p)
    with torch.no_grad():
        model.final.weight.add_(1.)
        model.final.bias.add_(2.)
    x = torch.randn(11, 7)
    assert torch.equal(model(x), reference(x))
