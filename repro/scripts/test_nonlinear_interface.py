"""Global compatibility, nonlinear learning, fusion and optimizer tests."""
import copy

import pytest
import torch
from torch import nn

from nonlinear_interface import NativeInterface, NonlinearSideInterface, continued_adam, encoder_accounting
from paired_readout_continuation import state_hash
from readout_nullspace import readout_projector


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(7, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
        self.primary = nn.Linear(8, 3)
        self.aux = nn.Linear(8, 1)


def fixture(width=3):
    torch.manual_seed(194)
    healthy = NativeInterface(Base())
    weight = torch.randn(3, 8, dtype=torch.float64)
    sd = torch.linspace(.3, 2.7, 8, dtype=torch.float64)
    p, _ = readout_projector(weight, sd)
    model = NonlinearSideInterface(healthy, width, p)
    return healthy, model, weight/sd


@pytest.mark.parametrize('width', [3, 5])
def test_arbitrary_nonlinear_changes_preserve_logits_and_fuse(width):
    healthy, model, a = fixture(width)
    x = torch.randn(31, 7)
    assert torch.equal(model(x), healthy(x))
    with torch.no_grad():
        for p in [*model.side_trunk.parameters(), *model.side_final.parameters()]:
            p.add_(3*torch.randn_like(p))
    before = model(x)
    assert (before-healthy(x)).norm() > 1.
    assert torch.max(torch.abs((before-healthy(x)) @ a.T)) < 1e-10
    fused = model.fuse()
    assert torch.max(torch.abs(fused(x)-before)) < 1e-10
    assert fused.final.in_features == 8+width
    assert not any('base_' in key or 'projector' in key or 'side_' in key for key in fused.state_dict())
    assert torch.equal(fused.trunk[0].weight[:8], healthy.trunk[0].weight)
    assert torch.equal(fused.trunk[2].weight[:8,:8], healthy.trunk[2].weight)
    assert torch.count_nonzero(fused.trunk[2].weight[:8,8:]) == 0
    assert torch.count_nonzero(fused.trunk[2].weight[8:,:8]) == 0
    assert encoder_accounting(fused)['parameters'] > encoder_accounting(healthy.fold())['parameters']


def test_adam_updates_the_nonlinear_prefix_and_preserves_base_states():
    healthy, model, a = fixture()
    x = torch.randn(29, 7)
    warm = torch.optim.Adam(healthy.parameters(), lr=.001)
    warm.zero_grad()
    healthy(x).square().mean().backward()
    warm.step()
    model = NonlinearSideInterface(healthy, 3, model.projector)
    moments = copy.deepcopy(warm.state_dict())
    optimizer = continued_adam(model, moments)
    old_count = len(moments['param_groups'][0]['params'])
    assert state_hash(optimizer.state_dict()['state']) == state_hash(moments['state'])
    base = state_hash(dict(trunk=model.base_trunk.state_dict(), final=model.base_final.state_dict()))
    first = model.side_trunk[0].weight.detach().clone()
    for _ in range(12):
        optimizer.zero_grad()
        (model(x)-torch.randn(29,8,dtype=torch.float64)).square().mean().backward()
        optimizer.step()
    assert not torch.equal(first, model.side_trunk[0].weight)
    assert state_hash(dict(trunk=model.base_trunk.state_dict(), final=model.base_final.state_dict())) == base
    for key in range(6):
        assert state_hash(optimizer.state_dict()['state'][key]) == state_hash(moments['state'][key])
    assert any(key >= old_count for key in optimizer.state_dict()['state'])
    assert torch.max(torch.abs((model(x)-healthy(x)) @ a.T)) < 1e-10


def test_unconstrained_and_base_drift_controls_break_preservation():
    healthy, model, a = fixture()
    x = torch.randn(13,7)
    with torch.no_grad():
        model.projector.copy_(torch.eye(8))
        model.side_final.bias.add_(1.)
    assert torch.max(torch.abs((model(x)-healthy(x)) @ a.T)) > .01
    healthy, model, a = fixture()
    with torch.no_grad():
        model.base_trunk[0].bias.add_(1.)
    assert torch.max(torch.abs((model(x)-healthy(x)) @ a.T)) > .01


def test_full_rank_consumer_blocks_any_side_displacement():
    healthy, _, _ = fixture()
    p, _ = readout_projector(torch.eye(8), torch.ones(8))
    model = NonlinearSideInterface(healthy, 3, p)
    with torch.no_grad():
        model.side_final.weight.add_(4.)
    x = torch.randn(17,7)
    assert torch.equal(model(x), healthy(x))


def test_fused_model_evaluates_three_affine_layers_only():
    _, model, _ = fixture()
    fused = model.fuse()
    calls = []
    handles = [module.register_forward_hook(lambda *_: calls.append(1))
               for module in [*fused.trunk.modules(), fused.final] if isinstance(module, nn.Linear)]
    fused(torch.randn(9,7))
    for handle in handles:
        handle.remove()
    assert len(calls) == 3
