"""Established side-network/projection ideas as a compatible nonlinear control.

The frozen base is retained inside a wider fused MLP. This is not a same-size
encoder, and the dense zero blocks still consume parameters and CPU arithmetic.
All feature-encoder arithmetic is native float64 in this separate study.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from compiled_readout import DeploymentInterface, TrainableInterface


class DeployedInterface(DeploymentInterface):
    def forward(self, x):
        return self.final(self.trunk(x.double()))


class NativeInterface(TrainableInterface):
    def __init__(self, base):
        super().__init__(base)
        self.trunk.double()

    def forward(self, x):
        weight, bias = self.effective_affine()
        return F.linear(self.trunk(x.double()), weight, bias)

    @torch.no_grad()
    def fold(self):
        weight, bias = self.effective_affine()
        final = copy.deepcopy(self.final)
        final.weight.copy_(weight)
        final.bias.copy_(bias)
        deployed = DeployedInterface(copy.deepcopy(self.trunk), final,
                                      copy.deepcopy(self.primary), copy.deepcopy(self.aux))
        return deployed.eval().requires_grad_(False)


class NonlinearSideInterface(nn.Module):
    def __init__(self, healthy, width, projector):
        super().__init__()
        if len(healthy.trunk) != 4 or width > healthy.final.in_features or width < 1:
            raise ValueError('The declared construction requires a two-ReLU MLP and a narrow channel')
        self.base_trunk = copy.deepcopy(healthy.trunk).requires_grad_(False)
        self.base_final = copy.deepcopy(healthy.final).requires_grad_(False)
        # Shared parameters retain the original optimizer ordering before the
        # newly introduced side parameters. No historical model is duplicated.
        self.primary = copy.deepcopy(healthy.primary)
        self.aux = copy.deepcopy(healthy.aux)
        din, m = healthy.trunk[0].in_features, healthy.final.out_features
        self.side_trunk = nn.Sequential(nn.Linear(din, width), nn.ReLU(),
                                        nn.Linear(width, width), nn.ReLU()).double()
        self.side_final = nn.Linear(width, m).double()
        with torch.no_grad():
            self.side_trunk[0].weight.copy_(healthy.trunk[0].weight[:width])
            self.side_trunk[0].bias.copy_(healthy.trunk[0].bias[:width])
            self.side_trunk[2].weight.copy_(healthy.trunk[2].weight[:width, :width])
            self.side_trunk[2].bias.copy_(healthy.trunk[2].bias[:width])
            self.side_final.weight.zero_()
            self.side_final.bias.zero_()
        if projector.shape != (m, m):
            raise ValueError('Feature and projector dimensions differ')
        self.register_buffer('projector', projector.detach().clone().double())

    def forward(self, x):
        xd = x.double()
        base = self.base_final(self.base_trunk(xd))
        side = self.side_final(self.side_trunk(xd))
        return base + side @ self.projector.T

    @torch.no_grad()
    def fuse(self):
        base_width = self.base_final.in_features
        side_width = self.side_final.in_features
        width = base_width + side_width
        first = nn.Linear(self.base_trunk[0].in_features, width).double()
        second = nn.Linear(width, width).double()
        final = nn.Linear(width, self.base_final.out_features).double()
        first.weight.copy_(torch.cat([self.base_trunk[0].weight, self.side_trunk[0].weight]))
        first.bias.copy_(torch.cat([self.base_trunk[0].bias, self.side_trunk[0].bias]))
        second.weight.copy_(torch.block_diag(self.base_trunk[2].weight, self.side_trunk[2].weight))
        second.bias.copy_(torch.cat([self.base_trunk[2].bias, self.side_trunk[2].bias]))
        final.weight.copy_(torch.cat([self.base_final.weight, self.projector @ self.side_final.weight], dim=1))
        final.bias.copy_(self.base_final.bias + self.projector @ self.side_final.bias)
        trunk = nn.Sequential(first, nn.ReLU(), second, nn.ReLU())
        deployed = DeployedInterface(trunk, final, copy.deepcopy(self.primary), copy.deepcopy(self.aux))
        return deployed.eval().requires_grad_(False)


def continued_adam(model, healthy_moments):
    """Keep old parameter/Adam ordering; new side states initialize lazily."""
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    fresh = optimizer.state_dict()
    prior = copy.deepcopy(healthy_moments)
    old_count = len(prior['param_groups'][0]['params'])
    assert len(fresh['param_groups']) == len(prior['param_groups']) == 1
    assert fresh['param_groups'][0]['params'][:old_count] == prior['param_groups'][0]['params']
    prior['param_groups'][0]['params'] = fresh['param_groups'][0]['params']
    optimizer.load_state_dict(prior)
    return optimizer


def encoder_accounting(model):
    params = [p for layer in [model.trunk, model.final] for p in layer.parameters()]
    return dict(width=model.final.in_features, parameters=sum(p.numel() for p in params),
                parameter_bytes=sum(p.numel()*p.element_size() for p in params),
                zero_weights=sum(int((p == 0).sum()) for p in params),
                shapes={name:list(t.shape) for name,t in model.state_dict().items()
                        if name.startswith(('trunk.', 'final.'))})
