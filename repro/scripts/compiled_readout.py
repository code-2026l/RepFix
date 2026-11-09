"""Exact affine-output preservation with a frozen trunk and a foldable last map.

This is an elementary constrained-affine construction. It does not permit an
unconstrained change to the nonlinear trunk and makes no algorithmic novelty
claim. Final affine arithmetic is float64 in all experimental branches.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F


class DeploymentInterface(nn.Module):
    """One trunk and one final affine map; no inference reference or projector."""

    def __init__(self, trunk, final, primary, auxiliary):
        super().__init__()
        self.trunk = trunk
        self.final = final
        self.primary = primary
        self.aux = auxiliary

    def forward(self, x):
        return self.final(self.trunk(x).double())


class TrainableInterface(DeploymentInterface):
    def __init__(self, base):
        layers = list(base.encoder.children())
        if not isinstance(layers[-1], nn.Linear):
            raise ValueError('A final affine feature layer is required')
        super().__init__(nn.Sequential(*layers[:-1]), layers[-1].double(),
                         base.primary, base.aux)
        self.register_buffer('projector', None)
        self.register_buffer('anchor_weight', None)
        self.register_buffer('anchor_bias', None)

    def freeze_trunk(self):
        self.trunk.requires_grad_(False)

    def constrain_last(self, projector):
        if projector.shape != (self.final.out_features,) * 2:
            raise ValueError('Projector and feature dimensions differ')
        self.freeze_trunk()
        self.projector = projector.detach().clone().double()
        self.anchor_weight = self.final.weight.detach().clone()
        self.anchor_bias = self.final.bias.detach().clone()

    def effective_affine(self):
        if self.projector is None:
            return self.final.weight, self.final.bias
        weight = self.anchor_weight + self.projector @ (self.final.weight - self.anchor_weight)
        bias = self.anchor_bias + self.projector @ (self.final.bias - self.anchor_bias)
        return weight, bias

    def forward(self, x):
        weight, bias = self.effective_affine()
        return F.linear(self.trunk(x).double(), weight, bias)

    @torch.no_grad()
    def fold(self):
        weight, bias = self.effective_affine()
        final = copy.deepcopy(self.final)
        final.weight.copy_(weight)
        final.bias.copy_(bias)
        deployed = DeploymentInterface(copy.deepcopy(self.trunk), final,
                                       copy.deepcopy(self.primary), copy.deepcopy(self.aux))
        deployed.eval().requires_grad_(False)
        return deployed


def interface_shapes(model):
    """Only deployed feature-encoder shapes, excluding heads and training buffers."""
    return {name: list(value.shape) for name, value in model.state_dict().items()
            if name.startswith(('trunk.', 'final.'))}
