"""Post-hoc affine coordinates, with no restriction on earlier nonlinear learning.

Row convention: emitted = native @ matrix + offset. Invertible coordinates can
be folded into a final affine encoder and undone in a current head. This is an
algebraic reparameterization, not a guarantee about an older head's accuracy.
"""
import copy

import torch
from torch import nn


@torch.no_grad()
def stabilize(matrix, offset, relative_floor=1e-3):
    matrix, offset = matrix.double(), offset.double().reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('A square feature coordinate map is required')
    if not 0 < relative_floor <= 1 or not torch.isfinite(matrix).all():
        raise ValueError('Finite coordinates and a positive relative floor required')
    u, singular, vh = torch.linalg.svd(matrix)
    if singular.max() <= 0 or not torch.isfinite(offset).all():
        raise ValueError('The zero map cannot be used as invertible coordinates')
    clipped = singular.clamp_min(relative_floor * singular.max())
    stable = (u * clipped) @ vh
    return dict(matrix=stable, offset=offset, condition_number=float(clipped.max()/clipped.min()),
                singular_values_before=singular.tolist(), singular_values_after=clipped.tolist(),
                clipped_directions=int((singular < relative_floor * singular.max()).sum()))


@torch.no_grad()
def raw_chart(standardized_map, healthy_mean, healthy_sd):
    """Convert the existing standardized readout map into raw old coordinates."""
    scale = healthy_sd.double().reshape(-1)
    return stabilize(standardized_map['matrix'] * scale,
                     standardized_map['offset'] * scale + healthy_mean.double())


@torch.no_grad()
def rebase_linear(layer, chart):
    """Return a float64 affine head with the same native logits."""
    if not isinstance(layer, nn.Linear):
        raise TypeError('The consumer must begin with an affine layer')
    matrix, offset = chart['matrix'], chart['offset']
    result = nn.Linear(layer.in_features, layer.out_features, bias=True).double()
    weight_t = torch.linalg.solve(matrix, layer.weight.detach().double().T)
    bias = (torch.zeros(layer.out_features, dtype=torch.float64) if layer.bias is None
            else layer.bias.detach().double())
    result.weight.copy_(weight_t.T)
    result.bias.copy_(bias - (offset @ weight_t).reshape(-1))
    return result


@torch.no_grad()
def fold_chart(model, chart):
    """Fold a chart after training, rebasing current primary and auxiliary heads."""
    result = model.fold()
    matrix, offset = chart['matrix'], chart['offset']
    result.final.weight.copy_(matrix.T @ result.final.weight.detach().clone())
    result.final.bias.copy_((result.final.bias.detach().clone() @ matrix + offset).reshape(-1))
    result.primary = rebase_linear(result.primary, chart)
    result.aux = copy.deepcopy(result.aux).double()
    if isinstance(result.aux, nn.Linear):
        result.aux = rebase_linear(result.aux, chart)
    elif isinstance(result.aux, nn.Sequential) and isinstance(result.aux[0], nn.Linear):
        result.aux[0] = rebase_linear(result.aux[0], chart)
    else:
        raise TypeError('Unsupported auxiliary head: an affine first layer is required')
    return result.eval().requires_grad_(False)
