"""Established single-update alignment objectives and affine serving algebra."""
import torch
from torch.nn import functional as F


def alignment_loss(mapped, reference, weight, bias, family, teacher_logits=None):
    if mapped.shape != reference.shape or mapped.ndim != 2:
        raise ValueError('Paired feature matrices required')
    if family == 'feature_map':
        return F.mse_loss(mapped, reference.detach())
    if family == 'logit_map':
        target=F.linear(reference.detach(), weight, bias) if teacher_logits is None else teacher_logits
        return F.mse_loss(F.linear(mapped, weight, bias),target.detach())
    raise ValueError('Unknown alignment family')


@torch.no_grad()
def folded_reader(map_weight, old_weight, old_bias, dtype=torch.float32):
    if map_weight.ndim != 2 or old_weight.shape[1] != map_weight.shape[0]:
        raise ValueError('Compatible linear map and old consumer required')
    return dict(weight=old_weight.to(dtype) @ map_weight.to(dtype),
                bias=old_bias.to(dtype).clone())


def reader_logits(features, reader):
    return F.linear(features.to(reader['weight'].dtype),reader['weight'],reader['bias'])
