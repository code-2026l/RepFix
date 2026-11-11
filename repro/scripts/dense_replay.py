"""Exact two-version serving for dense low-rank first-layer updates.

The sparse-row construction restricts which first-layer rows may change and
pays k*d to recompute them.  A dense rank-s update P Q changes every row and
still costs only s*d + m*s to evaluate on top of the shared healthy product;
the second layer then needs one m*m correction.  Sparse support is therefore a
sufficient, not a necessary, condition for exact replay.

Everything here is a new family; the released sparse-row module and its records
are left untouched.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F


def dense_delta_states(x, pre1, pre2, weight2, p, q, bias1, u, v, bias2):
    old1 = F.relu(pre1)
    delta_pre1 = F.linear(F.linear(x, q), p) + bias1
    current1 = F.relu(pre1 + delta_pre1)
    correction = F.linear(current1 - old1, weight2)
    current_pre2 = pre2 + correction + F.linear(F.linear(current1, v), u) + bias2
    return F.relu(current_pre2), F.relu(pre2)


class DenseReferenceMLP(nn.Module):
    def __init__(self, healthy, q0, v0, A, B):
        super().__init__()
        self.first = copy.deepcopy(healthy.encoder[0]).requires_grad_(False)
        self.second = copy.deepcopy(healthy.encoder[2]).requires_grad_(False)
        self.final = copy.deepcopy(healthy.encoder[4])
        self.primary = copy.deepcopy(healthy.primary)
        self.aux = copy.deepcopy(healthy.aux)
        self.healthy_final = copy.deepcopy(healthy.encoder[4]).requires_grad_(False)
        m = self.second.out_features
        s = q0.shape[0]
        rank = v0.shape[0]
        self.p = nn.Parameter(torch.zeros(m, s, dtype=self.first.weight.dtype))
        self.q = nn.Parameter(q0.clone().to(self.first.weight.dtype))
        self.delta1_bias = nn.Parameter(torch.zeros(m, dtype=self.first.weight.dtype))
        self.u = nn.Parameter(torch.zeros(m, rank, dtype=self.first.weight.dtype))
        self.v = nn.Parameter(v0.clone().to(self.first.weight.dtype))
        self.delta2_bias = nn.Parameter(torch.zeros(m, dtype=self.first.weight.dtype))
        self.register_buffer('old_A', A.detach().double().clone())
        self.register_buffer('old_B', B.detach().double().clone())

    def old_parameters(self):
        return [p for layer in (self.first, self.second, self.final, self.primary, self.aux)
                for p in layer.parameters()]

    def update_parameters(self):
        return [self.p, self.q, self.delta1_bias, self.u, self.v, self.delta2_bias]

    def forward_states(self, x, cache=None):
        if cache is None:
            pre1 = self.first(x)
            pre2 = self.second(F.relu(pre1))
        else:
            pre1, pre2 = cache
        return dense_delta_states(x, pre1, pre2, self.second.weight, self.p, self.q,
                                  self.delta1_bias, self.u, self.v, self.delta2_bias)

    def forward(self, x, cache=None):
        return self.final(self.forward_states(x, cache)[0])

    def legacy_logits(self, old2, compressed=False):
        if not compressed:
            return F.linear(self.healthy_final(old2).double(), self.old_A, self.old_B)
        weight = self.old_A @ self.healthy_final.weight.double()
        bias = self.old_B + self.old_A @ self.healthy_final.bias.double()
        return F.linear(old2.double(), weight, bias)

    @torch.no_grad()
    def serving(self):
        current_weight = (self.primary.weight.double() @ self.final.weight.double()).float()
        current_bias = (self.primary.bias.double() + self.primary.weight.double() @ self.final.bias.double()).float()
        old_weight = self.old_A @ self.healthy_final.weight.double()
        old_bias = self.old_B + self.old_A @ self.healthy_final.bias.double()
        return dict(first_weight=self.first.weight.detach().clone(), first_bias=self.first.bias.detach().clone(),
                    second_weight=self.second.weight.detach().clone(), second_bias=self.second.bias.detach().clone(),
                    p=self.p.detach().clone(), q=self.q.detach().clone(), delta1_bias=self.delta1_bias.detach().clone(),
                    u=self.u.detach().clone(), v=self.v.detach().clone(), delta2_bias=self.delta2_bias.detach().clone(),
                    current_weight=current_weight, current_bias=current_bias,
                    old_weight=old_weight.float(), old_bias=old_bias.float())


@torch.no_grad()
def serve_dense_mlp(x, s):
    pre1 = F.linear(x, s['first_weight'], s['first_bias'])
    pre2 = F.linear(F.relu(pre1), s['second_weight'], s['second_bias'])
    current, old = dense_delta_states(x, pre1, pre2, s['second_weight'], s['p'], s['q'],
                                      s['delta1_bias'], s['u'], s['v'], s['delta2_bias'])
    return F.linear(current, s['current_weight'], s['current_bias']), F.linear(old, s['old_weight'], s['old_bias'])


@torch.no_grad()
def merged_serving(s):
    first = s['first_weight'] + s['p'] @ s['q']
    first_bias = s['first_bias'] + s['delta1_bias']
    second = s['second_weight'] + s['u'] @ s['v']
    return dict(first_weight=first, first_bias=first_bias, second_weight=second,
                second_bias=s['second_bias'] + s['delta2_bias'],
                current_weight=s['current_weight'], current_bias=s['current_bias'])


@torch.no_grad()
def serve_merged_mlp(x, m):
    h = F.relu(F.linear(x, m['first_weight'], m['first_bias']))
    h = F.relu(F.linear(h, m['second_weight'], m['second_bias']))
    return F.linear(h, m['current_weight'], m['current_bias'])


@torch.no_grad()
def packed_serving(s):
    """One dense product serves both versions; the delta path is four small products."""
    m = s['second_weight'].shape[1]
    first = torch.cat([s['first_weight'], s['q']], 0)
    first_bias = torch.cat([s['first_bias'], torch.zeros(s['q'].shape[0], dtype=s['first_bias'].dtype)])
    return dict(first_weight=first, first_bias=first_bias, second_plain=s['second_weight'],
                second_bias=s['second_bias'], p=s['p'], delta1_bias=s['delta1_bias'], u=s['u'],
                v=s['v'], delta2_bias=s['delta2_bias'], width=m,
                current_weight=s['current_weight'], current_bias=s['current_bias'],
                old_weight=s['old_weight'], old_bias=s['old_bias'])


@torch.no_grad()
def serve_packed_mlp(x, s):
    m = s['width']
    pre = F.linear(x, s['first_weight'], s['first_bias'])
    pre1 = pre[:, :m]
    qx = pre[:, m:]
    old1 = F.relu(pre1)
    current1 = F.relu(pre1 + F.linear(qx, s['p']) + s['delta1_bias'])
    pre2 = F.linear(old1, s['second_plain'], s['second_bias'])
    correction = (F.linear(current1 - old1, s['second_plain'])
                  + F.linear(F.linear(current1, s['v']), s['u']) + s['delta2_bias'])
    return (F.linear(F.relu(pre2 + correction), s['current_weight'], s['current_bias']),
            F.linear(F.relu(pre2), s['old_weight'], s['old_bias']))
