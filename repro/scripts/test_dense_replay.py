import pytest
import torch
from torch import nn
from torch.nn import functional as F
from dense_replay import (DenseReferenceMLP, dense_delta_states, serve_dense_mlp,
                          merged_serving, serve_merged_mlp, packed_serving, serve_packed_mlp)


class Healthy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(9, 7), nn.ReLU(), nn.Linear(7, 7), nn.ReLU(), nn.Linear(7, 7))
        self.primary = nn.Linear(7, 4)
        self.aux = nn.Linear(7, 1)

    def forward(self, x):
        return self.encoder(x)


def build(seed, s=3, r=2, dtype=torch.float64):
    torch.manual_seed(seed)
    healthy = Healthy().to(dtype)
    A = torch.randn(4, 7, dtype=dtype)
    B = torch.randn(4, dtype=dtype)
    model = DenseReferenceMLP(healthy, torch.randn(s, 9, dtype=dtype), torch.randn(r, 7, dtype=dtype), A, B)
    with torch.no_grad():
        model.p.normal_(0, .4)
        model.u.normal_(0, .3)
        model.delta1_bias.normal_(0, .2)
        model.delta2_bias.normal_(0, .2)
    return healthy, model, A, B


@pytest.mark.parametrize('s,r', [(1, 1), (4, 3)])
def test_dense_update_matches_merged_network_and_preserves_old_function(s, r):
    healthy, model, A, B = build(311 + s, s, r)
    x = torch.randn(23, 9, dtype=torch.float64)
    current, old = model.forward_states(x)
    first = model.first.weight + model.p @ model.q
    first_bias = model.first.bias + model.delta1_bias
    second = model.second.weight + model.u @ model.v
    second_bias = model.second.bias + model.delta2_bias
    direct = F.relu(F.linear(F.relu(F.linear(x, first, first_bias)), second, second_bias))
    assert torch.allclose(current, direct, atol=1e-12)
    assert torch.equal(old, F.relu(model.second(F.relu(model.first(x)))))
    assert torch.allclose(model(x), model.final(direct), atol=1e-12)
    assert torch.equal(model.legacy_logits(old), F.linear(healthy(x), A, B))
    assert torch.allclose(model.legacy_logits(old, True), model.legacy_logits(old), atol=1e-12)
    assert int(((current > 0) != (old > 0)).sum()) > 0
    assert float((model.p @ model.q).abs().sum(1).min()) > 0


def test_update_parameter_gradients_match_the_merged_network():
    healthy, model, A, B = build(917, 3, 2)
    x = torch.randn(11, 9, dtype=torch.float64)
    params = model.update_parameters()
    replay = torch.autograd.grad(model(x).square().mean(), params, retain_graph=True)
    first = model.first.weight + model.p @ model.q
    first_bias = model.first.bias + model.delta1_bias
    second = model.second.weight + model.u @ model.v
    second_bias = model.second.bias + model.delta2_bias
    direct = model.final(F.relu(F.linear(F.relu(F.linear(x, first, first_bias)), second, second_bias)))
    merged = torch.autograd.grad(direct.square().mean(), params)
    assert all(torch.allclose(a, b, atol=1e-11, rtol=1e-9) for a, b in zip(replay, merged))


@pytest.mark.parametrize('s,r', [(1, 1), (4, 3)])
def test_packed_and_merged_serving_preserve_both_functions(s, r):
    healthy, model, A, B = build(607 + r, s, r, dtype=torch.float32)
    x = torch.randn(29, 9)
    p = model.serving()
    expected = serve_dense_mlp(x, p)
    for actual in [serve_packed_mlp(x, packed_serving(p)),
                   (serve_merged_mlp(x, merged_serving(p)),
                    F.linear(F.relu(model.second(F.relu(model.first(x)))), p['old_weight'], p['old_bias']))]:
        assert all(torch.allclose(a, b, atol=1e-5, rtol=1e-5) for a, b in zip(actual, expected))
    assert torch.allclose(expected[1], F.linear(healthy(x), A.float(), B.float()), atol=2e-6)


def test_dense_delta_states_accepts_cached_healthy_trace():
    healthy, model, A, B = build(424, 3, 2)
    x = torch.randn(17, 9, dtype=torch.float64)
    with torch.no_grad():
        pre1 = model.first(x)
        pre2 = model.second(F.relu(pre1))
    cached = dense_delta_states(x, pre1, pre2, model.second.weight, model.p, model.q,
                                model.delta1_bias, model.u, model.v, model.delta2_bias)
    assert all(torch.equal(a, b) for a, b in zip(cached, model.forward_states(x)))
