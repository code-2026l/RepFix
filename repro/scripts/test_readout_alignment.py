"""Identifiability tests for the diagnostic repair hierarchy."""
import torch

from readout_alignment import apply_map, fit_repairs, relative_residual


def data():
    rng = torch.Generator().manual_seed(17)
    x = torch.randn(600, 5, generator=rng, dtype=torch.float64)
    x = x * torch.tensor([1., 2., 3., 1.5, .7]) + torch.tensor([2., -1., 0., .5, 3.])
    return x, (x - x.mean(0)) / x.std(0)


def test_positive_global_scale_and_translation():
    reference, target = data()
    scale = .027
    current = scale * reference
    repairs = fit_repairs(reference, current)
    assert relative_residual(apply_map(current, repairs['scalar_only']), target) < 1e-12
    shifted = current + 4.3
    repairs = fit_repairs(reference, shifted)
    assert relative_residual(apply_map(shifted, repairs['mean_scalar']), target) < 1e-12
    assert relative_residual(apply_map(shifted, repairs['scalar_only']), target) > 1


def test_positive_diagonal_change():
    reference, target = data()
    current = reference * torch.tensor([.1, .4, 3., 1.2, .03]) + 2.5
    repairs = fit_repairs(reference, current)
    assert relative_residual(apply_map(current, repairs['diagonal_stats']), target) < 1e-12
    assert relative_residual(apply_map(current, repairs['paired_diagonal']), target) < 1e-12
    assert relative_residual(apply_map(current, repairs['mean_scalar']), target) > .2


def test_scaled_rotation_and_translation():
    reference, target = data()
    q, _ = torch.linalg.qr(torch.randn(5, 5, generator=torch.Generator().manual_seed(18),
                                     dtype=torch.float64))
    current = .3 * reference @ q + 2.5
    repairs = fit_repairs(reference, current)
    assert relative_residual(apply_map(current, repairs['scaled_procrustes']), target) < 1e-12
    assert relative_residual(apply_map(current, repairs['paired_ridge']), target) < .03


def test_covariance_cannot_identify_rotation_even_when_exactly_matched():
    rng = torch.Generator().manual_seed(19)
    x = torch.randn(300, 4, generator=rng, dtype=torch.float64)
    x -= x.mean(0)
    values, vectors = torch.linalg.eigh(x.T @ x / (len(x) - 1))
    x = x @ ((vectors * values.rsqrt()) @ vectors.T)
    current = -x
    repairs = fit_repairs(x, current)
    aligned = apply_map(current, repairs['covariance'])
    target = (x - x.mean(0)) / x.std(0)
    assert torch.allclose(aligned.T @ aligned, target.T @ target, atol=1e-8)
    assert relative_residual(aligned, target) > 1.99
    assert relative_residual(apply_map(current, repairs['scaled_procrustes']), target) < 1e-12
