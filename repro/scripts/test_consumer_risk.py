"""Exhaustive small-corpus coverage and sharpness checks, not fitted examples."""
import numpy as np
import pytest
from scipy.stats import hypergeom

from consumer_risk import certificate, first_certified, upper_total_disagreements


@pytest.mark.parametrize('population,sampled', [(9, 3), (13, 5), (17, 12)])
@pytest.mark.parametrize('alpha', [.001, .05, .2])
def test_exact_random_sample_coverage_for_every_finite_population(population, sampled, alpha):
    for true_total in range(population + 1):
        miss = 0.
        for observed in range(max(0, sampled - population + true_total), min(sampled, true_total) + 1):
            if true_total > upper_total_disagreements(population, sampled, observed, alpha):
                miss += hypergeom.pmf(observed, population, true_total, sampled)
        assert miss <= alpha + 1e-12


def test_count_and_confidence_monotonicity():
    bounds = [upper_total_disagreements(100, 25, k, .01) - k for k in range(26)]
    assert bounds == sorted(bounds)
    assert upper_total_disagreements(100, 25, 3, .001) >= upper_total_disagreements(100, 25, 3, .1)
    assert upper_total_disagreements(100, 25, 25, .01) == 100


def test_relative_loss_bound_is_sharp_and_disagreement_does_not_establish_harm():
    healthy = np.array([0, 1, 2, 0, 1, 2])
    updated = np.array([0, 2, 2, 1, 1, 0])
    disagreement = np.mean(updated != healthy)
    for labels, expected in [(healthy, disagreement), (updated, -disagreement)]:
        loss_difference = np.mean(updated != labels) - np.mean(healthy != labels)
        assert loss_difference == expected


def test_selection_uses_only_unlabeled_certificate_and_can_abstain():
    healthy = np.zeros(1000, dtype=int)
    monitor = np.arange(200, dtype=int)
    certs = {name: certificate(prediction, healthy, monitor, .0001)
             for name, prediction in [('frozen', np.ones(1000, dtype=int)), ('repair', healthy)]}
    assert first_certified(certs, .1, ['frozen', 'repair']) == 'repair'
    assert first_certified(certs, 0, ['frozen', 'repair']) is None


def test_prediction_certificate_leaves_temperature_only_nll_drift_uncontrolled():
    healthy_logits = np.tile([10., 0.], (3000, 1))
    changed_logits = healthy_logits * .001
    prediction = changed_logits.argmax(1)
    healthy_prediction = healthy_logits.argmax(1)
    c = certificate(prediction, healthy_prediction, np.arange(1024), .05 / 3015)
    assert c['observed_disagreements'] == 0 and c['upper_remaining_disagreement'] < .05
    healthy_nll = np.logaddexp.reduce(healthy_logits, axis=1).mean() - 10
    changed_nll = np.logaddexp.reduce(changed_logits, axis=1).mean() - .01
    assert changed_nll - healthy_nll > .6


@pytest.mark.parametrize('counts', [(10, 0, 0, .05), (10, 10, 0, .05),
                                   (10, 5, 6, .05), (10, 5, 0, 0)])
def test_invalid_count_designs_rejected(counts):
    with pytest.raises(ValueError):
        upper_total_disagreements(*counts)
