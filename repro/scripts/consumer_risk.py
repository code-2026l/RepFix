"""Standard finite-corpus disagreement control for a fixed classifier family.

The monitor is a uniform sample without replacement. Candidate classifiers must
be fixed independently of its membership; a simultaneous tail allocation then
permits candidate selection. This bounds relative 0-1 loss on the remainder,
not absolute accuracy, a new population, or future/adaptive model updates.
"""
from functools import lru_cache

import numpy as np
from scipy.stats import hypergeom


@lru_cache(maxsize=16384)
def upper_total_disagreements(population, sampled, observed, alpha):
    if not (isinstance(population, int) and isinstance(sampled, int) and isinstance(observed, int)):
        raise ValueError('Counts must be integers')
    if not (0 < sampled < population and 0 <= observed <= sampled and 0 < alpha < 1):
        raise ValueError('A nonempty monitor/remainder, valid count, and tail probability are required')
    lower, upper = observed, population - sampled + observed
    while lower < upper:
        middle = (lower + upper + 1) // 2
        # SciPy parameters: total population, total successes, sample size.
        if hypergeom.cdf(observed, population, middle, sampled) >= alpha:
            lower = middle
        else:
            upper = middle - 1
    return lower


def certificate(prediction, healthy_prediction, monitor_indices, alpha):
    prediction, healthy = np.asarray(prediction), np.asarray(healthy_prediction)
    monitor = np.asarray(monitor_indices)
    if prediction.ndim != 1 or prediction.shape != healthy.shape:
        raise ValueError('Equal one-dimensional prediction arrays required')
    if monitor.ndim != 1 or not np.issubdtype(monitor.dtype, np.integer):
        raise ValueError('Integer monitor indices required')
    if len(np.unique(monitor)) != len(monitor) or np.any(monitor < 0) or np.any(monitor >= len(prediction)):
        raise ValueError('Monitor indices must be unique and in range')
    n = len(monitor)
    observed = int(np.count_nonzero(prediction[monitor] != healthy[monitor]))
    upper = upper_total_disagreements(len(prediction), n, observed, float(alpha))
    return dict(population_examples=len(prediction), monitor_examples=n,
                observed_disagreements=observed, monitor_disagreement=observed / n,
                upper_total_disagreements=upper,
                upper_remaining_disagreement=(upper - observed) / (len(prediction) - n),
                allocated_tail_probability=float(alpha))


def first_certified(certificates, budget, order):
    if not 0 <= budget <= 1:
        raise ValueError('Loss budget must be a fraction in [0,1]')
    return next((name for name in order
                 if certificates[name]['upper_remaining_disagreement'] <= budget), None)
