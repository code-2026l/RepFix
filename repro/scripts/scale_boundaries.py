"""Distinct affine-classifier boundaries for increasing and decreasing scale."""
import numpy as np


def affine_scale_boundaries(intercept, signal):
    """Return bias-class exit and healthy-class first-flip thresholds.

    Logits are intercept + c * signal[i], c in [0, 1]. Unique winners are
    required at c=0 and c=1. Bias exit is infinity if the bias class never
    loses for c>0. Shrink-first is zero if the healthy class never loses on
    (0,1). Values at a tie depend on the declared tie-breaking convention.
    """
    b = np.asarray(intercept, dtype=np.float64)
    s = np.asarray(signal, dtype=np.float64)
    if b.ndim != 1 or s.ndim != 2 or s.shape[1] != len(b):
        raise ValueError('Expected intercept [classes] and signal [examples, classes]')
    if not np.isfinite(b).all() or not np.isfinite(s).all():
        raise ValueError('Finite logits are required')
    top = int(b.argmax())
    if np.count_nonzero(b == b[top]) != 1:
        raise ValueError('The zero-scale winner must be unique')
    den = s - s[:, top, None]
    gap = b[top] - b
    exit_candidates = np.full_like(s, np.inf)
    np.divide(gap[None, :], den, out=exit_candidates, where=den > 0)
    bias_exit = exit_candidates.min(axis=1)

    healthy_logits = b[None, :] + s
    winner = healthy_logits.argmax(axis=1)
    row = np.arange(len(s))
    best = healthy_logits[row, winner]
    if np.any((healthy_logits == best[:, None]).sum(axis=1) != 1):
        raise ValueError('The healthy-scale winners must be unique')
    shrink_den = s[row, winner, None] - s
    shrink_num = b[None, :] - b[winner, None]
    candidates = np.zeros_like(s)
    np.divide(shrink_num, shrink_den, out=candidates, where=shrink_den > 0)
    candidates[(shrink_den <= 0) | (candidates <= 0) | (candidates >= 1)] = 0
    shrink_first = candidates.max(axis=1)
    return dict(bias_class=top, healthy_class=winner,
                bias_exit=bias_exit, shrink_first=shrink_first)


def boundary_summary(values):
    """Keep conditional finite and unconditional quantiles explicitly separate."""
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    return dict(n=int(len(v)), n_finite=int(len(finite)),
                finite_fraction=float(len(finite) / len(v)) if len(v) else None,
                finite_median=float(np.median(finite)) if len(finite) else None,
                all_median=float(np.median(v)) if len(v) else None)
