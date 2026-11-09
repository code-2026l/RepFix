"""Fixed, label-free affine repairs into a healthy readout's coordinates.

All estimators consume fitting-split features only. Paired estimators require
old and new representations of the same anchors. No estimator sees labels or
chooses a regularizer from evaluation outcomes. These are diagnostic baselines,
not implementations of jointly trained backward-compatible encoders.
"""
import hashlib

import torch


ARMS = ('frozen', 'mean_only', 'scalar_only', 'mean_scalar', 'diagonal_stats',
        'paired_diagonal', 'scaled_procrustes', 'covariance', 'paired_ridge')
PAIRED = {'paired_diagonal', 'scaled_procrustes', 'paired_ridge'}
RIDGE = 1e-3
EIGEN_FLOOR = 1e-6


def _symmetric_power(covariance, power):
    values, vectors = torch.linalg.eigh(covariance)
    floor = EIGEN_FLOOR * values.max().clamp_min(1e-30)
    return (vectors * values.clamp_min(floor).pow(power)) @ vectors.T


@torch.no_grad()
def fit_repairs(reference, current, reference_mean=None, reference_sd=None):
    """Return maps z = h_new @ A + b in healthy standardized coordinates."""
    if reference.shape != current.shape or reference.ndim != 2 or len(reference) < 2:
        raise ValueError('Equal-sized, paired, two-dimensional fitting matrices required')
    ref, new = reference.double(), current.double()
    mu0 = ref.mean(0, keepdim=True) if reference_mean is None else reference_mean.double()
    sd0 = ref.std(0, keepdim=True).clamp_min(1e-6) if reference_sd is None else reference_sd.double()
    mu1 = new.mean(0, keepdim=True)
    sd1 = new.std(0, keepdim=True).clamp_min(1e-6)
    rc, nc = ref - mu0, new - mu1
    d = ref.shape[1]
    rms_ratio = (nc.square().sum() / rc.square().sum().clamp_min(1e-30)).sqrt().clamp_min(1e-15)
    inv_sd0 = torch.diag(1 / sd0.squeeze(0))
    maps = {}

    def add(name, matrix, center):
        maps[name] = dict(matrix=matrix, offset=-center @ matrix,
                          paired_anchors=name in PAIRED)

    add('frozen', inv_sd0, mu0)
    add('mean_only', inv_sd0, mu1)
    # Rescale around the raw origin, then apply the original healthy mean.
    maps['scalar_only'] = dict(matrix=inv_sd0 / rms_ratio, offset=-mu0 @ inv_sd0,
                               paired_anchors=False)
    add('mean_scalar', inv_sd0 / rms_ratio, mu1)
    add('diagonal_stats', torch.diag(1 / sd1.squeeze(0)), mu1)
    slopes = (nc * rc).sum(0) / nc.square().sum(0).clamp_min(1e-30)
    add('paired_diagonal', torch.diag(slopes) @ inv_sd0, mu1)
    u, singular, vh = torch.linalg.svd(nc.T @ rc, full_matrices=False)
    scale = singular.sum() / nc.square().sum().clamp_min(1e-30)
    add('scaled_procrustes', scale * (u @ vh) @ inv_sd0, mu1)
    cov_new = nc.T @ nc / (len(nc) - 1)
    cov_ref = rc.T @ rc / (len(rc) - 1)
    add('covariance', _symmetric_power(cov_new, -0.5) @
        _symmetric_power(cov_ref, 0.5) @ inv_sd0, mu1)
    new_std, ref_std = nc / sd1, rc / sd0
    ridge = torch.linalg.solve(new_std.T @ new_std / len(new_std) + RIDGE *
                               torch.eye(d, dtype=torch.float64),
                               new_std.T @ ref_std / len(new_std))
    add('paired_ridge', torch.diag(1 / sd1.squeeze(0)) @ ridge, mu1)
    assert tuple(maps) == ARMS
    for value in maps.values():
        if not torch.isfinite(value['matrix']).all() or not torch.isfinite(value['offset']).all():
            raise FloatingPointError('Non-finite alignment map')
    return maps


def apply_map(features, affine):
    return features.double() @ affine['matrix'] + affine['offset']


def map_hash(affine):
    digest = hashlib.sha256()
    for key in ('matrix', 'offset'):
        array = affine[key].cpu().contiguous().numpy()
        digest.update(str((key, array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def relative_residual(predicted, reference_standardized):
    return float((predicted - reference_standardized).norm() /
                 reference_standardized.norm().clamp_min(1e-30))
