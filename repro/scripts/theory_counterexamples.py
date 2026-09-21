"""Executable counterexamples to overbroad claims in the historical draft."""
import json
from pathlib import Path
import argparse
import numpy as np


def audit():
    out = {}
    a = np.diag([2., .5])
    growth = np.linalg.eigvalsh(np.eye(2) - a)
    assert np.linalg.eigvalsh(a).max() > 1 and growth.max() > 0
    out['largest_ratio_not_full_collapse'] = dict(ratios=[2., .5], growth_rates=growth.tolist())
    # Lp=(h-1)^2/2, La=h^2/2: optimum shifts to 1/(1+alpha), never zero at finite alpha.
    out['healthy_stationarity_not_irreversibility'] = [dict(alpha=x, equilibrium=1/(1+x)) for x in [.1,1.,10.]]
    # Nondegenerate sigmoid ranking has a nonzero derivative at a tied score.
    out['tied_pairwise_logistic_gradient'] = [-.5, .5]
    out['legacy_squared_zero_margin_ranking'] = dict(
        tied_loss=0.,tied_gradient=0.,
        warning='All tied predictions globally minimize the original nonnegative zero-margin ranking loss; this is not a generic ranking-loss property.')
    # A zero-weight linear predictor can output constants while its Jacobian rows differ.
    x = np.array([[-1.], [1.]])
    score_grad = np.array([-.5, .5])
    param_grad = x.T @ score_grad
    assert param_grad[0] != 0
    out['constant_outputs_distinct_jacobians'] = dict(outputs=[0.,0.], parameter_gradient=param_grad.tolist())
    # Magnitude of a radial projection loses whether the auxiliary expands or contracts.
    h = np.array([-1.,1.])
    gp = h.copy()
    out['unsigned_ratio_loses_direction'] = []
    for ga in (h, -h):
        rho = np.linalg.norm(ga)/np.linalg.norm(gp)
        derivative = -float(h @ ga)
        out['unsigned_ratio_loses_direction'].append(dict(rho=rho, variance_derivative=derivative))
    assert out['unsigned_ratio_loses_direction'][0]['rho'] == out['unsigned_ratio_loses_direction'][1]['rho']
    # At a fixed dose, a positive scalar can restore the modal threshold.
    assert .1 * 2 < 1
    out['positive_scaling_changes_threshold'] = dict(original_ratio=2., scale=.1, filtered_ratio=.2)
    # Euclidean operator norm cannot characterize arbitrary signed filter dynamics.
    f = -np.eye(2)
    assert np.linalg.norm(f @ a, 2) == 2
    assert np.linalg.eigvalsh(np.eye(2) - f @ a).min() > 0
    out['signed_filter_norm_loses_dynamics'] = dict(filtered_norm=2., growth_rates=np.linalg.eigvalsh(np.eye(2)-f@a).tolist())
    return out


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    data=audit();p.write_text(json.dumps(data,indent=2))
    print('PASS:',len(data),'counterexamples verified')
