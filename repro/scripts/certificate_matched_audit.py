"""Local matched-scale audit of archived deterministic certificate arms."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]


def auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return None
    ranks = rankdata(scores, method='average')
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'repro/results/PBA/PBA_har.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'repro/results/CPUCONT/certificate_matched.json')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = [dict(p, B=run['audit_batch']) for run in data['runs'] for p in run['points'] if not p.get('degenerate')]
    arms = {name: [] for name in [
        'frozen_canonical_triangle', 'frozen_optimized_triangle',
        'corrected_canonical_triangle', 'corrected_optimized_triangle',
        'corrected_canonical_combined', 'corrected_optimized_combined',
        'frozen_canonical_exact_residual']}
    free_k_agreements = 0
    for p in rows:
        d, a, alpha, r, c = (p[k] for k in ('d', 'a', 'alpha', 'r', 'c'))
        base = p['nu'] * p['n_p'] / p['B']
        exact = r - alpha * c
        q = p['frozen']['q']
        prediction_frozen = 1 - alpha * q
        prediction_corrected = d - alpha * a
        k_grid = np.logspace(np.log10(base) - 7, np.log10(base) + 7, 241)
        def envelope(k):
            ep, ea = r / (k * base) - d, c / (k * base) - a
            return (abs(d - 1) + abs(ep) + alpha * (abs(a - q) + abs(ea)),
                    abs(ep) + alpha * abs(ea), abs(ep - alpha * ea))
        canonical = envelope(p['k_tr'])
        grid = np.asarray([envelope(k) for k in k_grid])
        specifications = [
            ('frozen_canonical_triangle', prediction_frozen, canonical[0]),
            ('frozen_optimized_triangle', prediction_frozen, float(grid[:, 0].min())),
            ('corrected_canonical_triangle', prediction_corrected, canonical[1]),
            ('corrected_optimized_triangle', prediction_corrected, float(grid[:, 1].min())),
            ('corrected_canonical_combined', prediction_corrected, canonical[2]),
            ('corrected_optimized_combined', prediction_corrected, float(grid[:, 2].min())),
            ('frozen_canonical_exact_residual', prediction_frozen,
             abs(exact / (p['k_tr'] * base) - prediction_frozen))]
        for name, prediction, delta in specifications:
            arms[name].append(dict(fired=bool(abs(prediction) > delta),
                                   agree=bool(prediction * exact > 0),
                                   contracting=bool(exact < 0),
                                   envelope=float(delta),
                                   score=abs(prediction) / max(delta, 1e-300)))
        if prediction_corrected * exact > 0:
            analytic_k = exact / (base * prediction_corrected)
            if not np.isfinite(analytic_k) or analytic_k <= 0:
                raise AssertionError('Positive matching scale must exist')
            free_k_agreements += 1
    summaries = {}
    for name, records in arms.items():
        fired = [p for p in records if p['fired']]
        negatives = [p for p in records if p['contracting']]
        wrong = sum(not p['agree'] for p in fired)
        if wrong:
            raise AssertionError('A sufficient certificate fired wrongly')
        summaries[name] = dict(n=len(records), fired=len(fired), wrong=wrong,
                               agreement=sum(p['agree'] for p in records) / len(records),
                               contracting_n=len(negatives),
                               contracting_fired=sum(p['fired'] for p in negatives),
                               median_envelope_common_N=float(np.median([p['envelope'] for p in records])),
                               retrospective_auc=auc([p['score'] for p in records], [p['agree'] for p in records]))
    out = dict(protocol='matched scale, envelope, and predictor comparison',
               exact_derivative_used_in_all_certificates=True,
               optimization_grid='241 log-spaced k values in base*[1e-7,1e7]',
               unconstrained_corrected_matching_scale_count=free_k_agreements,
               note='For matching signs, k=(r-alpha*c)/(base*(d-alpha*a)) makes the combined geometry residual zero. This is retrospective sign agreement, not prospective prediction.',
               arms=summaries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
