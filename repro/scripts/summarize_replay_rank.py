"""Summarize the replay rank diagnostic on saved full-joint endpoints.

Each cell projects the full-joint first- and/or second-map deltas onto their
best rank-s / rank-r subspaces and scores the official test corpus.  The
projection is a member of the dense low-rank family, so a cell is an
achievable accuracy for that family, not a lower bound on it.  The sparse
cell restricts the first-map delta to its largest k rows instead and is
likewise a member of the sparse-row family.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'results' / 'CPUCONT' / 'replay_rank'


def load(pattern):
    rows = []
    for path in sorted(FOLDER.glob(pattern)):
        value = json.loads(path.read_text())
        rows.extend(value if isinstance(value, list) else [value])
    return rows


def summarize(rows, domain):
    joint = [r['joint'] for r in rows]
    out = dict(domain=domain, seeds=len(rows), joint_mean=float(np.mean(joint)),
               joint_sd=float(np.std(joint, ddof=1)), rank1={}, rank2={}, sparse={}, pair={})
    for key in ('rank1', 'rank2', 'sparse'):
        present = [r for r in rows if key in r]
        if not present:
            continue
        for k in sorted({int(x) for r in present for x in r[key]}):
            gaps = [r[key][str(k)] - r['joint'] for r in present if str(k) in r[key]]
            out[key][k] = dict(mean_gap=float(np.mean(gaps)), min_gap=float(np.min(gaps)),
                               max_gap=float(np.max(gaps)), seeds=len(gaps),
                               positive=int(sum(g > 0 for g in gaps)))
    for k in sorted({x for r in rows for x in r['pair']}):
        if all(k in r['pair'] for r in rows):
            gaps = [r['pair'][k] - r['joint'] for r in rows]
            out['pair'][k] = dict(mean_gap=float(np.mean(gaps)), min_gap=float(np.min(gaps)),
                                  max_gap=float(np.max(gaps)))
    return out


def dense_macs(s, r, d, m, n_classes):
    # s*d first factor, m*s projection, m*m second-map correction, 2*r*m low-rank
    # second update, two folded heads, two biases.
    return s * d + m * s + m * m + 2 * r * m + 2 * n_classes * m + 2 * m


def sparse_macs(k, r, d, m, n_classes):
    # k*d recomputed first rows, k*m correction columns, 2*r*m low-rank second
    # update, one folded head and one bias.
    return k * d + k * m + 2 * r * m + n_classes * m + m


def main():
    har = summarize(load('har_rank.json'), 'har')
    snr = summarize(load('rank_c0*.json'), 'radioml')
    encoders = dict(har=64 * 561 + 64 * 64 + 64 * 64, radioml=256 * 64 + 64 * 64 + 64 * 64)
    dims = dict(har=(561, 64, 6), radioml=(256, 64, 11))
    for domain, summary in (('har', har), ('radioml', snr)):
        d, m, nc = dims[domain]
        summary['encoder_macs'] = encoders[domain]
        summary['dense_cost'] = {key: dict(extra_macs=dense_macs(*map(int, key.split(',')), d, m, nc),
                                           encoder_fraction=dense_macs(*map(int, key.split(',')), d, m, nc) / encoders[domain])
                                 for key in summary['pair']}
        if summary['sparse']:
            summary['sparse_cost'] = {str(k): dict(extra_macs=sparse_macs(k, 8, d, m, nc),
                                                   encoder_fraction=sparse_macs(k, 8, d, m, nc) / encoders[domain])
                                      for k in summary['sparse']}
    out = dict(protocol='rank projection on saved full-joint confirmation endpoints',
               note='projection is an achievable member of the dense low-rank family, not a lower bound',
               har=har, radioml=snr)
    (ROOT / 'results' / 'CPUCONT' / 'replay_rank_summary.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
