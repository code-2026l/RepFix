"""Audit the exact synthetic auxiliary Hessian and counterexamples, no training."""
import argparse
import json
from pathlib import Path
import torch
from synth_rank_scan import RankModel, var_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    rows = []
    for m in (32, 48, 64, 96, 128, 256):
        for r in (4, 8, 16, 32):
            model = RankModel(16, m, 2, r)
            w = model.W_aux.double()
            # Hessian on the batch-centered subspace, excluding the 1/B factor.
            op = 2 * torch.einsum('kmr,knr->mn', w, w) / (2 * r)
            ev = torch.linalg.eigvalsh(op).clamp_min(0)
            rank = int((ev > ev.max() * 1e-9).sum())
            assert rank == min(m, 2 * r)
            h = torch.randn(7, m, dtype=torch.double, requires_grad=True)
            aux = torch.einsum('bm,kmr->bkr', h, w)
            grad, = torch.autograd.grad(var_loss(aux), h)
            predicted = (h - h.mean(0)) @ op / len(h)
            torch.testing.assert_close(grad, predicted)
            rows.append(dict(m=m, r=r, algebraic_rank=rank,
                trace=float(ev.sum()), op_norm=float(ev.max()),
                frobenius=float(ev.square().sum().sqrt()),
                participation_rank=float(ev.sum().square()/ev.square().sum())))
    # Feature-coordinate variance cannot detect collapse across examples.
    constant = torch.arange(8.).expand(10, -1)
    counterexample = dict(coordinate_variance=float(constant.var(dim=1).mean()),
                          between_sample_variance=float(constant.var(dim=0).mean()))
    assert counterexample['coordinate_variance'] > 0
    assert counterexample['between_sample_variance'] == 0
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(hessian='2/(k*r) sum_k W_k W_k^T',
        rows=rows, constant_encoder_counterexample=counterexample), indent=2))
    print('PASS: exact Hessian autograd checks and ranks in 24 cells; coordinate-scatter counterexample')
    for x in rows:
        if x['r'] == 32:
            print(x)


if __name__ == '__main__':
    main()
