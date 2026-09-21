"""Separate rank from curvature in a new synthetic control (not old SYN data).

The sample-normalized auxiliary Hessian is exactly Q diag(lambda) Q^T.
The original loss, primary, optimizer and data generation are reused. Per-seed
records are retained so uncertainty need not be inferred from curve averages.
"""
import argparse
import json
import math
from pathlib import Path
import torch
import synth_rank_scan as old


class ControlledRankModel(old.RankModel):
    normalization = 'eigenvalue'

    def __init__(self, d_in, m, k_aux, r):
        if not 1 <= r < m:
            raise ValueError('control requires 1 <= r < m to avoid rank saturation')
        super().__init__(d_in, m, k_aux, r)
        rng = torch.Generator().manual_seed(731)
        q, _ = torch.linalg.qr(torch.randn(m, r, generator=rng))
        lam = {'eigenvalue': 1., 'trace': 1./r, 'frobenius': 1./math.sqrt(r)}[self.normalization]
        # var_loss divides by r and differentiating squares supplies a factor 2.
        self.W_aux = (q * math.sqrt(r * lam / 2.)).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=32)
    ap.add_argument('--r', type=int, default=8)
    ap.add_argument('--normalization', choices=['eigenvalue', 'trace', 'frobenius'], default='eigenvalue')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--alphas', default='0,0.001,0.01,0.1,1,10')
    ap.add_argument('--epochs', type=int, default=800)
    ap.add_argument('--primary-loss', choices=['legacy', 'logistic'], default='legacy')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    torch.set_num_threads(1)
    ControlledRankModel.normalization = a.normalization
    if a.self_test:
        for norm in ('eigenvalue', 'trace', 'frobenius'):
            ControlledRankModel.normalization = norm
            model = ControlledRankModel(16, 32, 2, 8)
            w = model.W_aux.double()[0]
            ev = torch.linalg.eigvalsh(2 * w @ w.T / 8).clamp_min(0)
            target = {'eigenvalue':1., 'trace':1./8, 'frobenius':1./math.sqrt(8)}[norm]
            torch.testing.assert_close(ev[-8:], torch.full((8,),target,dtype=torch.double), atol=1e-6, rtol=1e-5)
            assert int((ev > 1e-6).sum()) == 8
        print('PASS: exact rank and Hessian amplitude under three independent normalizations')
        return
    if not a.out:
        ap.error('--out required')
    old.RankModel = ControlledRankModel
    if a.primary_loss == 'logistic':
        def logistic_rank(scores, targets):
            differences=scores[:,None]-scores[None,:]
            ordered=(targets[:,None]>targets[None,:]).to(scores.dtype)
            return (ordered*torch.nn.functional.softplus(-differences)).sum()/ordered.sum().clamp_min(1)
        old.rank_margin_loss=logistic_rank
    records = []
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    for alpha in map(float,a.alphas.split(',')):
        for seed in map(int,a.seeds.split(',')):
            row = old.run_one(alpha,a.m,a.r,seed,epochs=a.epochs)
            records.append(row)
            print(row,flush=True)
            path.write_text(json.dumps(dict(experimental=True, m=a.m,r=a.r,
                normalization=a.normalization, epochs=a.epochs,primary_loss=a.primary_loss,
                hessian='Q diag(lambda) Q^T', complete=False,raw=records),indent=2))
    data=json.loads(path.read_text())
    data['complete']=True
    path.write_text(json.dumps(data,indent=2))


if __name__=='__main__':
    main()
