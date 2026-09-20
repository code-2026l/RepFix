"""
Collapse phase-transition testbed (self-contained).

Faithful minimal model of the two-layer collapse mechanism:

  Layer-1 (encoder collapse):
      shared encoder E: x -> h
      K auxiliary heads = FIXED random projections W_k (non-learnable) with
          L_aux = Var_batch(W_k h)   -> minimized when h is constant
      Because W_k is fixed, the ONLY way an auxiliary head lowers its loss is
      to drag the shared encoder toward a zero-variance (constant) representation.

  Layer-2 (prediction-level trap):
      primary head: y_hat = w_p^T h + b
      primary loss = pairwise quadratic-margin RANK loss:
          L_p = sum_{i beats j} [y_hat_j - y_hat_i]_+^2
      This loss is *zero* BOTH at a perfect ranking AND at a constant y_hat
      (all margins zero).  A constant output is therefore a degenerate global
      minimizer with zero gradient -- the "prediction-level trap".  Once the
      encoder collapses, the rank loss gives no gradient to pull it back out.

Sweep the toxicity control parameter (order-parameter proxy):
  (A) alpha := auxiliary gradient scaling
  (B) K     := number of auxiliary heads (task count)

and measure:
  - sigma_h   : std of the shared representation (order parameter)
  - test_IC   : rank-IC of y_hat vs y on a held-out set (health metric)
  - test_mse  : MSE of y_hat vs z-scored y on the held-out set
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def _rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    sa = np.asarray(a, dtype=np.float64)
    sorted_a = sa[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def rank_ic(pred, y):
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if pred.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(_rankdata(pred), _rankdata(y))[0, 1])


def zscore(t):
    m = t.mean()
    s = t.std()
    return (t - m) / (s + 1e-8)


def rank_margin_loss(s, y):
    # pairwise quadratic-margin rank loss. s,y: (n,).
    # D[i,j] = s_i - s_j ;  i "beats" j iff y_i > y_j.
    D = s.unsqueeze(1) - s.unsqueeze(0)       # (n,n)
    P = (y.unsqueeze(1) > y.unsqueeze(0)).float()  # (n,n) indicator i beats j
    margin = torch.relu(-D)                    # wrong order -> s_j - s_i > 0
    npos = P.sum().clamp_min(1.0)
    return (P * margin * margin).sum() / npos


class CollapseModel(nn.Module):
    def __init__(self, d_in, m, k_aux):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, m), nn.ReLU(), nn.Linear(m, m))
        self.primary = nn.Linear(m, 1)
        g = torch.Generator().manual_seed(1234)
        W = torch.randn(k_aux, m, m // 2, generator=g)
        W = W / (m // 2) ** 0.5
        self.register_buffer("W_aux", W)

    def forward(self, x):
        h = self.encoder(x)                       # (B, m)
        y_hat = self.primary(h).squeeze(-1)       # (B,)
        a = torch.einsum("bm,kmn->bkn", h, self.W_aux)  # (B, k_aux, m//2)
        return h, y_hat, a


def var_loss(a):
    # a: (B, k, d) -> batch variance, averaged over heads and dims
    return ((a - a.mean(0)) ** 2).mean()


def train_healthy(d_in, m, k_aux, seed, epochs=800, lr=3e-2, n_tr=1500):
    set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    w_true = torch.randn(d_in, generator=g)
    w_true[torch.abs(w_true) < 1.0] = 0.0
    X = torch.randn(n_tr, d_in, generator=g)
    Y = (X @ w_true) + 0.4 * torch.randn(n_tr, generator=g)
    Xtr, Ytr = X.to(dev), Y.to(dev)
    model = CollapseModel(d_in, m, k_aux).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        h, y_hat, a = model(Xtr)
        loss = rank_margin_loss(y_hat, Ytr)  # alpha=0: healthy objective only
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, Xtr, Ytr


def _mean_abs_proj(grad, unit_dir):
    # per-sample signed projection onto a per-sample unit direction,
    # averaged in absolute value: dimension-free, robust to small gradients.
    p = (grad * unit_dir).sum(-1)          # (B,) signed projection per sample
    return float(p.abs().mean())           # mean |projection|


def measure_rho(model, Xtr, Ytr, eps: float = 1e-8, rho_def: str = "norm"):
    """Toxicity order parameter of Definition 3:  rho = ||Pi_c g_a|| / ||g_p||.

    Pi_c is the orthogonal projector onto the batch-centred variance-contracting
    direction u_c = h - E_b[h]; the numerator is the norm of the toxic
    (variance-contracting) component of the auxiliary gradient and the
    denominator the primary gradient norm.  This is the convention the paper
    states in Definition 3 (eq. order), in the abstract, and in the proofs of
    Theorem 1 and Theorem 2, and the one the real-domain probe uses
    (repro/scripts/cross_domain_mtl.py: rho_probe, "paper convention").

    Two readings of Definition 3 ship, selected by `rho_def`:

      "norm"    (default)  proj_norm / ||g_p||  -- batch-level reading of
                Definition 3, the estimator behind repro/results/th_rho.json
                and repro/results/th_rho12.json and behind the paper's
                rho = 0.74 +/- 0.19, alpha_c = 1.45 +/- 0.37.
      "percell" <|Pi_c g_a|>_samples / <|Pi_c g_p|>_samples -- the per-sample
                mean-absolute-projection ratio introduced with the CFG_AUX
                gate fix.  Its denominator is the *projected* primary gradient,
                not ||g_p||, so it is NOT Definition 3 and not the paper's rho;
                it is retained because repro/results/th_rho_v2.json was
                produced with it.
    """
    model.train()
    h, y_hat, a = model(Xtr)
    loss_p = rank_margin_loss(y_hat, Ytr)
    loss_a = var_loss(a)
    gp = torch.autograd.grad(loss_p, h, retain_graph=True)[0]   # (B, m)
    ga = torch.autograd.grad(loss_a, h, retain_graph=True)[0]   # alpha=1 gradient

    gp_norm = float(gp.norm())
    ga_norm = float(ga.norm())
    uc = h - h.mean(0)                                # (B, m) centred

    if rho_def == "percell":
        ucn = uc.norm(dim=1, keepdim=True).clamp_min(eps)   # (B, 1)
        uhat = uc / ucn                                     # unit dir (B, m)
        pi_ga = _mean_abs_proj(ga, uhat)                    # aux projection
        pi_gp = _mean_abs_proj(gp, uhat)                    # primary projection
        rho_proj = pi_ga / (pi_gp + eps)
        rho_full = ga_norm / (gp_norm + eps)
        return dict(gp_norm=gp_norm, ga_norm=ga_norm,
                    proj_aux=pi_ga, proj_pri=pi_gp,
                    rho_proj=rho_proj, rho_full=rho_full,
                    pred_alpha_c_proj=1.0 / (rho_proj + eps),
                    pred_alpha_c_full=1.0 / (rho_full + eps))

    # Definition 3: rho = ||Pi_c g_a|| / ||g_p||
    ucn = uc.norm()
    if ucn.item() > 1e-8:
        proj_norm = float(torch.abs((ga * uc).sum() / ucn))
        # also the primary gradient's component along u_c (for a signed ratio)
        gp_proj = float((gp * uc).sum() / ucn)
    else:
        proj_norm = 0.0
        gp_proj = 0.0
    rho_proj = proj_norm / (gp_norm + 1e-12)     # ‖Π_c g_aux‖ / ‖g_p‖
    rho_full = ga_norm / (gp_norm + 1e-12)
    return dict(gp_norm=gp_norm, ga_norm=ga_norm, proj_norm=proj_norm,
                rho_proj=rho_proj, rho_full=rho_full,
                pred_alpha_c_proj=1.0 / (rho_proj + 1e-12),
                pred_alpha_c_full=1.0 / (rho_full + 1e-12))


def run_rho(seed, d_in=16, m=64, k_aux=2, epochs=800, lr=3e-2, rho_def="norm"):
    model, Xtr, Ytr = train_healthy(d_in, m, k_aux, seed, epochs, lr)
    return measure_rho(model, Xtr, Ytr, rho_def=rho_def)


def init_data(seed, d_in=16, n_tr=1500):
    set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    w_true = torch.randn(d_in, generator=g)
    w_true[torch.abs(w_true) < 1.0] = 0.0
    X = torch.randn(n_tr, d_in, generator=g)
    Y = (X @ w_true) + 0.4 * torch.randn(n_tr, generator=g)
    return X.to(dev), Y.to(dev)


def run_rho_full(seed, d_in=16, m=64, k_aux=2, epochs=800, lr=3e-2,
                 rho_def="norm"):
    # rho at initialization (clean alpha_c = 1/rho prediction)
    Xtr, Ytr = init_data(seed, d_in)
    dev = Xtr.device
    model_init = CollapseModel(d_in, m, k_aux).to(dev)
    ri = measure_rho(model_init, Xtr, Ytr, rho_def=rho_def)
    # rho after healthy (alpha=0) training reaches its fixed point
    model, _, _ = train_healthy(d_in, m, k_aux, seed, epochs, lr)
    rh = measure_rho(model, Xtr, Ytr, rho_def=rho_def)
    with torch.no_grad():
        h, _, _ = model(Xtr)
        sigma_h_healthy = float(h.std(0).mean())
    rec = {
        "seed": seed,
        "rho_proj_init": ri["rho_proj"],
        "rho_full_init": ri["rho_full"],
        "alpha_c_init": 1.0 / (ri["rho_proj"] + 1e-12),
        "alpha_c_fit": 1.0 / (ri["rho_full"] + 1e-12),
        "rho_proj_conv": rh["rho_proj"],
        "rho_full_conv": rh["rho_full"],
        "alpha_c_conv": 1.0 / (rh["rho_proj"] + 1e-12),
        "gp_norm_init": ri["gp_norm"],
        "ga_norm_init": ri["ga_norm"],
        "sigma_h_healthy": sigma_h_healthy,
    }
    if rho_def == "percell":
        rec["proj_aux_init"] = ri["proj_aux"]
        rec["proj_pri_init"] = ri["proj_pri"]
    else:
        rec["proj_norm_init"] = ri["proj_norm"]
    return rec


def run_one(alpha, k_aux, seed, d_in=16, m=64, n_tr=1500, n_te=1500,
            epochs=800, lr=3e-2):
    set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    w_true = torch.randn(d_in, generator=g)
    w_true[torch.abs(w_true) < 1.0] = 0.0

    X = torch.randn(n_tr + n_te, d_in, generator=g)
    Y = (X @ w_true) + 0.4 * torch.randn(n_tr + n_te, generator=g)
    Xtr, Ytr = X[:n_tr].to(dev), Y[:n_tr].to(dev)
    Xte, Yte = X[n_tr:].to(dev), Y[n_tr:].to(dev)

    model = CollapseModel(d_in, m, k_aux).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        h, y_hat, a = model(Xtr)
        loss_p = rank_margin_loss(y_hat, Ytr)
        loss_a = var_loss(a) if k_aux > 0 else torch.zeros((), device=dev)
        loss = loss_p + alpha * loss_a
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        h, y_hat, a = model(Xte)
        sigma_h = float(h.std(0).mean())
        y_hat_np = y_hat.cpu().numpy()
        yte_np = Yte.cpu().numpy()
        ic = rank_ic(y_hat_np, yte_np)
        yz = zscore(Yte).cpu().numpy()
        mse = float(np.mean((y_hat_np - yz) ** 2))
        cflag = 1.0 if sigma_h < 1e-3 else 0.0
    return dict(alpha=alpha, k=k_aux, seed=seed, sigma_h=sigma_h,
                test_ic=ic, test_mse=mse, collapsed=cflag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["alpha", "tasks", "rho", "capacity"],
                    default="alpha")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--out", type=str, default="collapse_phase.json")
    ap.add_argument("--rho-def", choices=["norm", "percell"], default="norm",
                    help="rho estimator: 'norm' = Definition 3, "
                         "||Pi_c g_a||/||g_p|| (the paper's convention, default); "
                         "'percell' = per-sample mean-absolute-projection ratio "
                         "(legacy, reproduces results/th_rho_v2.json)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print(f"[device] {dev} : {name}", flush=True)

    if args.sweep == "rho":
        recs = [run_rho_full(seed=s, rho_def=args.rho_def)
                for s in range(args.seeds)]
        keys = ["rho_proj_init", "rho_full_init", "alpha_c_init", "alpha_c_fit",
                "rho_proj_conv", "rho_full_conv", "alpha_c_conv",
                "gp_norm_init", "ga_norm_init",
                "proj_norm_init" if args.rho_def == "norm" else "proj_aux_init",
                "proj_pri_init" if args.rho_def == "percell" else "sigma_h_healthy",
                "sigma_h_healthy"]
        keys = list(dict.fromkeys(keys))
        summary = {k: dict(mean=float(np.mean([r[k] for r in recs])),
                           std=float(np.std([r[k] for r in recs])))
                   for k in keys}
        for r in recs:
            print(f"seed={r['seed']} rho_proj_init={r['rho_proj_init']:.3f} "
                  f"alpha_c_init={r['alpha_c_init']:.5f} "
                  f"rho_proj_conv={r['rho_proj_conv']:.3f} "
                  f"alpha_c_conv={r['alpha_c_conv']:.5f} "
                  f"sigma_h_healthy={r['sigma_h_healthy']:.4f}", flush=True)
        out = {"records": recs, "summary": summary, "rho_def": args.rho_def}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {args.out}", flush=True)
        print(f"[summary] rho_def={args.rho_def} "
              f"rho_proj_init = {summary['rho_proj_init']['mean']:.6f} "
              f"± {summary['rho_proj_init']['std']:.6f} | "
              f"alpha_c_init = {summary['alpha_c_init']['mean']:.5f} "
              f"± {summary['alpha_c_init']['std']:.5f}", flush=True)
        return

    if args.sweep == "capacity":
        # E3 capacity scaling law: collapse tendency vs encoder width m.
        # 2D grid (m x alpha); derive alpha_c(m) = first alpha where the
        # majority of seeds collapsed, giving the "larger capacity -> lower
        # critical toxicity" scaling curve.
        capacities = [8, 16, 32, 64, 128, 256]
        alphas = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        k_aux = 2
        results = []
        for m in capacities:
            for a in alphas:
                agg = {"sigma_h": [], "test_ic": [], "test_mse": [], "collapsed": []}
                for s in range(args.seeds):
                    r = run_one(a, k_aux, seed=s, d_in=16, m=m)
                    for kk in agg:
                        agg[kk].append(r[kk])
                rec = {
                    "m": m, "alpha": a,
                    "sigma_h": float(np.mean(agg["sigma_h"])),
                    "sigma_h_std": float(np.std(agg["sigma_h"])),
                    "test_ic": float(np.mean(agg["test_ic"])),
                    "test_mse": float(np.mean(agg["test_mse"])),
                    "collapse_rate": float(np.mean(agg["collapsed"])),
                }
                results.append(rec)
                print(f"m={m:3d} alpha={a:6.3f} | sigma_h={rec['sigma_h']:.4f} "
                      f"testIC={rec['test_ic']:+.4f} collapse={rec['collapse_rate']:.2f}",
                      flush=True)
        # empirical critical alpha per capacity (first crossing >=50% collapse)
        alpha_c = {}
        for m in capacities:
            row = [r for r in results if r["m"] == m]
            ac = None
            for r in row:
                if r["collapse_rate"] >= 0.5:
                    ac = r["alpha"]
                    break
            alpha_c[str(m)] = ac
        out = {"records": results, "alpha_c_per_capacity": alpha_c,
               "capacities": capacities, "alphas": alphas}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {args.out}", flush=True)
        print(f"[alpha_c(m)] {json.dumps(alpha_c)}", flush=True)
        return

    if args.sweep == "alpha":
        alphas = [0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        k_aux = 2
        rows = [(a, k_aux) for a in alphas]
    else:
        alphas = [0.3]
        ks = [0, 1, 2, 3, 5, 8, 12]
        rows = [(alphas[0], k) for k in ks]

    results = []
    for alpha, k in rows:
        agg = {"sigma_h": [], "test_ic": [], "test_mse": [], "collapsed": []}
        for s in range(args.seeds):
            r = run_one(alpha, k, seed=s)
            for kk in agg:
                agg[kk].append(r[kk])
        rec = {
            "alpha": alpha, "k_aux": k,
            "sigma_h": float(np.mean(agg["sigma_h"])),
            "sigma_h_std": float(np.std(agg["sigma_h"])),
            "test_ic": float(np.mean(agg["test_ic"])),
            "test_ic_std": float(np.std(agg["test_ic"])),
            "test_mse": float(np.mean(agg["test_mse"])),
            "collapse_rate": float(np.mean(agg["collapsed"])),
        }
        results.append(rec)
        print(f"alpha={alpha:6.2f} k={k:2d} | sigma_h={rec['sigma_h']:.4f}±{rec['sigma_h_std']:.4f} "
              f"testIC={rec['test_ic']:+.4f}±{rec['test_ic_std']:.4f} "
              f"mse={rec['test_mse']:.4f} collapse={rec['collapse_rate']:.2f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()