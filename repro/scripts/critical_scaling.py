"""Fine-grained critical scaling sweep for the collapse phase transition.

Goal: resolve sigma_h(alpha) around the critical alpha_c with dense grid
points so we can fit the critical exponent beta via  sigma_h ~ |alpha - alpha_c|^beta
(or its collapsed/healthy branches), and check universality across widths m.

Design for precision + A100 throughput:
  - multiple widths m to test m-independence of beta (universality class)
  - dense log-alpha grid near presumed critical region
  - many seeds for tight error bars
  - records sigma_h, scatter order param, test IC/collapse status per (m, alpha)
"""
import argparse, json, math
import numpy as np
import torch
import collapse_phase as cp


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)


def order_param(h, gp, goals):
    """Geometry-aligned toxicity order parameter rho = <||P_c g_a||>/<||g_p||>,
    P_c = u u^T projector onto batch-mean collapse direction u=(h-Eh)/||h-Eh||.
    Theory: the constant-map fixed point is stable (rep collapses) exactly when
    rho > rho_c = 1, i.e. the toxic projection dominates the primary gradient."""
    hc = h - h.mean(0)
    n = hc.norm(dim=1, keepdim=True).clamp_min(1e-8)
    u = hc / n
    ga = 2.0 * (h - h.mean(0, keepdim=True))            # grad of var_loss wrt h
    ga_c = (ga * u).sum(-1, keepdim=True) * u           # P_c g_a per sample
    den = gp.norm(dim=1)
    return float(((ga_c.norm(dim=1) / (den + 1e-8)).mean()).item())


def _run(alpha, m, k_aux, seed, epochs, lr, n_tr, n_te, d_in=16, probe=True):
    cp.set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    w_true = torch.randn(d_in, generator=g)
    w_true[torch.abs(w_true) < 1.0] = 0.0
    X = torch.randn(n_tr + n_te, d_in, generator=g)
    Y = (X @ w_true) + 0.4 * torch.randn(n_tr + n_te, generator=g)
    Xtr, Ytr = X[:n_tr].to(dev), Y[:n_tr].to(dev)
    Xte, Yte = X[n_tr:].to(dev), Y[n_tr:].to(dev)
    model = cp.CollapseModel(d_in, m, k_aux).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        h, y_hat, a = model(Xtr)
        loss_p = cp.rank_margin_loss(y_hat, Ytr)
        loss_a = cp.var_loss(a) if k_aux > 0 else torch.zeros((), device=dev)
        opt.zero_grad(); (loss_p + alpha * loss_a).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        h, y_hat, a = model(Xte)
        sigma_h = float(h.std(0).mean())
        ic = cp.rank_ic(y_hat.cpu().numpy(), Yte.cpu().numpy())
        cflag = 1.0 if sigma_h < 1e-3 else 0.0
        rho = float("nan")
        if probe and k_aux > 0:
            try:
                gp = torch.autograd.grad(cp.rank_margin_loss(y_hat, Yte), h,
                                         retain_graph=False, allow_unused=True)[0]
                if gp is not None:
                    rho = order_param(h, gp.detach(), Yte)
            except Exception:
                pass
    return dict(alpha=alpha, m=m, seed=seed, sigma_h=sigma_h, test_ic=ic,
                collapsed=cflag, rho=rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="8,16,32,64,128,256")
    ap.add_argument("--alphas", default=None, help="explicit dense alpha grid (csv)")
    ap.add_argument("--lo", type=float, default=-2.5)
    ap.add_argument("--hi", type=float, default=1.5)
    ap.add_argument("--n-alphas", type=int, default=28)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--n-tr", type=int, default=1500)
    ap.add_argument("--n-te", type=int, default=1500)
    ap.add_argument("--out", type=str, default="../results/th_critical.json")
    app = ap.parse_args()
    widths = [int(x) for x in app.widths.split(",")]
    if app.alphas:
        alphas = [float(x) for x in app.alphas.split(",")]
    else:
        alphas = list(np.logspace(app.lo, app.hi, app.n_alphas))
    print(f"[device] cuda A100; widths={widths} n_alphas={len(alphas)} seeds={app.seeds}", flush=True)
    out = []
    for m in widths:
        for alpha in alphas:
            sh, sd, ics, cols, rhos = [], [], [], [], []
            for s in range(app.seeds):
                r = _run(alpha, m, 2, s, app.epochs, app.lr, app.n_tr, app.n_te)
                sh.append(r["sigma_h"]); ics.append(r["test_ic"])
                cols.append(r["collapsed"]); rhos.append(r["rho"])
            rec = dict(m=m, alpha=float(alpha),
                       sigma_h=float(np.mean(sh)), sigma_h_std=float(np.std(sh)),
                       test_ic=float(np.mean(ics)), collapse_rate=float(np.mean(cols)),
                       rho=float(np.nanmean(rhos)),
                       rho_std=float(np.nanstd(rhos) if any(math.isfinite(r) for r in rhos) else 0.0))
            out.append(rec)
            print(f"m={m:3d} a={alpha:9.3e} -> sigma_h={rec['sigma_h']:.5f}+-{rec['sigma_h_std']:.5f} "
                  f"ic={rec['test_ic']:+.4f} coll={rec['collapse_rate']:.2f} "
                  f"rho={rec['rho']:.3f}", flush=True)
    with open(app.out, "w") as f:
        json.dump(dict(records=out, widths=widths, alphas=alphas), f, indent=2)
    print(f"[saved] {app.out}", flush=True)


if __name__ == "__main__":
    main()