"""calib_cert_b.py -- finite-batch calibration certificate (Propositions 6 and 7).

The calibration probe reads the order parameter rho from ONE minibatch of size
B, and Propositions 6 and 7 quantify how far that one-shot read can sit from the
population value.  This script measures the sampling law instead of asserting
it:

  1. train the HEALTHY anchor exactly as the calibrated filter does (a short
     primary-only warm-up, auxiliary branch detached);
  2. freeze it and re-run the shipped probe on independent minibatches drawn
     from the same training pool, over a geometric sweep of B;
  3. check Var(rho_hat) = rho^2 c0^2 / B (Prop. 6) as the log-log slope of the
     estimator's spread against B, which must be -1/2;
  4. check P(|rho_hat - rho| > eps) <= 2 exp(-2 B eps^2 / c^2) with the uniform
     envelope c = 2 G (p_min + G) / p_min^2 (Prop. 7), and the sample-size
     prescription B ~ 2 c0^2 eta^-2 ln(2/delta) at relative accuracy eps = eta rho.

Three estimators of the same order parameter are evaluated side by side, so the
released code and the propositions can be compared term by term:

  rho_ps : mean_i ||a_i|| / ||p_i||            per-cell ratios (shipped probe)
  rho_bm : ||mean_i a_i|| / ||mean_i p_i||     batch-mean ratio
  rho_nm : mean_i ||a_i|| / mean_i ||p_i||     ratio of mean norms

with a_i = Pi_c g_a^(i) the projected per-cell auxiliary gradient and
p_i = g_p^(i) the per-cell primary gradient.  For the variance-shrinkage
toxifier sum_i g_a^(i) = 0 exactly, so rho_bm is identically zero -- the reason
the shipped probe averages per-cell ratios rather than batch means, and the
reason Proposition 6/7 state the estimator in the per-cell form.

Each estimator is computed twice: with the batch-mean centring the shipped probe
uses (suffix _ps), and with a centring by the population mean of h over the whole
training pool (suffix _pop).  The second removes the shared-batch-mean dependence
that couples the cells, so its cells are genuinely i.i.d. and the B^-1/2 law is
tested without that lower-order correction.

Usage:
  python calib_cert_b.py --domain har --m 64 --warm-epochs 4 --seeds 0,1,2 \
      --reps 2000 --out results/CERT/calib_cert_har_m64.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_domain_mtl as C  # noqa: E402


def warmup_anchor(domain, m, seed, warm_epochs, lr, batch, threads, dev):
    """Primary-only warm-up = the calibration warm-up of Section 4.

    The calibrated filter trains with the auxiliary branch detached for
    `cal_warm_epochs` epochs and probes at the first step after that, so the
    anchor is the encoder after exactly `warm_epochs` epochs of primary loss.
    """
    torch.set_num_threads(threads)
    C.set_seed(seed)
    D = C.load_domain(domain)
    regress = D["n_classes"] == 1
    model = C.MTLModel(D["d_in"], D["n_classes"], m=m, k_aux=3, seed=seed,
                       regress=regress, y_dim=int(D.get("y_dim", 1))).to(dev)
    X = torch.from_numpy(D["X_tr"]).float().to(dev)
    y = (torch.from_numpy(D["y_tr"]).float() if regress
         else torch.from_numpy(D["y_tr"]).long()).to(dev)
    criterion = nn.MSELoss() if regress else nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    for _ in range(warm_epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            opt.zero_grad()
            out = model.primary(model(X[idx]))
            loss = criterion(out.squeeze(-1) if regress else out, y[idx])
            loss.backward()
            opt.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # population mean of h over the whole training pool, for the i.i.d. variant
    with torch.no_grad():
        hbar = torch.zeros(m, device=dev)
        for b0 in range(0, n, 2048):
            hbar += model(X[b0:b0 + 2048]).sum(0)
        hbar = (hbar / n).detach()
    return model, X, y, criterion, regress, D, hbar


@torch.no_grad()
def probe_all(model, X, y, criterion, regress, idx, hbar_pop):
    """The shipped probe plus its alternative aggregations and centreings."""
    xb = X[idx]
    yb = y[idx]
    with torch.enable_grad():
        h = model(xb).detach().float().requires_grad_(True)
        pk = model.primary(h)
        if regress:
            pk = pk.squeeze(-1)
        yt = yb.float() if regress else yb
        gp = torch.autograd.grad(criterion(pk, yt), h, retain_graph=True,
                                 allow_unused=True)[0]
        mu = h.mean(0, keepdim=True)
        ga_b = torch.autograd.grad(((h - mu) ** 2).mean(), h, retain_graph=True,
                                   allow_unused=True)[0]
        ga_p = torch.autograd.grad(((h - hbar_pop) ** 2).mean(), h,
                                   retain_graph=False, allow_unused=True)[0]
    out = {}
    gp_n = gp.norm(dim=1)
    gm = gp.mean(0)
    for tag, ga, ref in (("ps", ga_b, mu), ("pop", ga_p, hbar_pop)):
        hc = h.detach() - ref
        nrm = hc.norm(dim=1, keepdim=True).clamp_min(1e-8)
        u = hc / nrm
        a = ((ga * u).sum(-1, keepdim=True)) * u
        a_n = a.norm(dim=1)
        R = a_n / gp_n.clamp_min(1e-12)
        out[f"rho_{tag}"] = float(R.mean())
        out[f"R_{tag}"] = R.cpu().numpy()
        out[f"a_{tag}"] = a_n.cpu().numpy()
        out[f"p_{tag}"] = gp_n.cpu().numpy()
        out[f"aux_sum_rel_{tag}"] = float(
            ga.sum(0).norm() / ga.norm(dim=1).sum().clamp_min(1e-12))
        if tag == "ps":
            out["rho_bm"] = float(a.mean(0).norm() / gm.norm().clamp_min(1e-12))
            out["rho_nm"] = float(a_n.mean() / gp_n.mean().clamp_min(1e-12))
    return out


def fit_slope(xs, ys):
    """OLS slope of log|y| on log x, with R^2; None if degenerate."""
    lx = np.log(np.asarray(xs, float))
    ly = np.log(np.abs(np.asarray(ys, float)))
    m = np.isfinite(lx) & np.isfinite(ly)
    if m.sum() < 3:
        return None
    lx, ly = lx[m], ly[m]
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ sol
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return dict(slope=float(sol[0]), intercept=float(sol[1]),
                r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="har")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--warm-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256,
                    help="training batch = the probe batch B of the shipped runs")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--bs", default="16,32,64,128,256,512,1024,2048,4096")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--replace", action="store_true",
                    help="draw cells WITH replacement, so the B cells are strictly "
                         "i.i.d. and the proposition's sampling model holds exactly; "
                         "without it the cells are a without-replacement subsample of "
                         "the finite training pool, which carries a finite-population "
                         "correction at B comparable to n (the operational case)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Bs = [int(x) for x in a.bs.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    dev = a.device
    rows, per_seed = [], []

    for seed in seeds:
        model, X, y, criterion, regress, D, hbar = warmup_anchor(
            a.domain, a.m, seed, a.warm_epochs, a.lr, a.batch, a.threads, dev)
        n = X.shape[0]
        g = torch.Generator().manual_seed(1000 + seed)
        recs = []
        for B in Bs:
            if B > n:
                continue
            acc = {k: [] for k in ("rho_ps", "rho_pop", "rho_bm", "rho_nm",
                                   "aux_sum_rel_ps", "aux_sum_rel_pop")}
            c0_reps = {"ps": [], "pop": []}
            an_ps, an_pop, pn_all = [], [], []
            R_ps_all, R_pop_all = [], []
            for _ in range(a.reps):
                idx = (torch.randint(0, n, (B,), generator=g).to(dev)
                       if a.replace else torch.randperm(n, generator=g)[:B].to(dev))
                o = probe_all(model, X, y, criterion, regress, idx, hbar)
                for k in acc:
                    acc[k].append(o[k])
                an_ps.append(o["a_ps"])
                an_pop.append(o["a_pop"])
                pn_all.append(o["p_ps"])
                R_ps_all.append(o["R_ps"])
                R_pop_all.append(o["R_pop"])
                for tag in ("ps", "pop"):
                    R = o[f"R_{tag}"]
                    if R.mean() != 0:
                        c0_reps[tag].append(float(R.std(ddof=0) / abs(R.mean())))
            an_ps = np.concatenate(an_ps)
            an_pop = np.concatenate(an_pop)
            pn_all = np.concatenate(pn_all)
            R_ps_all = np.concatenate(R_ps_all)
            R_pop_all = np.concatenate(R_pop_all)
            ps = np.asarray(acc["rho_ps"], float)
            pop = np.asarray(acc["rho_pop"], float)
            rec = dict(
                B=int(B), reps=int(a.reps),
                rho_ps_mean=float(ps.mean()), rho_ps_sd=float(ps.std(ddof=0)),
                rho_ps_cv=float(ps.std(ddof=0) / max(abs(ps.mean()), 1e-300)),
                rho_pop_mean=float(pop.mean()), rho_pop_sd=float(pop.std(ddof=0)),
                rho_pop_cv=float(pop.std(ddof=0) / max(abs(pop.mean()), 1e-300)),
                rho_bm_mean=float(np.mean(acc["rho_bm"])),
                rho_bm_sd=float(np.std(acc["rho_bm"], ddof=0)),
                rho_nm_mean=float(np.mean(acc["rho_nm"])),
                aux_sum_rel_ps=float(np.max(acc["aux_sum_rel_ps"])),
                aux_sum_rel_pop=float(np.max(acc["aux_sum_rel_pop"])),
                c0_ps=float(np.mean(c0_reps["ps"])) if c0_reps["ps"] else None,
                c0_pop=float(np.mean(c0_reps["pop"])) if c0_reps["pop"] else None,
                G_p=float(pn_all.max()), p_min=float(pn_all.min()),
                G_a_ps=float(an_ps.max()),
                G_a_pop=float(an_pop.max()),
            )
            G = max(rec["G_a_ps"], rec["G_p"])
            rec["c_envelope"] = 2 * G * (rec["p_min"] + G) / rec["p_min"] ** 2
            rec["c_percell"] = 2 * G / rec["p_min"]
            rec["R_max_ps"] = float(R_ps_all.max())
            rec["R_max_pop"] = float(R_pop_all.max())
            for eta in (0.05, 0.1, 0.2):
                for tag, arr in (("ps", ps), ("pop", pop)):
                    mu = float(arr.mean())
                    eps = eta * abs(mu)
                    emp = float(np.mean(np.abs(arr - mu) > eps))
                    rec[f"exceed_{tag}_eta{eta}"] = emp
                    for ctag, c in (("env", rec["c_envelope"]),
                                    ("cell", rec["c_percell"])):
                        rec[f"bound_{tag}_eta{eta}_{ctag}"] = float(
                            2.0 * math.exp(min(0.0, -2.0 * B * eps ** 2 / c ** 2)))
                    # Prop. 7 (Bernstein): envelope from the probe's own moments
                    c0 = rec[f"c0_{tag}"] or 0.0
                    Rmax = rec[f"R_max_{tag}"]
                    kap2 = 2.0 * mu ** 2 * c0 ** 2 + (2.0 / 3.0) * Rmax * eps
                    rec[f"bound_{tag}_eta{eta}_bern"] = float(
                        2.0 * math.exp(min(0.0, -B * eps ** 2 / max(kap2, 1e-300))))
            rows.append(rec)
            recs.append(rec)
        per_seed.append(dict(seed=seed, n_train=int(n), rows=recs))

    fit = {}
    for tag, key in (("sd_ps", "rho_ps_sd"), ("cv_ps", "rho_ps_cv"),
                     ("sd_pop", "rho_pop_sd"), ("cv_pop", "rho_pop_cv"),
                     ("mean_ps", "rho_ps_mean"), ("mean_pop", "rho_pop_mean"),
                     ("mean_bm", "rho_bm_mean")):
        fit[tag] = fit_slope([r["B"] for r in rows], [r[key] for r in rows])

    # Prop. 6 constant from the CV fit: CV ~ c0 / sqrt(B)
    c0_fit = {}
    for tag in ("ps", "pop"):
        f = fit[f"cv_{tag}"]
        c0_fit[tag] = float(math.exp(f["intercept"])) if f else None

    # Prop. 7 sample-size prescription, relative accuracy eta = 0.1
    zq = {0.1: 1.6448536269514722, 0.01: 2.5758293035489004,
          1e-3: 3.2905267314919255}
    presc = {}
    for tag, c0 in c0_fit.items():
        if not c0:
            continue
        presc[tag] = {
            f"delta_{d}": dict(
                bound_B=c0 ** 2 / (2 * 0.1 ** 2) * math.log(2 / d),
                gaussian_B=(c0 * zv / 0.1) ** 2)
            for d, zv in zq.items()}
    out = dict(
        domain=a.domain, kind="calib_cert_b", m=int(a.m), seeds=seeds,
        warm_epochs=int(a.warm_epochs), lr=a.lr, batch=int(a.batch),
        alpha=a.alpha, reps=int(a.reps), Bs=Bs, device=dev,
        with_replacement=bool(a.replace),
        threads=int(a.threads), fit=fit, c0_from_cv_fit=c0_fit,
        prescription=presc,
        delta_ratio_bound=math.log(2 / 1e-3) / math.log(2 / 0.1),
        delta_ratio_gaussian=(zq[1e-3] / zq[0.1]) ** 2,
        rows=rows, per_seed=per_seed,
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"=== calib_cert_b | {a.domain} m={a.m} seeds={seeds} "
          f"warm={a.warm_epochs}ep reps={a.reps} dev={dev} "
          f"cells={'iid(w/ replacement)' if a.replace else 'subsample(no replacement)'} ===")
    print(f"{'B':>6} {'rho_ps':>11} {'sd_ps':>10} {'CV_ps':>9} "
          f"{'rho_pop':>11} {'sd_pop':>10} {'CV_pop':>9} {'rho_bm':>10} "
          f"{'c0_ps':>7} {'c0_pop':>7}")
    for r in rows:
        print(f"{r['B']:>6} {r['rho_ps_mean']:>11.5g} {r['rho_ps_sd']:>10.4g} "
              f"{r['rho_ps_cv']:>9.4g} {r['rho_pop_mean']:>11.5g} "
              f"{r['rho_pop_sd']:>10.4g} {r['rho_pop_cv']:>9.4g} "
              f"{r['rho_bm_mean']:>10.3g} {r['c0_ps']:>7.4g} {r['c0_pop']:>7.4g}")
    print("\n--- Prop. 6 log-log fits (slope must be -0.5) ---")
    for k, v in fit.items():
        if v:
            print(f"  {k:<9} slope={v['slope']:+.4f}  R2={v['r2']:.4f}")
    print(f"\n  c0 from CV fit: ps={c0_fit['ps']}  pop={c0_fit['pop']}")
    print(f"  delta 0.1 -> 1e-3: bound-form x{out['delta_ratio_bound']:.3f}, "
          f"Gaussian-truth x{out['delta_ratio_gaussian']:.3f}")
    print("\n--- Prop. 7 bound check (eta=0.1) ---")
    print(f"{'B':>6} {'exceed_ps':>10} {'bern_ps':>9} {'env_ps':>9} "
          f"{'exceed_pop':>11} {'bern_pop':>9} {'env_pop':>9}")
    for r in rows:
        print(f"{r['B']:>6} {r['exceed_ps_eta0.1']:>10.4f} "
              f"{r['bound_ps_eta0.1_bern']:>9.3g} {r['bound_ps_eta0.1_env']:>9.3g} "
              f"{r['exceed_pop_eta0.1']:>11.4f} "
              f"{r['bound_pop_eta0.1_bern']:>9.3g} {r['bound_pop_eta0.1_env']:>9.3g}")
    print("\n--- Bernstein bound vs empirical tail (tag=ps) ---")
    for e in (0.05, 0.1, 0.2):
        print(f"  eta={e}")
        for r in rows:
            print(f"    B={r['B']:>5}  exceed={r[f'exceed_ps_eta{e}']:>8.4f}  "
                  f"bern={r[f'bound_ps_eta{e}_bern']:>10.4g}  "
                  f"mcdiarmid_env={r[f'bound_ps_eta{e}_env']:>10.4g}")
    print(f"\n  max relative |sum_i g_a^(i)| / sum_i ||g_a^(i)||: "
          f"ps={max(r['aux_sum_rel_ps'] for r in rows):.3g}  "
          f"pop={max(r['aux_sum_rel_pop'] for r in rows):.3g}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
