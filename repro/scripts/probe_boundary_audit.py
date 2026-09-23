"""probe_boundary_audit.py -- empirical audit of Theorem `thm:probe_boundary`
("Local validity boundary of a frozen probe").

WHAT IT MEASURES
----------------
At a trained checkpoint we evaluate, exactly, every term of the theorem's
decomposition of the signed variance flux

        dV/dt = r - alpha c                                  (eq. signed_variance)
        dV/dt / N = 1 - alpha q + E                          (eq. probe_discrepancy)
        N = k ||u|| n_p / B
        E = (d - 1) + eps_p - alpha (a - q + eps_a)
        Delta = |d-1| + |eps_p| + alpha (|a-q| + |eps_a|)    (eq. validity_margin)

with  u = vec(C_B H)  (unfiltered, batch-centred representation)
      g_p = grad_h L_p,  g_a = grad_h L_a,  n_p = ||g_p||,  e = u/||u||
      a = <e, g_a>/n_p,  d = -<e, g_p>/n_p
      K = J J^T,  k = tr(K)/(B m),  E_K = K/k - I
      eps_a = <e, E_K g_a>/n_p,  eps_p = -<e, E_K g_p>/n_p
      q    = the probe value frozen at calibration time t_0
      r    = -v^T G_p,  c = v^T G_a,  v = grad_theta V,  G = grad_theta L

and the certificate  |1 - alpha q| > Delta  =>  sign(dV/dt) = sign(1 - alpha q).

KEY IMPLEMENTATION POINT
------------------------
K = J J^T is (Bm x Bm) and is NEVER formed.  Every product with K is the
parameter-space inner product of two vector-Jacobian products,
        u^T K g = <J^T u, J^T g> = <B v, G>,
so a, d, eps_a, eps_p, r, c cost O(1) backward passes.  Only the trace
tr(K) = ||J||_F^2 needs one batched VJP per representation coordinate, which is
what `_trace_K` does (chunked, exact, no Hutchinson estimator).

TWO PROBE VALUES ARE AUDITED
----------------------------
  q_frozen : the probe recorded at the end of the detached warm-up (the
             paper's `q = rho_hat_{t_0}`).  Its margin absorbs temporal drift.
  q_live   : the probe recomputed at the audited checkpoint.  Its margin
             isolates the surrogate/aggregation and geometry terms.
Reporting both separates "the statistic drifted" from "the statistic is the
wrong statistic", which is what eq. (mismatch_split) decomposes.

SCALE k
-------
k is a free positive scale; the paper requires it to be reported.  We record
  k_tr  = tr(K)/(Bm)   (canonical zero-trace decomposition, tr(E_K)=0)
  k_opt = argmin_k Delta(k),  Delta_min = Delta(k_opt)
Delta(k) is convex piecewise-linear in 1/N(k) prop. 1/k, so the grid search is
exact to grid resolution and Delta_min is the tightest possible envelope.

SELF-CHECKS (recorded in the output JSON)
-----------------------------------------
  ident_resid  : |dV/dt/N - (1 - alpha q + E)|            ~1e-15
  fd_rel_err   : relative error of a float64 central difference of V along the
                 actual gradient-flow step against dV/dt      ~1e-6
  fired_and_wrong : checkpoints where the certificate fired and the sign still
                 disagreed                                      must be 0
  --selftest   : dense-J ground truth for _trace_K on a toy model

USAGE
-----
  python probe_boundary_audit.py --selftest
  python probe_boundary_audit.py --m-list 32,64,128 --alphas 0.03,100 \
      --seeds 0,1,2 --epochs 40 --warm-epochs 4 --audit-epochs 5,8,12,20,40 \
      --out ../results/PBA/PBA_har.json
"""
import argparse
import importlib.util
import json
import math
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# import the archived driver (its _path() resolves <code>/../../data_x)
# --------------------------------------------------------------------------
def _load_driver():
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.abspath(os.path.join(here, "..", "..", "code",
                                        "cross_domain_mtl.py"))
    spec = importlib.util.spec_from_file_location("cdm_audit", cand)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CDM = _load_driver()
ARCH = "mlp"


# --------------------------------------------------------------------------
# exact trace of K = J J^T  (== ||J||_F^2), chunked batched VJPs
# --------------------------------------------------------------------------
def _trace_K(flat_h, enc, n_out, dev, budget_bytes=1.5e8):
    P = sum(p.numel() for p in enc)
    chunk = int(max(32, min(1024, budget_bytes / max(P * 4, 1))))
    total = 0.0
    for s0 in range(0, n_out, chunk):
        e0 = min(chunk, n_out - s0)
        go = torch.zeros(e0, n_out, device=dev, dtype=flat_h.dtype)
        rows = torch.arange(s0, s0 + e0, device=dev)
        go[torch.arange(e0, device=dev), rows] = 1.0
        gr = torch.autograd.grad(flat_h, enc, grad_outputs=go,
                                 is_grads_batched=True, retain_graph=True)
        for g in gr:
            total += float(g.reshape(e0, -1).pow(2).sum().item())
        del go, gr
    return total


def _rho_hat(hc, gp, ga):
    """the archived probe rho_u: per-example radial pull / per-example ||g_p||."""
    with torch.no_grad():
        hcn = hc / hc.norm(dim=1, keepdim=True).clamp_min(1e-8)
        proj = ((ga * hcn).sum(-1, keepdim=True)) * hcn
        return float((proj.norm(dim=1) / (gp.norm(dim=1) + 1e-8)).mean().item())


def _delta_of_k(d, a, q, alpha, r, c, base):
    """Delta(k) with N(k) = k * base; base = ||u|| n_p / B.  Convex in 1/k."""
    if base <= 0:
        return float("inf"), 0.0
    lo = math.log10(base) + math.log10(1e-7)
    hi = math.log10(base) + math.log10(1e7)
    ks = np.logspace(lo, hi, 241)
    best, bk = float("inf"), float(ks[0])
    for k in ks:
        N = k * base
        ep = r / N - d
        ea = c / N - a
        val = abs(d - 1.0) + abs(ep) + alpha * (abs(a - q) + abs(ea))
        if val < best:
            best, bk = val, float(k)
    return best, bk


def _min_geom(alpha, r, c, d, a, base):
    """min over k>0 of |eps_p(k) - alpha eps_a(k)| / |d| -- the residual
    envelope of the corrected probe q* = a/d, and its minimising scale."""
    if base <= 0 or abs(d) < 1e-300:
        return float("inf"), 0.0
    ks = np.logspace(math.log10(base) + math.log10(1e-7),
                     math.log10(base) + math.log10(1e7), 241)
    best, bk = float("inf"), float(ks[0])
    for k in ks:
        N = k * base
        v = abs((r / N - d) - alpha * (c / N - a))
        if v < best:
            best, bk = v, float(k)
    return best / abs(d), bk


def _sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


# --------------------------------------------------------------------------
# the audit itself
# --------------------------------------------------------------------------
def audit(model, xb, yb, criterion, regress, alpha, q_frozen, fd_rel=1e-6):
    dev = next(model.parameters()).device
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    enc_names = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
    out = {"alpha": float(alpha), "q_frozen": float(q_frozen)}

    with torch.enable_grad(), torch.autocast("cuda", enabled=False):
        xb = xb.float()
        h = model(xb)                                   # (B, m)
        B, m = h.shape
        n_out = B * m
        logits = model.primary(h)
        yt = yb.float() if regress else yb
        if regress:
            logits = logits.squeeze(-1)
        Lp = criterion(logits, yt)
        hc = h - h.mean(0, keepdim=True)
        La = (hc ** 2).mean()                           # == var_loss(h)

        gp = torch.autograd.grad(Lp, h, retain_graph=True)[0]     # (B, m)
        ga = torch.autograd.grad(La, h, retain_graph=True)[0]

        V = 0.5 * (hc ** 2).sum() / B
        v = torch.autograd.grad(V, enc, retain_graph=True)
        Gp = torch.autograd.grad(Lp, enc, retain_graph=True)
        Ga = torch.autograd.grad(La, enc, retain_graph=True)

        u = hc.reshape(-1)
        nu = float(u.norm().item())
        gp_f, ga_f = gp.reshape(-1), ga.reshape(-1)
        n_p = float(gp_f.norm().item())
        r = -float(sum((x.double() * y.double()).sum().item()
                       for x, y in zip(v, Gp)))
        c = float(sum((x.double() * y.double()).sum().item()
                      for x, y in zip(v, Ga)))
        dVdt = r - alpha * c
        q_live = _rho_hat(hc, gp, ga)

        out.update(Lp=float(Lp.item()), La=float(La.item()), V=float(V.item()),
                   nu=nu, n_p=n_p, r=r, c=c, dVdt=dVdt, q_live=q_live)

        degenerate = (nu < 1e-9) or (n_p < 1e-12)
        out["degenerate"] = bool(degenerate)
        if degenerate:
            out["certificate"] = "undefined (u=0 or ||g_p||=0)"
            return out

        a = float(torch.dot(u, ga_f).item()) / (nu * n_p)
        d = -float(torch.dot(u, gp_f).item()) / (nu * n_p)
        trK = _trace_K(h.reshape(-1), enc, n_out, dev)
        k_tr = trK / n_out
        base = nu * n_p / B
        N = k_tr * base
        eps_p = r / N - d
        eps_a = c / N - a

        out.update(a=a, d=d, k_tr=k_tr, trK=trK, N=N,
                   eps_p=eps_p, eps_a=eps_a, rho_b=abs(a),
                   agg_gap=a - q_live, drift=a - q_frozen)

        # ---- the corrected probe implied by the decomposition ------------
        # q* = a/d replaces the two estimator choices the theory names as the
        # dominant error sources: the UNSIGNED per-example numerator becomes the
        # signed radial loading a, and the NORM primary reference ||g_p|| becomes
        # the DIRECTIONAL one -<e, g_p> = d*n_p.  q* is as cheap as the frozen
        # probe (a and d need only two head-level backward passes).  Its
        # prediction d(1 - alpha q*) = d - alpha a is exact in the identity
        # metric; the only residual left is the encoder-geometry term.
        if abs(d) > 1e-12:
            q_star = a / d
            pred_star = d - alpha * a
            Ds, k_star = _min_geom(alpha, r, c, d, a, base)
            out["corrected"] = dict(
                q_star=float(q_star), pred=float(pred_star),
                Delta_star_min=float(Ds), k_star=float(k_star),
                fired=bool(abs(1.0 - alpha * q_star) > Ds),
                agree=bool(_sgn(pred_star) == _sgn(dVdt)))
        else:
            out["corrected"] = None

        # ---- certificate for each probe value ---------------------------
        for tag, q in (("frozen", q_frozen), ("live", q_live)):
            if q is None or not np.isfinite(q):
                continue
            probe_says = 1.0 - alpha * q
            E_exact = (d - 1.0) + eps_p - alpha * (a - q + eps_a)
            Delta = (abs(d - 1.0) + abs(eps_p)
                     + alpha * (abs(a - q) + abs(eps_a)))
            Dmin, kopt = _delta_of_k(d, a, q, alpha, r, c, base)
            true_sign = 1 if dVdt > 0 else (-1 if dVdt < 0 else 0)
            p_sign = 1 if probe_says > 0 else (-1 if probe_says < 0 else 0)
            out[tag] = dict(
                q=float(q), probe_says=float(probe_says), E=float(E_exact),
                Delta=float(Delta), Delta_min=float(Dmin), k_opt=float(kopt),
                fired=bool(abs(probe_says) > Delta),
                fired_at_kopt=bool(abs(probe_says) > Dmin),
                margin=float(abs(probe_says) - Delta),
                margin_kopt=float(abs(probe_says) - Dmin),
                E_over_Delta=float(abs(E_exact) / Delta) if Delta > 0 else None,
                true_sign=true_sign, probe_sign=p_sign,
                agree=bool(true_sign == p_sign),
                ident_resid=float(abs(dVdt / N - (probe_says + E_exact))),
            )

    # ---- float64 central-difference validation of dVdt --------------------
    try:
        dirs = [-(x + alpha * y) for x, y in zip(Gp, Ga)]
        p_dict = {n: p.detach().double() for n, p in zip(enc_names, enc)}
        d_dict = {n: dv.detach().double() for n, dv in zip(enc_names, dirs)}
        dnorm = math.sqrt(sum(float(dv.pow(2).sum().item()) for dv in dirs))
        def V_at(scale):
            with torch.no_grad():
                pp = {n: p_dict[n] + scale * d_dict[n] for n in p_dict}
                h2 = torch.func.functional_call(model.encoder, pp, (xb.double(),))
                hc2 = h2 - h2.mean(0, keepdim=True)
                return float(0.5 * (hc2 ** 2).sum().item() / B)

        eta = fd_rel / max(dnorm, 1e-30)
        fd_raw = (V_at(eta) - V_at(-eta)) / (2 * eta)
        fd_half = (V_at(eta / 2) - V_at(-eta / 2)) / eta
        fd = (4 * fd_half - fd_raw) / 3        # Richardson: removes the O(eta^2) term
        out["fd_dVdt_raw"] = fd_raw
        out["fd_dVdt"] = fd
        out["fd_rel_err_raw"] = (abs(fd_raw - dVdt) / abs(dVdt)
                                 if abs(dVdt) > 1e-12 else None)
        out["fd_rel_err"] = (abs(fd - dVdt) / abs(dVdt)) if abs(dVdt) > 1e-12 else None
        out["fd_eta"] = eta
        out["step_norm"] = dnorm
    except Exception as exc:                                # pragma: no cover
        out["fd_error"] = repr(exc)
    return out


# --------------------------------------------------------------------------
# training (archived joint protocol) + audit at fixed epochs
# --------------------------------------------------------------------------
def run(domain, m, alpha, seed, epochs, lr, batch, warm_epochs, audit_epochs,
        audit_batch=256, verbose=True):
    torch.set_num_threads(1)
    torch.cuda.manual_seed(seed)
    CDM.set_seed(seed)
    CDM.enable_max_perf()
    CDM.ARCH = ARCH
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = CDM.load_domain(domain)
    regress = D["n_classes"] == 1
    model = CDM.MTLModel(D["d_in"], D["n_classes"], m=m, k_aux=3, seed=seed,
                         regress=regress, y_dim=int(D.get("y_dim", 1))).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr = torch.from_numpy(D["X_tr"]).float().to(dev)
    ytr = (torch.from_numpy(D["y_tr"]).float() if regress
           else torch.from_numpy(D["y_tr"]).long()).to(dev)
    criterion = nn.MSELoss() if regress else nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    use_amp = dev == "cuda"
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    # a FIXED audit batch so q and every later term sit on the same examples
    rng = np.random.RandomState(1234 + seed)
    aidx = torch.from_numpy(rng.choice(n, size=min(audit_batch, n),
                                       replace=False)).long()
    xa, ya = Xtr[aidx], ytr[aidx]

    rec = dict(domain=domain, m=m, alpha=float(alpha), seed=seed,
               warm_epochs=warm_epochs, epochs=epochs, lr=lr, batch=batch,
               audit_batch=int(xa.shape[0]), arch=ARCH, points=[])
    q = None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad(set_to_none=True)
            with amp:
                h = model(xb)
                out = model.primary(h)
                if regress:
                    out = out.squeeze(-1)
                L = criterion(out, yb)
                if ep > warm_epochs:
                    mu = h.mean(0, keepdim=True)
                    L = L + alpha * ((h - mu) ** 2).mean()
            L.backward()
            opt.step()

        model.eval()
        if ep == warm_epochs:
            with torch.enable_grad(), torch.autocast("cuda", enabled=False):
                h = model(xa.float())
                hc = h - h.mean(0, keepdim=True)
                logits = model.primary(h)
                if regress:
                    logits = logits.squeeze(-1)
                Lp = criterion(logits, ya.float() if regress else ya)
                La = (hc ** 2).mean()
                gp = torch.autograd.grad(Lp, h, retain_graph=True)[0]
                ga = torch.autograd.grad(La, h, retain_graph=True)[0]
                q = _rho_hat(hc, gp, ga)
            rec["q"] = q
            rec["q_epoch"] = ep
            if verbose:
                print("  [%s m=%d a=%g s=%d] calibrated q=%.6g at ep %d"
                      % (domain, m, alpha, seed, q, ep), flush=True)

        if ep in audit_epochs:
            if q is None:
                if verbose:
                    print("  [%s m=%d a=%g s=%d] ep=%d skipped (no q)"
                          % (domain, m, alpha, seed, ep), flush=True)
                continue
            pt = audit(model, xa, ya, criterion, regress, alpha, q)
            pt["epoch"] = ep
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                h = model(xa.float())
                pt["scatter"] = float((h - h.mean(0)).std(dim=1).mean().item())
            rec["points"].append(pt)
            if verbose:
                fz, lv, co = (pt.get("frozen"), pt.get("live"),
                              pt.get("corrected"))
                print("  [%s m=%d a=%g s=%d] ep=%-3d a=%.3f d=%.3f qf=%.4g "
                      "ql=%.4g | E=%+.3f D=%.3f Dmin=%.3f | "
                      "fired(f/l)=%s/%s agree(f/l)=%s/%s |1-aq|=%.3f "
                      "| q*=%.3f D*=%.2e agree*=%-5s fd=%.2e"
                      % (domain, m, alpha, seed, ep, pt.get("a", float("nan")),
                         pt.get("d", float("nan")), q, pt.get("q_live", float("nan")),
                         fz["E"] if fz else float("nan"),
                         fz["Delta"] if fz else float("nan"),
                         fz["Delta_min"] if fz else float("nan"),
                         fz["fired"] if fz else None, lv["fired"] if lv else None,
                         fz["agree"] if fz else None, lv["agree"] if lv else None,
                         abs(1 - alpha * q),
                         co["q_star"] if co else float("nan"),
                         co["Delta_star_min"] if co else float("nan"),
                         co["agree"] if co else None,
                         pt.get("fd_rel_err") or float("nan")),
                      flush=True)
    return rec


# --------------------------------------------------------------------------
# aggregate summary
# --------------------------------------------------------------------------
def _stats(points, tag):
    ok = [p for p in points if not p.get("degenerate") and tag in p]
    fired = [p for p in ok if p[tag]["fired"]]
    agree = [p for p in ok if p[tag]["agree"]]
    wrong = [p for p in fired if not p[tag]["agree"]]
    fired_opt = [p for p in ok if p[tag]["fired_at_kopt"]]
    return dict(
        n=len(ok),
        fired_rate=(len(fired) / len(ok)) if ok else None,
        fired_at_kopt_rate=(len(fired_opt) / len(ok)) if ok else None,
        agree_rate=(len(agree) / len(ok)) if ok else None,
        fired_and_wrong=len(wrong),
        median_slack=float(np.median([p[tag]["margin"] for p in ok])) if ok else None,
        median_E_over_Delta=(float(np.median([p[tag]["E_over_Delta"] for p in ok
                                              if p[tag]["E_over_Delta"] is not None]))
                             if ok else None),
        max_ident_resid=max([p[tag]["ident_resid"] for p in ok], default=None),
    )


def summarise(recs):
    pts = [p for r in recs for p in r["points"]]
    ok = [p for p in pts if not p.get("degenerate")]
    cor = [p for p in ok if p.get("corrected")]
    fd = [p for p in pts if p.get("fd_rel_err") is not None]
    cor_fired = [p for p in cor if p["corrected"]["fired"]]
    cor_agree = [p for p in cor if p["corrected"]["agree"]]
    return dict(n_points=len(pts), n_defined=len(ok),
                n_degenerate=len(pts) - len(ok),
                frozen=_stats(pts, "frozen"), live=_stats(pts, "live"),
                corrected=dict(
                    n=len(cor),
                    fired_rate=(len(cor_fired) / len(cor)) if cor else None,
                    agree_rate=(len(cor_agree) / len(cor)) if cor else None,
                    fired_and_wrong=len([p for p in cor_fired
                                         if not p["corrected"]["agree"]]),
                    median_Delta_star=(float(np.median(
                        [p["corrected"]["Delta_star_min"] for p in cor]))
                        if cor else None),
                    max_Delta_star=max([p["corrected"]["Delta_star_min"]
                                        for p in cor], default=None)),
                max_ident_resid=max([p["frozen"]["ident_resid"] for p in ok
                                     if "frozen" in p], default=None),
                median_fd_rel_err=(float(np.median([p["fd_rel_err"] for p in fd]))
                                   if fd else None),
                max_fd_rel_err=max([p["fd_rel_err"] for p in fd], default=None))


# --------------------------------------------------------------------------
# dense-J ground truth for _trace_K
# --------------------------------------------------------------------------
def selftest():
    torch.manual_seed(0)
    dev = "cpu"
    model = CDM.MTLModel(7, 3, m=5, k_aux=3, seed=0).double().to(dev)
    xb = torch.randn(4, 7).double()
    enc = [p for p in model.encoder.parameters()]
    h = model(xb)
    B, m = h.shape
    flat = h.reshape(-1)
    n_out = B * m
    dense = []
    for i in range(n_out):
        go = torch.zeros(n_out, dtype=torch.float64)
        go[i] = 1.0
        g = torch.autograd.grad(flat, enc, grad_outputs=go, retain_graph=True)
        dense.append(torch.cat([t.reshape(-1) for t in g]))
    J = torch.stack(dense)                      # (n_out, P)
    trK_dense = float((J ** 2).sum().item())
    trK_batched = _trace_K(flat, enc, n_out, dev)
    K = J @ J.t()
    # cross-check the K-free identity u^T K g == <J^T u, J^T g>
    u = torch.randn(n_out, dtype=torch.float64)
    g = torch.randn(n_out, dtype=torch.float64)
    lhs = float(u @ (K @ g))
    Ju = torch.cat([t.reshape(-1) for t in torch.autograd.grad(
        flat, enc, grad_outputs=u, retain_graph=True)])
    Jg = torch.cat([t.reshape(-1) for t in torch.autograd.grad(
        flat, enc, grad_outputs=g, retain_graph=True)])
    rhs = float(Ju @ Jg)
    print("selftest: n_out=%d P=%d" % (n_out, J.shape[1]))
    print("  trK dense   = %.12f" % trK_dense)
    print("  trK batched = %.12f   rel_err=%.3e"
          % (trK_batched, abs(trK_batched - trK_dense) / trK_dense))
    print("  u^T K g  dense=%.12f  K-free=%.12f  rel_err=%.3e"
          % (lhs, rhs, abs(lhs - rhs) / abs(lhs)))
    ok = (abs(trK_batched - trK_dense) / trK_dense < 1e-9
          and abs(lhs - rhs) / abs(lhs) < 1e-9)
    print("  SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="har")
    ap.add_argument("--m-list", default="64")
    ap.add_argument("--alphas", default="0.03")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warm-epochs", type=int, default=4)
    ap.add_argument("--audit-epochs", default="5,8,12,20,40")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--audit-batch", type=int, default=256)
    ap.add_argument("--arch", default="mlp", choices=["mlp", "transformer"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(selftest())

    global ARCH
    ARCH = a.arch
    ms = [int(x) for x in a.m_list.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    audit_epochs = set(int(x) for x in a.audit_epochs.split(","))

    recs = []
    for m in ms:
        for al in alphas:
            for sd in seeds:
                print("=== run domain=%s m=%d alpha=%g seed=%d ==="
                      % (a.domain, m, al, sd), flush=True)
                recs.append(run(a.domain, m, al, sd, a.epochs, a.lr, a.batch,
                                a.warm_epochs, audit_epochs, a.audit_batch))
    agg = summarise(recs)
    print(json.dumps(agg, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="\n") as f:
            json.dump(dict(domain=a.domain, kind="probe_boundary_audit",
                           config={k: v for k, v in vars(a).items()},
                           summary=agg, runs=recs), f, indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
