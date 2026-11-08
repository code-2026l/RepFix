"""Wall-clock cost of the exact signed flux versus the Theorem-1 margin.

At one HAR checkpoint (m=64, B=256) we time

  (a) dV/dt = r - alpha c, obtained from three backward passes (V, L_p, L_a);
  (b) the certificate terms, whose only expensive part is tr(K)=||J||_F^2,
      computed exactly by the same chunked batched-VJP routine used in
      repro/scripts/probe_boundary_audit.py.

The result is intentionally unflattering: the certificate is an offline audit,
not a cheap substitute for measuring the flux.  The cheap object in the deployed
rule is the probe q, not the certificate.

USAGE
-----
  export REPFIX_DATA_X=/path/to/data_x
  python repro/scripts/certificate_cost.py --out results/CERTCOST/cost.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import cross_domain_mtl as cdm  # noqa: E402


def trace_K(flat_h, enc, n_out, dtype, budget_bytes=1.5e8):
    P = sum(p.numel() for p in enc)
    chunk = int(max(32, min(1024, budget_bytes / max(P * 4, 1))))
    total = 0.0
    for s0 in range(0, n_out, chunk):
        e0 = min(chunk, n_out - s0)
        go = torch.zeros(e0, n_out, dtype=dtype)
        rows = torch.arange(s0, s0 + e0)
        go[torch.arange(e0), rows] = 1.0
        gr = torch.autograd.grad(flat_h, enc, grad_outputs=go,
                                 is_grads_batched=True, retain_graph=True)
        for g in gr:
            total += float(g.reshape(e0, -1).pow(2).sum().item())
        del go, gr
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="har")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="results/CERTCOST/cost.json")
    a = ap.parse_args()

    torch.set_num_threads(1)
    cdm.set_seed(0)
    D = cdm.load_domain(a.domain)
    model = cdm.MTLModel(D["d_in"], D["n_classes"], m=a.m, k_aux=3, seed=0).double()
    xb = torch.from_numpy(D["X_tr"][:a.batch]).double()
    yb = torch.from_numpy(D["y_tr"][:a.batch]).long()
    enc = list(model.encoder.parameters())
    params = list(model.parameters())
    crit = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    h = model(xb)
    hc = h - h.mean(0, keepdim=True)
    V = hc.pow(2).sum() / (2 * len(xb))
    Lp = crit(model.primary(h), yb)
    La = hc.pow(2).mean()
    gp = torch.autograd.grad(Lp, params, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(La, params, retain_graph=True, allow_unused=True)
    v = torch.autograd.grad(V, params, retain_graph=False, allow_unused=True)
    dot = lambda x, y: sum(float((p * q).sum()) for p, q in zip(x, y)
                           if p is not None and q is not None)
    r, c = -dot(v, gp), dot(v, ga)
    flux_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    h2 = model(xb)
    hc2 = h2 - h2.mean(0, keepdim=True)
    Lp2 = crit(model.primary(h2), yb)
    La2 = hc2.pow(2).mean()
    torch.autograd.grad(Lp2, h2, retain_graph=True)
    torch.autograd.grad(La2, h2, retain_graph=True)
    trK = trace_K(h2.reshape(-1), enc, h2.numel(), h2.dtype)
    cert_seconds = time.perf_counter() - t0

    payload = dict(domain=a.domain, m=a.m, batch=a.batch,
                   flux_r=r, flux_c=c, flux_seconds=flux_seconds,
                   trace_K=trK, certificate_seconds=cert_seconds,
                   ratio=cert_seconds / max(flux_seconds, 1e-9))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
