"""Experimental backward-only filter with a minimum-rank RMS certificate.

The certificate is empirical on the calibration batch, not a population or
whole-training guarantee. Baseline training, splits and seeds use real_aux_mtl.
This script is separate from the released driver and never changes old arms.
"""
import argparse
import json
from pathlib import Path

import torch
import real_aux_mtl as base


def rms_basis(ga, gp):
    """Top eigenvectors minimize the squared normalized residual at every rank."""
    v = ga.double() / (gp.double().norm(dim=1, keepdim=True) + 1e-8)
    c = v.T @ v / len(v)
    ev, u = torch.linalg.eigh(c)
    ev = ev.flip(0).clamp_min(0)
    u = u.flip(1)
    tail = torch.cat((ev.flip(0).cumsum(0).flip(0), ev.new_zeros(1))).sqrt()
    return u.to(ga.dtype), tail.tolist()


def backward_filter(h, u):
    # Forward is exactly h; only the encoder derivative loses the selected span.
    delta = h - h.detach()
    return h.detach() + delta - (delta @ u) @ u.T


def variance_guard(h):
    """Minimum Frobenius correction making <H_centered, g_aux> <= 0.

    This is an instantaneous feature-space bound, not a parameter-update bound.
    One direction is removed in flattened batch space, not feature space.
    """
    hc = (h.detach().float() - h.detach().float().mean(0))
    out = h.clone()
    def guard(g):
        gf = g.float()
        coeff = (hc * gf).sum().clamp_min(0) / hc.square().sum().clamp_min(1e-20)
        return (gf - coeff * hc).to(g.dtype)
    out.register_hook(guard)
    return out


class Controller(base.CalController):
    def calibrate(self, probe):
        super().calibrate(probe)
        if probe is None:
            raise ValueError('RMS experiment requires a calibration probe')
        self.rho_u = self.rho_res[0]
        self.gate_val = float(self.alpha * self.rho_u >= 1)
        feasible = [r for r, v in enumerate(self.rho_res)
                    if r <= self.rank_max and self.alpha * v < 1]
        # A capped probe must not silently label an unsafe capped rank safe.
        self.full_fallback = not feasible
        self.r_use = min(feasible) if feasible else self.U.shape[0]

    def step_h(self, h, mode, warm):
        self.step += 1
        self.w_n += 1
        if warm:
            return h.detach(), 0., 0
        if mode == 'stfvariance':
            return variance_guard(h), 1., 1
        if not self.gate_val:
            return h, 0., 0
        self.w_sum += 1
        self.r_sum += self.r_use
        if self.full_fallback or self.r_use == h.shape[1]:
            return h.detach(), 1., h.shape[1]
        u = self.U[:, :self.r_use].to(h)
        return backward_filter(h, u), 1., self.r_use


def probe(model, xb, yb, ab, aux_kind, aux_dim, criterion, regress, R=64, basis='aux'):
    with torch.autocast(device_type=xb.device.type, enabled=False), torch.enable_grad():
        h = model(xb).detach().float().requires_grad_(True)
        pred = model.primary(h)
        lp = criterion(pred.squeeze(-1) if regress else pred, yb)
        gp, = torch.autograd.grad(lp, h, retain_graph=True)
        ga, = torch.autograd.grad(base._aux_loss(model, h, ab, aux_kind), h)
    with torch.no_grad():
        if basis == 'aux':
            u, tail = rms_basis(ga, gp)
        else:
            hc = h.detach() - h.detach().mean(0)
            _, u = torch.linalg.eigh(hc.T @ hc)
            u = u.flip(1)
            v = ga / (gp.norm(dim=1, keepdim=True) + 1e-8)
            tail = [float((v - (v @ u[:, :r]) @ u[:, :r].T).square().sum(1).mean().sqrt())
                    for r in range(h.shape[1] + 1)]
            tail[-1] = 0.
    return dict(rho_u=tail[0], rho_res=tail[:R+1], U=u[:, :R].detach())


def self_test():
    torch.manual_seed(7)
    h = torch.randn(20, 8, dtype=torch.double, requires_grad=True)
    u, _ = torch.linalg.qr(torch.randn(8, 3, dtype=torch.double))
    out = backward_filter(h, u)
    assert torch.equal(out, h)
    g = torch.randn_like(h)
    actual, = torch.autograd.grad((out * g).sum(), h)
    torch.testing.assert_close(actual, g - (g @ u) @ u.T)
    ga, gp = torch.randn(100, 8), torch.randn(100, 8)
    u, tail = rms_basis(ga, gp)
    v = ga / (gp.norm(dim=1, keepdim=True) + 1e-8)
    for r in range(9):
        residual = v - (v @ u[:, :r]) @ u[:, :r].T
        rms = float(residual.square().sum(1).mean().sqrt())
        assert abs(rms - tail[r]) < 1e-5
        assert float(residual.norm(dim=1).mean()) <= tail[r] + 1e-5
        for _ in range(10):
            q, _ = torch.linalg.qr(torch.randn(8, r))
            other = float((v - (v @ q) @ q.T).square().sum(1).mean().sqrt())
            assert rms <= other + 1e-5
    cal = Controller(100., 0, rank_max=2)
    cal.calibrate(dict(rho_u=2., rho_res=[2., 1., .5], U=torch.eye(8)[:, :2]))
    assert cal.full_fallback and cal.r_use == 8
    cal = Controller(.01, 0, rank_max=8)
    cal.calibrate(dict(rho_u=tail[0], rho_res=tail, U=u))
    assert cal.r_use == 0 and not cal.gate_val
    hh = torch.randn(20, 8, requires_grad=True)
    hc = hh.detach() - hh.detach().mean(0)
    gg = hc + torch.randn_like(hh) * .1
    output = variance_guard(hh)
    torch.testing.assert_close(output, hh)
    guarded, = torch.autograd.grad((output * gg).sum(), hh)
    assert float((hc * guarded).sum()) <= 1e-4
    print('PASS: forward identity, backward projection, RMS tails, Jensen bound, rank optimality, capped fallback, rank zero', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--domain', default='har')
    ap.add_argument('--aux', default='recon')
    ap.add_argument('--alpha', type=float, default=12.)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--basis', choices=['pca', 'aux'], default='aux')
    ap.add_argument('--variant', choices=['rms', 'variance'], default='rms')
    ap.add_argument('--out')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.self_test:
        self_test()
        return
    if not args.out:
        ap.error('--out required')
    base.CalController = Controller
    base.real_aux_probe = probe
    mode = 'stfrms' if args.variant == 'rms' else 'stfvariance'
    base.CAL_MODES = (mode,)
    seeds = [int(s) for s in args.seeds.split(',')]
    rows, agg = base.run(args.domain, args.aux, args.alpha, 64, seeds,
                         args.epochs, 1e-3, 256, 1., 8., cal_warm=3,
                         modes=('joint', 'stf0', mode), rank_max=64, basis=args.basis)
    if args.variant == 'variance':
        for row in rows:
            if row['mode'] == mode:
                row['rank'] = None
                row['gate'] = None
                row['margin'] = None
        for row in agg:
            if row['mode'] == mode:
                row['rank_mean'] = None
                row['gate_rate'] = None
                row['margin_mean'] = None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(experimental=True, variant=args.variant,
        certificate='calibration-batch RMS only' if args.variant == 'rms' else 'instantaneous feature-space scatter derivative only',
        forward='identity', domain=args.domain, aux=args.aux, alpha=args.alpha,
        m=64, epochs=args.epochs, seeds=seeds, basis=args.basis, results=agg, raw=rows), indent=2))
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
