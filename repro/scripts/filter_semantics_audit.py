"""Matched forward/backward filtering experiment; historical drivers unchanged.

Both variants share a fixed projector and Jacobian. Only the auxiliary forward
value changes. This isolates an implementation distinction, not a new theorem
or a validated hyperparameter-selection rule. The real_aux harness reports
final-epoch test performance; this pilot must not be used to tune on test data.
"""
import argparse
import json
from pathlib import Path
import torch
import real_aux_mtl as base

_historical_probe = base.real_aux_probe


def full_precision_probe(model, xb, *args, **kwargs):
    # The historical function disables autocast only for gradient extraction.
    # Its subsequent eigendecomposition can inherit the caller's bf16 context.
    with torch.autocast(device_type=xb.device.type, enabled=False):
        return _historical_probe(model, xb, *args, **kwargs)


def forward_projection(h, u):
    hc = h - h.mean(0, keepdim=True)
    return h - (hc @ u) @ u.T


def backward_projection(h, u):
    delta = h - h.detach()
    dc = delta - delta.mean(0, keepdim=True)
    return h.detach() + delta - (dc @ u) @ u.T


class Controller(base.CalController):
    def step_h(self, h, mode, warm):
        if mode != 'stfbackward' or warm or self.gate_val <= 0:
            return super().step_h(h, mode, warm)
        self.step += 1
        self.w_n += 1
        r = min(self.r_use, self.U.shape[1])
        u = self.U[:, :r].to(h)
        self.w_sum += 1
        self.r_sum += r
        return backward_projection(h, u), 1., r


def self_test():
    torch.manual_seed(23)
    h = torch.randn(19, 8, dtype=torch.double, requires_grad=True)
    u, _ = torch.linalg.qr(torch.randn(8, 3, dtype=torch.double))
    f, b = forward_projection(h, u), backward_projection(h, u)
    assert torch.equal(b, h)
    assert not torch.allclose(f, h)
    g = torch.randn_like(h)
    expected = g - ((g-g.mean(0)) @ u) @ u.T
    for y in (f,b):
        actual, = torch.autograd.grad((y*g).sum(), h, retain_graph=True)
        torch.testing.assert_close(actual, expected)
    w = torch.randn(8, 4, dtype=torch.double, requires_grad=True)
    target = torch.randn(19, 4, dtype=torch.double)
    losses = [((x@w-target)**2).mean() for x in (h,f,b)]
    gh, gf, gb = [torch.autograd.grad(x,(h,w),retain_graph=True) for x in losses]
    torch.testing.assert_close(gh[1],gb[1])
    assert not torch.allclose(gh[1],gf[1])
    assert not torch.allclose(gf[0],gb[0])
    full = torch.eye(8,dtype=h.dtype)
    torch.testing.assert_close(forward_projection(h,full),h.mean(0).expand_as(h))
    gbfull, = torch.autograd.grad((backward_projection(h,full)*g).sum(),h)
    torch.testing.assert_close(gbfull,g.mean(0).expand_as(g))
    assert gbfull.norm()>0  # Removing centered variation is not full detachment.
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=base.RealAuxModel(6,3,8,'recon',6).to(device).eval()
    x=torch.randn(24,6,device=device);y=torch.arange(24,device=device)%3
    for basis in ('pca','aux'):
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            probe_amp=_historical_probe(model,x,y,x,'recon',6,torch.nn.CrossEntropyLoss(),False,R=8,basis=basis)
        probe_fp32=_historical_probe(model,x,y,x,'recon',6,torch.nn.CrossEntropyLoss(),False,R=8,basis=basis)
        assert probe_amp['U'].dtype==torch.float32
        torch.testing.assert_close(probe_amp['U'],probe_fp32['U'],rtol=0,atol=0)
        assert probe_amp['rho_res']==probe_fp32['rho_res']
    print('PASS: PCA/aux probes are fp32 and identical under enclosing bf16 autocast',flush=True)
    print('PASS: matched Jacobians; backward forward-identity; unchanged head gradient; full-rank mean channel retained',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--alpha',type=float,default=12.)
    ap.add_argument('--seeds',default='0,1,2')
    ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--basis',choices=['pca','aux'],default='aux')
    ap.add_argument('--out')
    args=ap.parse_args()
    torch.set_num_threads(1)
    self_test()
    if args.self_test:
        return
    if not args.out:
        ap.error('--out required')
    base.CalController=Controller
    base.real_aux_probe=full_precision_probe
    base.CAL_MODES=('stfcalrank','stfbackward')
    seeds=[int(s) for s in args.seeds.split(',')]
    rows,agg=base.run('har','recon',args.alpha,64,seeds,args.epochs,1e-3,256,1.,8.,
                      cal_warm=3,modes=('joint','stf0','stfcalrank','stfbackward'),rank_max=64,basis=args.basis)
    for seed in seeds:
        pair=[r for r in rows if r['seed']==seed and r['mode'] in base.CAL_MODES]
        assert len(pair)==2
        for key in ('rho_cal','rank','gate'):
            assert pair[0][key]==pair[1][key],(seed,key)
    result=dict(experimental=True,complete=True,configuration=vars(args),
                comparison='matched fixed-projector Jacobian; changed auxiliary forward value only',
                scope=f'{len(seeds)} initialization seeds, exploratory; no test-based hyperparameter selection',
                results=agg,raw=rows)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print('wrote',out,flush=True)


if __name__=='__main__':
    main()
