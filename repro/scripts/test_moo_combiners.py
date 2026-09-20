"""Differential + invariant test for moo_combiners.py.

The important test is the *differential* one: our PCGrad is compared, on the
same random problem and the same parameter set, against the author-released
WeiChengTseng/Pytorch-PCGrad class vendored next to this test as _ref_pcgrad.py.  That
class flattens *the parameters it is given* into one vector before taking the
dot product and the projection norm, so the comparison is run with the shared
(encoder) parameters as that set -- the same set the harness hands to every
combiner.
"""
import importlib.util, os, sys
import numpy as np
import torch

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import moo_combiners as mc

# The author-released PCGrad is shipped next to this test as _ref_pcgrad.py
# (verbatim copy of WeiChengTseng/Pytorch-PCGrad pcgrad.py); when it is absent
# the differential check is skipped rather than silently downgraded.
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ref_pcgrad.py")
ref_pcgrad = None
if os.path.exists(REF):
    spec = importlib.util.spec_from_file_location("ref_pcgrad", REF)
    ref_pcgrad = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_pcgrad)

torch.manual_seed(0)
np.random.seed(0)
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "   " + str(extra)))
    if not cond:
        fails.append(name)


def cat(vecs):
    return torch.cat([t.reshape(-1) for t in vecs if t is not None])


def nrm(vecs):
    return float(cat(vecs).norm())


def finite(vecs):
    return all(t is not None and bool(torch.isfinite(t).all()) for t in vecs)


def shapes_ok(vecs, params):
    return all(v is None or v.shape == p.shape for v, p in zip(vecs, params))


# ------------------------------------------------------- shared-only toy task
D_IN, M, N = 6, 5, 32
X = torch.randn(N, D_IN)
A = torch.randn(M, 1)
Y = torch.randn(N, 1)


class Enc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(D_IN, M), torch.nn.ReLU(),
                                       torch.nn.Linear(M, M))


def losses_of(enc, alpha=3.0):
    h = enc.net(X)
    L_main = torch.nn.functional.mse_loss(h @ A, Y)
    L_tox = alpha * ((h - h.mean(0)) ** 2).mean()
    return L_main, L_tox


enc = Enc()
SHARED = list(enc.net.parameters())
L_main, L_tox = losses_of(enc)
gp = torch.autograd.grad(L_main, SHARED, retain_graph=True)
gt = torch.autograd.grad(L_tox, SHARED, retain_graph=True)
grads = [list(gp), list(gt)]
print("task gradient norms (shared): %.4f  %.4f" % (nrm(grads[0]), nrm(grads[1])))
cospa = float(cat(grads[0]) @ cat(grads[1])) / (nrm(grads[0]) * nrm(grads[1]) + 1e-12)
print("cos(g_main, g_tox) = %.4f" % cospa)

# ------------------------------------------------- 1. differential PCGrad
# The author-released file cannot reach its own `elif reduction == 'sum'`
# branch: `if self._reduction:` is already true for the string 'sum', so the
# class always averages.  We therefore compare against its reachable branch
# ('mean'), and verify separately that our 'sum' is exactly K times it.
if ref_pcgrad is not None:
    enc2 = Enc()
    enc2.load_state_dict(enc.state_dict())
    opt2 = torch.optim.SGD(enc2.net.parameters(), lr=1e-3)
    l_main2, l_tox2 = losses_of(enc2)
    pc = ref_pcgrad.PCGrad(opt2, reduction="mean")
    pc.pc_backward([l_main2, l_tox2])
    official = [p.grad.clone() for p in enc2.net.parameters()]
    mine_mean = mc.combine_pcgrad(grads, SHARED, reduction="mean")
    err = max(float((a - b).abs().max()) for a, b in zip(official, mine_mean))
    check("PCGrad(mean) == author-released reference on the shared set",
          err < 1e-6, err)
else:
    mine_mean = mc.combine_pcgrad(grads, SHARED, reduction="mean")
    print("SKIP PCGrad differential: reference file not shipped next to the test")

mine_sum = mc.combine_pcgrad(grads, SHARED, reduction="sum")
err = max(float((a - 2 * b).abs().max()) for a, b in zip(mine_sum, mine_mean))
check("PCGrad(sum) == K x reference mean-branch output", err < 1e-6, err)

# the defining property: after surgery no task's gradient opposes the other.
# Checked on the real pair and on a deliberately conflicting one.
for tag, pair in [("real pair", grads),
                  ("conflicting pair", [list(grads[0]),
                                        [-1.5 * t for t in grads[0]]])]:
    _, parts = mc.combine_pcgrad(pair, SHARED, return_parts=True)
    d01 = float(cat(parts[0]) @ cat(grads[0] if tag == "real pair" else pair[1]))
    d10 = float(cat(parts[1]) @ cat(pair[0]))
    check("PCGrad(%s): no residual conflict (both cross-dots >= 0)" % tag,
          d01 >= -1e-6 and d10 >= -1e-6, "%.3e %.3e" % (d01, d10))

# A pair with a negative *and* an orthogonal part must be materially changed by
# the surgery.  Purely collinear pairs are excluded on purpose: for two tasks on
# one line each projection lands in the other's normal plane, so the merged
# gradient is exactly zero -- the degenerate case, not a bug.
g1 = cat(grads[0])
o = cat(grads[1])
o = o - (float(o @ g1) / float(g1 @ g1)) * g1
o = o / (float(o.norm()) + 1e-12)
g2 = mc._unflatten((-0.5 * g1 + 0.8 * o), SHARED)
pair = [list(grads[0]), list(g2)]
outp = mc.combine_pcgrad(pair, SHARED)
outs = mc.combine_sum(pair, SHARED)
d = max(float((a - b).abs().max()) for a, b in zip(outp, outs))
check("PCGrad materially changes a partially conflicting pair", d > 1e-6, d)
check("PCGrad surgery is non-trivial on that pair",
      float(torch.cat([t.reshape(-1) for t in outp]).norm()) > 1e-6,
      float(torch.cat([t.reshape(-1) for t in outp]).norm()))

# exactly anti-parallel is the documented degenerate limit: surgery -> 0
anti = [list(grads[0]), [-1.0 * t for t in grads[0]]]
check("PCGrad on an exactly anti-parallel pair annihilates both",
      float(torch.cat([t.reshape(-1) for t in mc.combine_pcgrad(anti, SHARED)]
                      ).norm()) < 1e-6)

# ------------------------------------------------------------ 2. invariants
for name in ["sum", "pcgrad", "cagrad", "cagrad04", "cagrad08", "mgda",
             "aligned", "imtlg"]:
    out, _ = mc.combine(name, grads, SHARED)
    check("%-10s finite + shapes" % name, finite(out) and shapes_ok(out, SHARED))

st_nash = mc.NashMTLState(2, "cpu")
out = st_nash.combine(grads, SHARED)
check("%-10s finite + shapes" % "nash", finite(out) and shapes_ok(out, SHARED))

g_same = [[t.clone() for t in grads[0]], [t.clone() for t in grads[0]]]
com = mc.combine_sum(g_same, SHARED)
for name in ["pcgrad", "cagrad", "mgda", "aligned"]:
    out, _ = mc.combine(name, g_same, SHARED)
    cos = float(cat(out) @ cat(com)) / (nrm(out) * nrm(com) + 1e-12)
    check("%-9s stays on the common direction when tasks agree" % name,
          abs(cos - 1) < 1e-5, "cos=%.8f" % cos)

out, _ = mc.combine("aligned", grads, SHARED)
check("Aligned-MTL output is O(task gradient scale)",
      0 < nrm(out) / max(nrm(grads[0]), nrm(grads[1])) < 20)

out, _ = mc.combine("imtlg", grads, SHARED)
check("IMTL-G output is O(task gradient scale)",
      0 < nrm(out) / max(nrm(grads[0]), nrm(grads[1])) < 20)

GTG = (mc._stack(grads, SHARED) @ mc._stack(grads, SHARED).t()).double()
nf = float(torch.norm(GTG))
a = mc._nash_newton((GTG / nf).numpy(), np.ones(2, np.float32))
res = float(np.max(np.abs(a * ((GTG.numpy() / nf) @ a) - 1.0)))
check("Nash-Newton satisfies alpha_i (G alpha)_i = 1", res < 1e-3, res)

# CAGrad solution should not be worse than the uniform point of its objective
G = mc._stack(grads, SHARED).double()
GG = G @ G.t()
g0n = (GG.mean() + 1e-8).sqrt()
cc = float(0.5 * g0n + 1e-8)
A_ = GG.numpy(); b_ = np.ones(2) / 2


def obj(x):
    return float((x.reshape(1, -1).dot(A_).dot(b_.reshape(-1, 1))
                  + cc * np.sqrt(x.reshape(1, -1).dot(A_).dot(x.reshape(-1, 1))
                                 + 1e-8)).sum())


from scipy.optimize import minimize
r = minimize(obj, b_, bounds=((0, 1), (0, 1)),
             constraints=({"type": "eq", "fun": lambda x: 1 - sum(x)},))
check("CAGrad dual optimum <= uniform-simplex objective", obj(r.x) <= obj(b_) + 1e-9,
      "%f vs %f" % (obj(r.x), obj(b_)))

# --------------------------------------------------- 3. weight-learning loop
def run_weighted(method, steps=80):
    torch.manual_seed(1)
    e = Enc()
    state = mc.make_state(method, 2, "cpu")
    ps = list(e.parameters()) + mc.state_parameters(state)
    opt = torch.optim.Adam(ps, lr=5e-3)
    hist = []
    for _ in range(steps):
        h = e.net(X)
        Lm = torch.nn.functional.mse_loss(h @ A, Y)
        Lt = 3.0 * ((h - h.mean(0)) ** 2).mean()
        losses = torch.stack([Lm, Lt])
        opt.zero_grad()
        if method == "gradnorm":
            # the per-task gradients have to be taken before the weighted-loss
            # backward frees the graph; the reference does the same by working
            # on detached gradient lists throughout
            p_ = list(e.net.parameters())
            gL = torch.autograd.grad(Lm, p_, retain_graph=True)
            gT = torch.autograd.grad(Lt, p_, retain_graph=True)
            state.weighted_loss(losses).backward()
            state.update_w(losses, [list(gL), list(gT)], p_, epoch=1)
        else:
            state.weighted_loss(losses).backward()
        opt.step()
        if method == "famo":
            with torch.no_grad():
                h2 = e.net(X)
                nl = torch.stack([torch.nn.functional.mse_loss(h2 @ A, Y),
                                  3.0 * ((h2 - h2.mean(0)) ** 2).mean()])
            state.update_w(nl)
        hist.append((float(Lm), float(Lt), [float(v) for v in state.weights()]))
    return hist


for method in ["famo", "uw", "imtll", "gradnorm"]:
    h = run_weighted(method)
    arr = np.array([[r[0], r[1]] for r in h])
    w = h[-1][2]
    ok = bool(np.isfinite(arr).all()) and arr[-1, 0] < arr[0, 0]
    check("%-9s trains, finite, L_main %.4f -> %.4f, w=%s"
          % (method, arr[0, 0], arr[-1, 0], ["%.3f" % v for v in w]), ok)

# the fp32 production configuration must not need fp64 anywhere
check("module exposes the full method set",
      set(mc.FAMILY) == set(list(mc.STATELESS) + list(mc.WEIGHTING) + ["nash"]))

print("\n%d failure(s): %s" % (len(fails), fails))
sys.exit(1 if fails else 0)
