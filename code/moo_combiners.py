"""Faithful ports of the multi-objective / conflict-resolution gradient
combiners benchmarked in the Section-7.2 ablation.

Every routine is a line-for-line port of an author-released or maintained
reference implementation, named in its docstring, so that a reader (or a
reviewer diffing the released code) can verify the correspondence directly.
Nothing here is a "proxy" or a "structured re-implementation" of the idea.

  PCGrad      WeiChengTseng/Pytorch-PCGrad   pcgrad.py::_project_conflicting
              cross-checked against AvivNavon/nash-mtl  weight_methods.py::PCGrad
  CAGrad      Cranial-XIX/CAGrad             cagrad.py::cagrad
              identical to LibMTL/grad/toolkit CAGrad and nash-mtl's CAGrad
  FAMO        Cranial-XIX/FAMO               famo.py::FAMO
              identical to LibMTL FAMO
  MGDA        isl-org/MultiObjectiveOptimization, via LibMTL::MGDA
  Aligned-MTL SamsungLabs/MTL, via LibMTL::Aligned_MTL
  IMTL-G / L  runtao-liu/IMTL, via LibMTL::IMTL
  Nash-MTL    AvivNavon/nash-mtl  methods/weight_methods.py::NashMTL
              identical to LibMTL::Nash_MTL
  UW          Kendall et al. 2018, via LibMTL::UW
  GradNorm    Chen et al. 2018, via LibMTL::GradNorm

Interface
---------
A combiner consumes ``grads``: a list of ``K`` per-task gradients, each itself
a list of per-parameter tensors (``None`` where the task does not touch the
parameter), in a fixed parameter order; and ``params``: the parameter list the
gradients were taken with respect to -- by construction the *shared* (encoder)
parameters, which is the set every reference implementation operates on.
It returns the combined gradient in the same list-of-tensors structure.

Stateful methods keep their learnable/optimiser state in a small ``*State``
object that lives across training steps, exactly as the reference keeps it.
"""

from __future__ import annotations

import contextlib
import functools

import numpy as np
import torch
from scipy.optimize import minimize

try:                                              # Nash-MTL's reference solver
    import cvxpy as cp
    _HAVE_CVXPY = True
except Exception:                                 # pragma: no cover
    cp = None
    _HAVE_CVXPY = False


# --------------------------------------------------------------- flattening
def n_params(params):
    return sum(int(p.numel()) for p in params)


def _flatten(grad, params):
    """One task's gradient -> a single vector, zeros where it is undefined."""
    parts = []
    for p, g in zip(params, grad):
        parts.append(torch.zeros(p.numel(), dtype=p.dtype, device=p.device)
                     if g is None else g.reshape(-1))
    return torch.cat(parts)


def _unflatten(vec, params):
    """Vector -> list of tensors shaped like ``params`` (dtype/shape restored)."""
    out, off = [], 0
    for p in params:
        n = int(p.numel())
        out.append(vec[off:off + n].reshape(p.shape).to(device=p.device,
                                                        dtype=p.dtype))
        off += n
    return out


def _stack(grads, params):
    return torch.stack([_flatten(g, params) for g in grads])


def _f32(G):
    """Promote a gradient stack to fp32 for the weight-selection arithmetic.

    The references were released against fp32 models, and every one of them
    decides its weights from a Gram matrix: in bf16 (which the battery flagship
    driver uses under autocast) that spectrum is too coarse to tell a conflict
    from a near-parallel pair, and ``linalg.eigh`` is not even implemented for
    bf16.  The merged gradient is cast back to the parameter dtype by
    ``_unflatten``, so this only affects how the weights are chosen.
    """
    return G.float() if G.dtype in (torch.bfloat16, torch.float16) else G


_flat_stack = _stack


def _merge(G, alpha, params):
    """Combined gradient for weights ``alpha`` over the rows of ``G``."""
    return _unflatten((G * alpha.reshape(-1, 1)).sum(0), params)


@contextlib.contextmanager
def _no_autocast():
    """Weight selection is computed outside the autocast region.

    Every family here decides its weights from a Gram matrix, and autocast
    re-casts the matmuls that build it back down to bf16: that spectrum is too
    coarse to tell a conflict from a near-parallel pair, and ``linalg.eigh`` is
    not even implemented for bf16.  The combined gradient is cast back to the
    parameter dtype by ``_unflatten``, so this changes only how the weights are
    chosen, never the training dtype.
    """
    if torch.cuda.is_available() and torch.is_autocast_enabled():
        with torch.autocast("cuda", enabled=False):
            yield
    else:
        yield


def _fp32_weights(fn):
    """Decorator: run a combiner outside autocast (see :func:`_no_autocast`)."""
    @functools.wraps(fn)
    def g(*a, **kw):
        with _no_autocast():
            return fn(*a, **kw)
    return g


# ------------------------------------------------------- stateless combiners
@_fp32_weights
def combine_sum(grads, params, **kw):
    """Plain summation -- the ``joint`` baseline the ablation starts from."""
    out = []
    for i in range(len(params)):
        acc = None
        for g in grads:
            if g[i] is None:
                continue
            acc = g[i].clone() if acc is None else acc + g[i]
        out.append(acc)
    return out


@_fp32_weights
def combine_pcgrad(grads, params, reduction="sum", return_parts=False, **kw):
    """Yu et al. 2020, *Gradient Surgery for Multi-Task Learning*.

    Reference: ``WeiChengTseng/Pytorch-PCGrad`` ``pcgrad.py``,
    ``_project_conflicting``: each task's gradient is projected off any task it
    conflicts with, using the *unmodified* peers and the **global** parameter
    norm of the peer (the reference flattens the whole network before taking
    the dot product).  For ``K=2``, which is our setting (one primary, one
    auxiliary), the reference's per-step task shuffling does not change the
    outcome, so this is deterministic.

    ``reduction='sum'`` matches the summed baseline the ablation compares
    against.  The author-released file cannot actually reach its own
    ``elif self._reduction == 'sum'`` branch (the preceding ``if self._reduction:``
    already swallows ``'sum'``, so it always averages), which is why the port
    follows ``nash-mtl``'s ``PCGrad``, where ``reduction='sum'`` is reachable and
    means exactly what it says.  The two differ only by the constant factor
    ``K``, i.e. by the effective step size, and the ablation fixes the step size
    across methods by construction.
    """
    K = len(grads)
    flat = _f32(_stack(grads, params))
    pc = [f.clone() for f in flat]
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            dot = torch.dot(pc[i], flat[j])
            if dot < 0:
                nj = flat[j].norm() ** 2
                pc[i] = pc[i] - dot / nj * flat[j]
    parts = [_unflatten(v, params) for v in pc]
    merged = torch.stack(pc)
    merged = merged.mean(0) if reduction == "mean" else merged.sum(0)
    if return_parts:
        return _unflatten(merged, params), parts
    return _unflatten(merged, params)


@_fp32_weights
def combine_cagrad(grads, params, c=0.5, rescale=1, **kw):
    """Liu et al. 2021, *Conflict-Averse Gradient Descent*.

    Reference: ``Cranial-XIX/CAGrad`` ``cagrad.py::cagrad`` (also ``LibMTL``
    ``CAGrad`` and ``nash-mtl``'s ``CAGrad``): solve over the simplex
    ``min_w  w' A b + c||g0|| sqrt(w' A w)`` with ``A = G G'`` and ``b`` uniform,
    then ``g = mean_i g_i + (c||g0||/||g_w||) g_w``, rescaled by ``1/(1+c^2)``.

    ``c=0.5`` and ``rescale=1`` are the reference defaults (``cagrad.py`` uses
    ``alpha=0.5``; LibMTL's config ``calpha=0.5``).  ``c`` is swept separately
    to foreclose a "you tuned CAGrad badly" objection.
    """
    G = _f32(_flat_stack(grads, params))
    K = G.shape[0]
    GG = (G @ G.t()).detach().cpu().double()
    g0_norm = (GG.mean() + 1e-8).sqrt()
    x0 = np.ones(K) / K
    A = GG.numpy()
    b = x0.copy()
    cc = float(c * g0_norm + 1e-8)

    def objfn(x):
        return (x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1))
                + cc * np.sqrt(x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1))
                               + 1e-8)).sum()

    res = minimize(objfn, x0, bounds=tuple((0, 1) for _ in x0),
                   constraints=({"type": "eq", "fun": lambda x: 1 - sum(x)},))
    ww = torch.as_tensor(res.x, dtype=G.dtype, device=G.device)
    gw = (G * ww.reshape(-1, 1)).sum(0)          # g_w = sum_i w_i g_i   (gradient space)
    lmbda = cc / (gw.norm() + 1e-8)
    g = G.mean(0) + lmbda * gw                   # g_0 + lambda g_w
    if rescale == 1:
        g = g / (1 + c ** 2)
    elif rescale == 2:
        g = g / (1 + c)
    return _unflatten(g, params)


@_fp32_weights
def combine_mgda(grads, params, gn="none", **kw):
    """Sener & Koltun 2018, *Multi-Task Learning as Multi-Objective
    Optimization*; reference ``isl-org/MultiObjectiveOptimization`` via
    ``LibMTL`` ``MGDA._find_min_norm_element``.

    ``LibMTL`` returns the closed-form 2-task min-norm pair when ``n < 3``, which
    is our case, so that branch is ported exactly (including the ``0.999`` /
    ``0.001`` clamps).  ``gn='none'`` is the reference default and leaves the
    gradients unnormalised.
    """
    G = _flat_stack(grads, params)
    K = G.shape[0]
    assert K == 2, "MGDA branch ported for the 2-task ablation setting"
    Gd = _f32(G)
    GM = Gd @ Gd.t()
    v1v1, v1v2, v2v2 = GM[0, 0], GM[0, 1], GM[1, 1]
    if v1v2 >= v1v1:
        gamma = 0.999
    elif v1v2 >= v2v2:
        gamma = 0.001
    else:
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
    alpha = torch.zeros(K, dtype=Gd.dtype, device=Gd.device)
    alpha[0], alpha[1] = gamma, 1 - gamma
    return _merge(G, alpha, params)


@_fp32_weights
def combine_aligned(grads, params, **kw):
    """Senushkin et al. 2023, *Independent Component Alignment for MTL*.

    Reference: ``SamsungLabs/MTL`` via ``LibMTL`` ``Aligned_MTL``: the ``K x K``
    Gram matrix ``M = G G'`` is eigendecomposed, eigenvalues below the rank
    tolerance are dropped, and the surviving spectrum is replaced by a constant
    (``sqrt(lambda_min) / sqrt(lambda)``), which makes the system's condition
    number one.  Task weights are the column sums of the resulting operator.
    """
    G = _flat_stack(grads, params)
    M = _f32(G)
    M = M @ M.t()
    lmbda, V = torch.linalg.eigh(M)
    tol = lmbda.max() * max(M.shape[-2:]) * torch.finfo(lmbda.dtype).eps
    rank = int((lmbda > tol).sum())
    order = torch.argsort(lmbda, descending=True)
    lmbda, V = lmbda[order][:rank], V[:, order][:, :rank]
    sigma = torch.diag(1.0 / lmbda.sqrt())
    B = lmbda[-1].sqrt() * ((V @ sigma) @ V.t())
    alpha = B.sum(0)
    return _merge(G, alpha, params)


@_fp32_weights
def combine_imtl_g(grads, params, **kw):
    """Liu et al. 2021, *Towards Impartial Multi-Task Learning* -- the
    gradient-balance block (IMTL-G).

    Reference: ``runtao-liu/IMTL`` via ``LibMTL`` ``IMTL``: with ``u_i`` the unit
    gradients, ``D = g_1 - g_{2..K}``, ``U = u_1 - u_{2..K}``, the closed form is
    ``alpha_{2..K} = (g_1' U')(D U')^{-1}`` and ``alpha_1 = 1 - sum``.  The
    Gram inverse is singular-degenerate in fp32 for near-parallel tasks, so the
    identity-regularised fallback of ``nash-mtl``'s ``IMTLG`` is used here, as in
    that reference.

    Note ``alpha`` is not sign-constrained; the aggregated gradient is therefore
    *not* a positive per-task reweighting, so Proposition 3 does not cover it.
    """
    G = _flat_stack(grads, params)
    K = G.shape[0]
    Gd = _f32(G)
    Guo = Gd / Gd.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    D = Gd[0:1].repeat(K - 1, 1) - Gd[1:]
    U = Guo[0:1].repeat(K - 1, 1) - Guo[1:]
    try:
        inv = torch.inverse(D @ U.t())
    except Exception:
        inv = torch.inverse(torch.eye(K - 1, dtype=Gd.dtype, device=Gd.device) * 1e-8
                            + D @ U.t())
    a = (Gd[0] @ U.t()) @ inv
    alpha = torch.cat((1 - a.sum().unsqueeze(0), a), dim=0)
    return _merge(G, alpha, params)


STATELESS = {
    "sum": combine_sum,
    "pcgrad": combine_pcgrad,
    "cagrad": combine_cagrad,
    "cagrad04": lambda g, p: combine_cagrad(g, p, c=0.4),
    "cagrad08": lambda g, p: combine_cagrad(g, p, c=0.8),
    "mgda": combine_mgda,
    "aligned": combine_aligned,
    "imtlg": combine_imtl_g,
}


# -------------------------------------------------------- stateful combiners
class FAMOState:
    """Liu et al. 2023, *FAMO: Fast Adaptive Multitask Optimization*.

    Reference: ``Cranial-XIX/FAMO`` ``famo.py`` == ``LibMTL`` ``FAMO``.  Weights
    are ``softmax(w)`` with ``w`` optimised by Adam (``lr=0.025``,
    ``weight_decay=1e-3``) after each model step, via the first-order progress
    ``delta`` between the pre- and post-step losses.  ``min_losses`` is fixed at
    zero exactly as in both references.
    """

    def __init__(self, K, device, w_lr=0.025, w_gamma=1e-3):
        self.min_losses = torch.zeros(K, device=device)
        self.w = torch.zeros(K, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=w_gamma)
        self.prev_losses = None

    def weighted_loss(self, losses):
        self.prev_losses = losses
        z = torch.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        return (D.log() * z / c).sum()

    @torch.enable_grad()
    def update_w(self, curr_losses):
        delta = ((self.prev_losses - self.min_losses + 1e-8).log()
                 - (curr_losses - self.min_losses + 1e-8).log())
        w = self.w.detach().requires_grad_(True)
        d = torch.autograd.grad(torch.softmax(w, -1), w,
                                grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad(set_to_none=False)
        self.w.grad = d
        self.w_opt.step()

    def weights(self):
        return torch.softmax(self.w, -1).detach()

    def parameters(self):
        return []                                # FAMO owns its own optimiser


class UWState:
    """Kendall et al. 2018, *Multi-Task Learning Using Uncertainty to Weigh
    Losses*; reference ``LibMTL`` ``UW``: ``sum_i (L_i / (2 e^{s_i}) + s_i / 2)``
    with ``s`` a learnable parameter initialised to ``-0.5`` and handed to the
    main optimiser.  This is a strictly positive per-task reweighting, i.e.
    inside Proposition 3's sign-preserving class.
    """

    def __init__(self, K, device):
        self.s = torch.nn.Parameter(torch.full((K,), -0.5, device=device))

    def weighted_loss(self, losses):
        return (losses / (2 * self.s.exp()) + self.s / 2).sum()

    def weights(self):
        return 1.0 / (2 * torch.exp(self.s.detach()))

    def parameters(self):
        return [self.s]


class IMTLLossState:
    """Liu et al. 2021 -- the loss-balance block (IMTL-L); reference ``LibMTL``
    ``IMTL``: ``sum_i (e^{s_i} L_i - s_i)``, ``s`` init 0.  Also a strictly
    positive reweighting (``e^{s_i} > 0``), so likewise inside Proposition 3.
    """

    def __init__(self, K, device):
        self.s = torch.nn.Parameter(torch.zeros(K, device=device))

    def weighted_loss(self, losses):
        return (self.s.exp() * losses - self.s).sum()

    def weights(self):
        return torch.exp(self.s.detach())

    def parameters(self):
        return [self.s]


class GradNormState:
    """Chen et al. 2018, *GradNorm*; reference ``LibMTL`` ``GradNorm``.

    ``loss_scale = K * softmax(theta)`` (``theta`` init 1.0); the restoring
    objective ``sum_i |G_i - G (r_i)^alpha_gn|`` with ``G_i = ||w_i g_i||``,
    ``G`` its mean and ``r_i`` the relative inverse training rate w.r.t. the
    first-epoch loss.  Epoch 0 is plain summation, as in the reference.
    ``alpha_gn=1.5`` is the paper's default.
    """

    def __init__(self, K, device, alpha_gn=1.5):
        self.K = K
        self.alpha_gn = alpha_gn
        self.theta = torch.nn.Parameter(torch.ones(K, device=device))
        self.buffer = None

    def weights(self):
        return self.K * torch.softmax(self.theta.detach(), dim=-1)

    def parameters(self):
        return [self.theta]

    def weighted_loss(self, losses):
        if self.buffer is None:
            self.buffer = [max(float(v), 1e-8) for v in losses.detach()]
        return (self.weights() * losses).sum()

    def update_w(self, losses, grads, params, epoch):
        """Sets ``theta.grad``; the caller's optimiser performs the step."""
        if epoch < 1:
            return
        w = self.K * torch.softmax(self.theta, dim=-1)
        G_per_loss = torch.stack([(w[i] * _flatten(grads[i], params)).norm()
                                  for i in range(self.K)])
        G = G_per_loss.mean()
        L_i = torch.stack([losses[i].detach() / self.buffer[i]
                           for i in range(self.K)])
        r_i = L_i / L_i.mean()
        constant_term = (G * (r_i ** self.alpha_gn)).detach()
        L_grad = (G_per_loss - constant_term).abs().sum()
        L_grad.backward()


class NashMTLState:
    """Navon et al. 2022, *Multi-Task Learning as a Bargaining Game*; reference
    ``AvivNavon/nash-mtl`` ``NashMTL`` == ``LibMTL`` ``Nash_MTL``.

    Solves the Nash bargaining programme for the weight vector ``alpha`` on the
    ``||GG'||_F``-normalised Gram matrix ``G G'`` (per-step, warm-started from
    the previous solution), then combines and clips the shared gradient to
    ``max_norm=1`` exactly as the reference does.

    The cvxpy/ECOS interior-point solve of the reference is used when cvxpy is
    importable; otherwise ``_nash_newton`` solves the same KKT system
    ``alpha_i (G alpha)_i = 1`` by damped Newton, which is the programme the
    reference hands to the solver.
    """

    def __init__(self, K, device, max_norm=1.0, optim_niter=20,
                 update_weights_every=1):
        self.K = K
        self.max_norm = max_norm
        self.optim_niter = optim_niter
        self.every = update_weights_every
        self.step = 0
        self.prvs_alpha = np.ones(K, dtype=np.float32)
        self.normalization_factor = np.ones((1,))
        self._prob = None

    # -- reference solver -------------------------------------------------
    def _init_problem(self):
        self.alpha_param = cp.Variable(shape=(self.K,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(shape=(self.K,),
                                             value=self.prvs_alpha)
        self.G_param = cp.Parameter(shape=(self.K, self.K), value=np.eye(self.K))
        self.norm_param = cp.Parameter(shape=(1,), value=np.array([1.0]))
        G_prvs = self.G_param @ self.prvs_alpha_param
        phi = (1 / self.prvs_alpha_param + (1 / G_prvs) @ self.G_param) @ (
            self.alpha_param - self.prvs_alpha_param)
        G_alpha = self.G_param @ self.alpha_param
        cons = [-cp.log(self.alpha_param[i] * self.norm_param)
                - cp.log(G_alpha[i]) <= 0 for i in range(self.K)]
        self._prob = cp.Problem(cp.Minimize(cp.sum(G_alpha)
                                           + phi / self.norm_param), cons)

    def _solve_reference(self, gtg):
        self.G_param.value = gtg
        self.norm_param.value = self.normalization_factor
        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t
            try:
                self._prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except Exception:
                self.alpha_param.value = self.prvs_alpha_param.value
            if (self.alpha_param.value is None
                    or np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3
                    or np.linalg.norm(self.alpha_param.value
                                      - self.prvs_alpha_param.value) < 1e-6):
                break
            alpha_t = self.alpha_param.value
        if alpha_t is not None:
            self.prvs_alpha = alpha_t
        return self.prvs_alpha

    def alpha(self, G):
        if (self.step % self.every) == 0:
            self.step += 1
            if cp is not None and self._prob is None:
                self._init_problem()
            GTG = _f32(G) @ _f32(G).t()
            self.normalization_factor = (
                torch.norm(GTG).detach().cpu().double().numpy().reshape((1,)))
            gtg = (GTG / self.normalization_factor.item()).detach().cpu().double().numpy()
            if cp is not None:
                a = self._solve_reference(gtg)
            else:
                a = _nash_newton(gtg, self.prvs_alpha, self.optim_niter)
                self.prvs_alpha = a
        else:
            self.step += 1
            a = self.prvs_alpha
        return torch.as_tensor(np.asarray(a), dtype=G.dtype, device=G.device)

    @_fp32_weights
    def combine(self, grads, params):
        G = _flat_stack(grads, params)
        alpha = self.alpha(G)
        g = _merge(G, alpha, params)
        if self.max_norm > 0:
            with torch.no_grad():
                total = torch.sqrt(sum((t.float() ** 2).sum() for t in g))
                if float(total) > self.max_norm:
                    s = self.max_norm / (float(total) + 1e-12)
                    g = [t * s for t in g]
        return g

    def weighted_loss(self, losses):
        return None

    def parameters(self):
        return []


def _nash_newton(gtg, alpha0, n_iter=20):
    """Solver-free Nash-MTL: damped Newton on ``alpha_i (G alpha)_i = 1``,
    ``alpha >= 0``.  Same KKT system as the reference's convex programme."""
    G = torch.as_tensor(gtg, dtype=torch.float64)
    K = G.shape[0]
    a = torch.as_tensor(np.asarray(alpha0), dtype=torch.float64).clone()
    a = a.clamp_min(1e-6)
    for _ in range(max(n_iter * 5, 50)):
        Ga = G @ a
        F = a * Ga - 1.0
        if float(F.abs().max()) < 1e-10:
            break
        J = torch.diag(Ga) + torch.diag(a) @ G
        try:
            step = torch.linalg.solve(J, F)
        except Exception:
            step = F / (J.diagonal().abs() + 1e-12)
        t, ok = 1.0, False
        for _ in range(40):                      # backtracking line search
            cand = (a - t * step).clamp_min(1e-8)
            if float((cand * (G @ cand) - 1.0).abs().max()) < float(F.abs().max()):
                ok = True
                break
            t *= 0.5
        a = (a - t * step).clamp_min(1e-8) if not ok else cand
    return a.numpy().astype(np.float32)


# ------------------------------------------------------------ registry glue
def make_state(method, K, device):
    """Learner-side state for the methods that carry parameters."""
    if method == "famo":
        return FAMOState(K, device)
    if method == "uw":
        return UWState(K, device)
    if method == "imtll":
        return IMTLLossState(K, device)
    if method == "gradnorm":
        return GradNormState(K, device)
    if method in ("nash", "nash_newton"):
        return NashMTLState(K, device)
    return None


def state_parameters(state):
    """Parameters a stateful method contributes to the *main* optimiser.
    FAMO returns nothing: it carries its own Adam over the logits, exactly as
    the reference does."""
    if state is None:
        return []
    return list(state.parameters())


def shape_stand_ins(vecs):
    """Shape/dtype-correct stand-ins for the parameter list, built from the
    gradient tensors themselves.  The combiners read ``params`` only for
    ``numel``/``shape``/``dtype`` when flattening and restoring -- never for
    values -- so this lets the legacy list-of-tensors call sites (which keep no
    parameter objects) drive the identical, unmodified kernel.  Device is taken
    from the gradient tensor so the merged gradient lands on the same device as
    the parameters it is written back to (assigning a CPU tensor to
    ``cuda_param.grad`` raises)."""
    n = max(len(v) for v in vecs)
    out = []
    for i in range(n):
        t = next((v[i] for v in vecs if i < len(v) and v[i] is not None), None)
        out.append(torch.zeros(t.shape, dtype=t.dtype, device=t.device)
                   if t is not None else torch.zeros(0))
    return out


def combine_flat(method, vecs, state=None):
    """Apply a combiner to raw per-task gradient lists.

    ``vecs`` is ``[K]`` lists of per-parameter tensors (``None`` allowed), in a
    fixed parameter order.  Returns ``(merged, info)``; ``merged`` has the same
    structure and can be written straight onto ``p.grad``.
    """
    params = shape_stand_ins(vecs)
    if method == "nash":
        st = state if state is not None else NashMTLState(len(vecs), params[0].device)
        return st.combine(vecs, params), np.asarray(st.prvs_alpha)
    if method not in STATELESS:
        raise KeyError("no flat combiner for %r (loss-weighting methods must be "
                       "driven through weighted_loss/update_w)" % method)
    return STATELESS[method](vecs, params), None


def combine(method, grads, params, state=None, losses=None):
    """Dispatch.  ``losses`` is the K-vector of *scaled* per-task losses
    ``[L_main, alpha * L_tox]`` -- the same instance the ``sum`` baseline
    optimises, so every method sees one and the same problem."""
    if method in STATELESS:
        return STATELESS[method](grads, params), None
    if method == "nash":
        st = state if state is not None else NashMTLState(len(grads),
                                                         grads[0][0].device)
        return st.combine(grads, params), np.asarray(st.prvs_alpha)
    if method in ("famo", "uw", "imtll", "gradnorm"):
        raise ValueError("loss-weighting method %r must be driven through "
                         "weighted_loss/update_w, not combine()" % method)
    raise KeyError(method)


WEIGHTING = ("famo", "uw", "imtll", "gradnorm")
FAMILY = {
    "sum": "baseline",
    "pcgrad": "pairwise direction",
    "cagrad": "pairwise direction",
    "cagrad04": "pairwise direction",
    "cagrad08": "pairwise direction",
    "mgda": "Pareto",
    "aligned": "gradient geometry",
    "imtlg": "gradient geometry",
    "nash": "gradient geometry",
    "famo": "task progress",
    "uw": "scalar loss weights",
    "imtll": "scalar loss weights",
    "gradnorm": "scalar loss weights",
}
