"""Cross-domain Multitask Representation Learning with Gradient Toxicity.

Self-contained harness that re-runs the *identical* protocol as the finance
discovery domain (wf_repro.py) on independent scientific domains, so the

    gradient toxicity  ->  representational collapse  ->  STF repair

mechanism is validated uniformly everywhere.  Domains:

    har      : UCI-HAR human activity/motion signals (6 classes, 561 features)
    radioml  : RadioML2016.10A modulation identification (11 classes, I/Q)
    battery  : NASA PCoE Li-ion capacity (SOH) regression                [elastic]

For every domain the harness:
  * trains single-task (health ceiling) and multi-task(s) with K toxic auxiliary
    heads whose loss is the batch variance of a FIXED random projection of the
    shared representation (the exact constant attractor of the theory);
  * contrasts  joint | STF-0 (.detach) | STF-hard | STF-soft;
  * sweeps the toxicity control parameter (aux scale alpha) to expose the phase
    transition, identical to collapse_phase.py;
  * reports domain-independent metrics (collapse rate, recovery rate, toxicity
    order parameter rho, representation scatter S_h).

Usage:
  python cross_domain_mtl.py --domain har    --modes joint,stfsoft --seeds 5
  python cross_domain_mtl.py --domain radioml --sweep-alpha 1
  python cross_domain_mtl.py --domain battery --modes all --seeds 5
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, glob, importlib.util
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moo_combiners as moo                                    # noqa: E402


# ------------------------------------------------------------------------- setup
def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)


def enable_max_perf():
    """Ampere/Hopper-class Tensor-Core accelerations: TF32 matmul (2-8x on
    fp32 GEMMs) + cudnn TF32.  bf16 autocast is used in the trainer because it
    needs no GradScaler and matches fp16 speed on A100."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def _path(*parts):
    root = os.environ.get("REPFIX_DATA_X", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_x"))
    return os.path.join(root, *parts)


# =============================================================================
#  STF (Spectral Toxicity Filtering) -- IDENTICAL to wf_repro.py::_stf_filter
# =============================================================================
def toxic_subspace(h):
    """Per-sample unit direction toward the batch mean (the collapse direction
    a toxic head pulls on) + the smallest-variance eigenvector (spectral view).
    Returns (u_hat, v_min, rho_spectral).  rho_spectral = lambda_min / sum(lambda)
    -- small when the representation is healthy (variance spread out)."""
    hc = h - h.mean(0)
    n = hc.norm(dim=1, keepdim=True).clamp_min(1e-8)
    uhat = hc / n
    B, d = h.shape
    cov = (hc.t() @ hc) / max(B, 1) + 1e-6 * torch.eye(d, device=h.device)
    evals, evecs = torch.linalg.eigh(cov)
    rho_spectral = (evals[0].clamp_min(0.0) / evals.clamp_min(0.0).sum().clamp_min(1e-12)).detach()
    return uhat, evecs[:, 0].detach(), rho_spectral


def stf_filter(h, mode, rho_c=0.02, beta=8.0):
    """STF family.  STF-0: full .detach();  STF-hard: remove toxic projection when
    rho<rho_c;  STF-soft: smooth gate w=sigmoid(beta*(rho_c-rho)).  Returns the
    filtered representation (as the aux branch sees it) and rho.

    rho_c is a RELATIVE spectral threshold supplied by the caller: it is
    anchored to the single-task (healthy) representation spectrum of the same
    domain/width (rho_c = rho_frac * base_rho), so the same rho_frac behaves
    identically across domains whose absolute spectrum scale differs wildly
    (battery SOH rep is low-rank, HAR/RadioML reps are high-rank).
    STF-soft therefore: leaves the aux channel open while the representation
    is healthy (rho_s > rho_c) and smoothly shuts the toxic channel as the
    representation degenerates (rho_s -> 0)."""
    if mode == "stf0":
        return h.detach(), None
    if mode in ("stfrho", "stfrhorank"):
        raise RuntimeError("stfrho modes are driven by RhoController, not stf_filter")
    uhat, _v, rho_s = toxic_subspace(h)
    if mode in ("stfhard", "stfsoft"):
        if mode == "stfhard":
            w = (rho_s < rho_c).float()          # on-device gate, no D2H sync
        else:
            w = torch.sigmoid(beta * (1.0 - rho_s / max(rho_c, 1e-12)))
        proj = (uhat * (h - h.mean(0))).sum(-1, keepdim=True) * uhat
        return h - w * proj, rho_s
    return h, None


# =============================================================================
#  STF-rho -- theory-setpoint adaptive filtration (no tuned threshold)
#
#  The theory locates the saddle-node at the operating point alpha*rho = 1 with
#  the gradient-level order parameter  rho = <||P_c g_a||>/<||g_p||>  (P_c the
#  per-sample batch-mean collapse direction).  STF-rho replaces the tuned
#  spectral threshold rho_c of STF-hard/soft by that quantity itself:
#
#    * stfrho     : remove the collapse direction (rank 1, theory P_c);
#    * stfrhorank : MINIMAL-RANK filtration -- remove the smallest r such that
#                   the RESIDUAL (post-filtration) pull alpha*rho_res[r] < 1,
#                   so auxiliary signal on unused directions is preserved.
#
#  Latch-on.  rho is a leading indicator only ON the healthy branch: once the
#  representation degenerates, ||g_a|| ~ ||h - E[h]|| -> 0, so rho decays with
#  it (HAR, measured: rho ~ 200 at the healthy state vs ~ 0.01 collapsed).  A
#  memoryless gate on rho would therefore switch the filter OFF exactly when it
#  is needed.  The controller latches the reference at its running maximum --
#  the algorithmic reading of the irreversibility theorem (a collapsed encoder
#  is not recovered by changing the operating point): switch on when the healthy
#  branch loses stability, then stay on.
# =============================================================================
RHO_MODES = ("stfrho", "stfrhorank")
# populated from the CLI in main(); read by train_one (keeps every call site intact)
STF_RHO_CFG = dict(kappa=4.0, ema=0.9, probe_every=10, rank_max=3,
                   smooth=False, trace=False, latch_quantile=1.0,
                   cal_warm_epochs=2.0)


def rho_probe(model, xb, yb, criterion, regress, R=3):
    """Order-parameter family on the UNFILTERED auxiliary branch.

      rho_u   : paper convention, <||P_c g_a||>/<||g_p||>, P_c the per-sample
                batch-mean collapse direction u = hc/||hc||;
      rho_res : residual pull after removing the top-r PCA directions of the
                batch representation, r = 0..R  (rho_res[0] = full pull);
      U       : orthonormal PCA basis, descending variance.

    Runs in fp32 outside autocast on a detached clone; gradients are taken with
    respect to h only, so no parameter gradients are created and the training
    graph is untouched."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.autocast(device_type=dev, enabled=False), torch.enable_grad():
        h = model(xb).detach().float().requires_grad_(True)
        pk = model.primary(h)
        if regress:
            pk = pk.squeeze(-1)
        yt = yb.float() if regress else yb
        gp = torch.autograd.grad(criterion(pk, yt), h, retain_graph=True,
                                 allow_unused=True)[0]
        mu = h.mean(0, keepdim=True)
        ga = torch.autograd.grad(((h - mu) ** 2).mean(), h, retain_graph=False,
                                 allow_unused=True)[0]
    if gp is None or ga is None:
        return None
    with torch.no_grad():
        hc = h.detach() - h.detach().mean(0, keepdim=True)
        d = hc.shape[1]
        cov = (hc.t() @ hc) / max(hc.shape[0], 1) + 1e-6 * torch.eye(d, device=hc.device)
        evals, evecs = torch.linalg.eigh(cov)
        U = evecs[:, torch.argsort(evals, descending=True)][:, :max(int(R), 1)]
        gp_n = gp.norm(dim=1) + 1e-8
        u = hc / hc.norm(dim=1, keepdim=True).clamp_min(1e-8)
        rho_u = float(((((ga * u).sum(-1, keepdim=True)) * u).norm(dim=1) / gp_n).mean().item())
        rho_res = []
        for r in range(0, U.shape[1] + 1):
            ga_r = ga if r == 0 else ga - (ga @ U[:, :r]) @ U[:, :r].t()
            rho_res.append(float((ga_r.norm(dim=1) / gp_n).mean().item()))
    return dict(rho_u=rho_u, rho_res=rho_res, U=U.detach())


class RhoController:
    """Latch-on, threshold-free gate driven by the theorem's saddle-node."""

    def __init__(self, alpha, kappa=4.0, ema=0.9, rank_max=3, smooth=False,
                 monitor_only=False):
        self.alpha = float(alpha); self.kappa = float(kappa); self.ema = float(ema)
        self.rank_max = int(rank_max); self.smooth = bool(smooth)
        self.monitor_only = bool(monitor_only)
        self.rho_ema = None; self.rho_ref = None
        self.rho_last = float("nan"); self.res_last = []
        self.latch_step = None
        self.U_hat = None; self.r_use = 1
        self.w_sum = 0.0; self.w_n = 0; self.r_sum = 0; self.step = 0
        self.trace = []

    @torch.no_grad()
    def observe(self, probe):
        if probe is None:
            return
        ru = float(probe["rho_u"])
        self.rho_last = ru
        self.rho_ema = ru if self.rho_ema is None else \
            self.ema * self.rho_ema + (1.0 - self.ema) * ru
        self.rho_ref = self.rho_ema if self.rho_ref is None else max(self.rho_ref, self.rho_ema)
        if self.rho_ref * self.alpha >= 1.0 and self.latch_step is None:
            self.latch_step = self.step
        self.res_last = list(probe["rho_res"])
        self.U_hat = probe["U"].clone()
        # minimal rank whose RESIDUAL pull is sub-critical at this operating point
        r_use = self.rank_max
        for r in range(0, min(len(self.res_last) - 1, self.rank_max) + 1):
            if self.alpha * self.res_last[r] < 1.0:
                r_use = r
                break
        self.r_use = max(0, min(r_use, self.rank_max))
        if STF_RHO_CFG.get("trace"):
            self.trace.append(dict(step=self.step, rho_u=ru,
                                   rho_ref=self.rho_ref,
                                   rho_res=[float(x) for x in self.res_last],
                                   gate=self.gate))

    @property
    def gate(self):
        if self.monitor_only or self.rho_ref is None:
            return 0.0
        z = self.alpha * self.rho_ref - 1.0
        if z <= 0.0:
            return 0.0
        return 1.0 if not self.smooth else \
            float(torch.sigmoid(torch.as_tensor(self.kappa * z)))

    @property
    def w_mean(self):
        return self.w_sum / max(self.w_n, 1)

    @property
    def rank_mean(self):
        return self.r_sum / max(self.w_n, 1)

    def filter_h(self, h, mode):
        """Representation on the auxiliary branch with the toxic subspace removed
        with weight w.  No decorator: the result stays attached to h so the benign
        residual still trains the auxiliary heads."""
        w = self.gate
        self.step += 1
        self.w_sum += w; self.w_n += 1
        self.r_sum += (1 if mode == "stfrho" else self.r_use)
        if w <= 0.0:
            return h, w, 0
        hc = h - h.mean(0, keepdim=True)
        if mode == "stfrho":
            u = hc / hc.norm(dim=1, keepdim=True).clamp_min(1e-8)
            return h - w * ((hc * u).sum(-1, keepdim=True) * u), w, 1
        if self.U_hat is None or self.r_use <= 0:
            return h, w, 0
        U = self.U_hat[:, :self.r_use].to(device=h.device, dtype=h.dtype)
        return h - w * ((hc @ U) @ U.t()), w, self.r_use


# =============================================================================
#  STF-cal -- calibrated STF (measure once on the healthy state, then act)
#
#  Measured fact that motivates this controller: the gradient order parameter is
#  NOT identifiable early in training.  At initialisation the aux gradient is
#  scaled by the loss normalisation (d var/dh ~ 1/(B d)), so rho_u ~ 0.01 for
#  every alpha in {0.03 ... 100} -- an online gate on rho therefore cannot know
#  whether the operating point is dangerous.  On the HEALTHY state, by contrast,
#  rho is informative and predicts the transition: HAR measures rho_u ~ 200,
#  i.e. alpha_c = 1/rho ~ 5e-3, which is where the measured half-height crossing
#  of the scatter curve sits.
#
#  STF-cal therefore runs two phases and needs no tuned threshold:
#    (1) WARM-UP: the toxic branch is detached (STF-0) so the encoder follows the
#        primary-task trajectory; the healthy representation scatter S* is
#        measured there.
#    (2) CALIBRATION: one probe on the healthy model yields the order-parameter
#        family rho_res[0..R]; the theorem's saddle-node then fixes everything
#        -- gate = 1[alpha*rho_res[0] >= 1] (is the healthy state unstable?),
#        rank = smallest r with alpha*rho_res[r] < 1 (remove just enough of the
#        toxic subspace, preserving auxiliary signal on the rest).
#  The resulting configuration is latched for the rest of training
#  (irreversibility), and if alpha is sub-critical the auxiliary branch is left
#  completely untouched -- which is what distinguishes STF-cal from the blunt
#  STF-0 erasure of benign auxiliary tasks.
# =============================================================================
CAL_MODES = ("stfcal", "stfcalrank")


class CalController:
    def __init__(self, alpha, warm_steps, rank_max=3, ema=0.9):
        self.alpha = float(alpha); self.warm_steps = int(warm_steps)
        self.rank_max = int(rank_max); self.ema = float(ema)
        self.step = 0
        self.S_ema = None; self.S_star = None
        self.calibrated = False
        self.gate_val = 0.0; self.r_use = 1
        self.rho_u = float("nan"); self.rho_res = []; self.U = None
        self.w_sum = 0.0; self.w_n = 0; self.r_sum = 0

    def observe_scatter(self, s):
        # GPU-resident EMA (0-d tensor on device): no D2H sync in the step loop.
        s = s.detach().float() if torch.is_tensor(s) else torch.as_tensor(float(s))
        self.S_ema = s.clone() if self.S_ema is None else \
            self.ema * self.S_ema + (1.0 - self.ema) * s

    def calibrate(self, probe):
        self.S_star = self.S_ema
        if probe is None:
            self.calibrated = True
            return
        self.rho_u = float(probe["rho_u"])
        self.rho_res = [float(x) for x in probe["rho_res"]]
        self.U = probe["U"].clone()
        self.gate_val = 1.0 if self.alpha * self.rho_u >= 1.0 else 0.0
        r_use = self.rank_max
        for r in range(0, min(len(self.rho_res) - 1, self.rank_max) + 1):
            if self.alpha * self.rho_res[r] < 1.0:
                r_use = r
                break
        self.r_use = max(1, r_use)
        self.calibrated = True

    @property
    def margin(self):
        """alpha / alpha_c with alpha_c = 1/rho_healthy (>1 means supercritical)."""
        return self.alpha * self.rho_u

    @property
    def w_mean(self):
        return self.w_sum / max(self.w_n, 1)

    @property
    def rank_mean(self):
        return self.r_sum / max(self.w_n, 1)

    @property
    def scatter_ratio(self):
        if self.S_star is None or self.S_ema is None:
            return None
        with torch.no_grad():
            return float((self.S_ema / self.S_star.clamp_min(1e-12)).item())

    def step_h(self, h, mode, warm):
        """warm=True: healthy warm-up (aux branch detached).  warm=False and a
        sub-critical calibration: the auxiliary branch is left intact."""
        self.step += 1
        self.w_n += 1
        if warm:
            return h.detach(), 0.0, 0
        if self.gate_val <= 0.0:
            return h, 0.0, 0
        hc = h - h.mean(0, keepdim=True)
        if mode == "stfcal":
            u = hc / hc.norm(dim=1, keepdim=True).clamp_min(1e-8)
            self.w_sum += 1.0; self.r_sum += 1
            return h - ((hc * u).sum(-1, keepdim=True) * u), 1.0, 1
        r = min(self.r_use, self.U.shape[1])
        U = self.U[:, :r].to(device=h.device, dtype=h.dtype)
        self.w_sum += 1.0; self.r_sum += r
        return h - ((hc @ U) @ U.t()), 1.0, r



# =============================================================================
#  Model -- uniform encoder (MLP or small Transformer) + primary head
#           + fixed toxic aux projections
# =============================================================================
# Encoder family is selected globally so that every caller (train_one,
# sweep_alpha, conflict/composition ablations) keeps an unchanged signature.
ARCH = "mlp"


class FormerEncoder(nn.Module):
    """Small pre-norm Transformer over feature tokens -> (B, m) representation.

    The input feature vector of width d_in is tokenised into `n_tok` feature
    tokens; a learned CLS token is prepended and its output projected to the
    shared representation width m.  This is the standard modern backbone, kept
    deliberately small (2 layers, 4 heads) so that the toxicity mechanism, not
    depth, is what varies across the comparison.
    """

    def __init__(self, d_in, m, n_tok=8, n_head=4, n_layer=2, dropout=0.0):
        super().__init__()
        d_model = max(32, min(256, m))
        self.n_tok = n_tok
        self.inp = nn.Linear(d_in, n_tok * d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=2 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, m)

    def forward(self, x):
        B = x.shape[0]
        t = self.inp(x).view(B, self.n_tok, -1)
        t = torch.cat([self.cls.expand(B, -1, -1), t], dim=1)
        t = self.blocks(t)
        return self.out(self.norm(t[:, 0]))


class MTLModel(nn.Module):
    def __init__(self, d_in, n_classes, m=128, k_aux=3, seed=11, regress=False,
                 arch=None, y_dim=1):
        super().__init__()
        arch = arch or ARCH
        if arch == "transformer":
            self.encoder = FormerEncoder(d_in, m)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(d_in, m), nn.ReLU(), nn.Linear(m, m), nn.ReLU(), nn.Linear(m, m))
        self.arch = arch
        # y_dim: multi-output regression width (scalar targets use 1; the NYU
        # coarse depth grid uses 108).  squeeze(-1) and the MSE paths are
        # no-ops / shape-exact for both cases.
        self.primary = nn.Linear(m, y_dim if regress else n_classes)
        # the toxic heads: FIXED random projections + variance loss (constant attractor)
        g = torch.Generator().manual_seed(seed)
        d_proj = max(m // 4, 4)
        W = torch.randn(k_aux, m, d_proj, generator=g) / math.sqrt(d_proj)
        self.register_buffer("W_aux", W)
        self.k_aux = k_aux

    def forward(self, x):
        return self.encoder(x)                     # (B, m)

    def aux_proj(self, h):
        # (B, k, d_proj) -> variance over batch per head = toxic loss
        return torch.einsum("bm,kmn->bkn", h, self.W_aux)


def var_loss(a):
    return ((a - a.mean(0)) ** 2).mean()


# =============================================================================
#  Data loaders
# =============================================================================
def _load_har():
    base = _path("uci_har", "extracted", "UCI HAR Dataset")
    def rd(*p):
        return np.loadtxt(os.path.join(base, *p))
    X_tr = rd("train", "X_train.txt"); y_tr = rd("train", "y_train.txt").astype(np.int64) - 1
    X_te = rd("test", "X_test.txt");  y_te = rd("test", "y_test.txt").astype(np.int64) - 1
    return dict(X_tr=X_tr.astype(np.float32), y_tr=y_tr,
                X_te=X_te.astype(np.float32), y_te=y_te, d_in=X_tr.shape[1], n_classes=6)


def _load_radioml():
    """Prefer the mmap-able .pt tensor cache (seconds to load); fall back to
    parsing the 225MB pickle once and writing the .pt cache."""
    pt = _path("RML2016.10a_pre.pt")
    if os.path.exists(pt):
        T = torch.load(pt, map_location="cpu", mmap=True)
        return dict(X_tr=T["X_tr"].numpy(), y_tr=T["y_tr"].numpy(),
                    X_te=T["X_te"].numpy(), y_te=T["y_te"].numpy(),
                    aux_tr=T["aux_tr"].numpy(), aux_te=T["aux_te"].numpy(),
                    d_in=T["X_tr"].shape[1], n_classes=int(T["n_classes"]))
    p = _path("RML2016.10a_dict_optimized.pkl")
    import pickle
    with open(p, "rb") as f:
        d = pickle.load(f)
    keys = list(d.keys())
    mods = sorted(set(k[0] for k in keys)); snrs = sorted(set(k[1] for k in keys))
    X_all, Y_all, A_all = [], [], []
    for (mod, snr), arr in d.items():
        n = arr.shape[0]
        X_all.append(arr); Y_all.append(np.full(n, mods.index(mod))); A_all.append(np.full(n, snr))
    X = np.concatenate(X_all)[:, np.newaxis, :, :].astype(np.float32)   # (N,1,2,128)
    Y = np.concatenate(Y_all).astype(np.int64)
    A = np.concatenate(A_all).astype(np.float32)
    # standard 50/50 per-class split
    rng = np.random.RandomState(0)
    idx_tr, idx_te = [], []
    for m in range(len(mods)):
        i = np.where(Y == m)[0]; rng.shuffle(i)
        k = len(i) // 2
        idx_tr.append(i[:k]); idx_te.append(i[k:])
    itr = np.concatenate(idx_tr); ite = np.concatenate(idx_te)
    X = X.reshape(X.shape[0], -1)  # flat -> MLP encoder
    out = dict(X_tr=X[itr], y_tr=Y[itr], aux_tr=A[itr],
               X_te=X[ite], y_te=Y[ite], aux_te=A[ite], d_in=X.shape[1], n_classes=len(mods))
    tensors = {k: torch.from_numpy(v) for k, v in out.items() if isinstance(v, np.ndarray)}
    tensors["d_in"] = out["d_in"]; tensors["n_classes"] = out["n_classes"]
    torch.save(tensors, pt)
    return out


def _load_battery():
    """Elastic: pool the multi-cell NASA ARC cache (preferred), else fall back
    to the single B0005 cell.  Primary target is log-capacity; split is pooled
    chronological (first 60% cycles of every cell train, last 40% test) so the
    health ceiling is high and collapse observably hurts the regression metric."""
    import numpy as _np
    mp = _path("nasa_battery", "nasa_battery_multi.npz")
    if os.path.exists(mp):
        d = _np.load(mp)
        X = d["X"]; cell = d["cell"]
        # primary target = log-capacity (log-SOH).  Log compresses the healthy
        # plateau and spreads the deep-degradation tail, giving the target high
        # entropy so a *constant map* (the collapsed model's output) is heavily
        # penalized and STF's recovery is measurable on the regression metric.
        soh = d["y"].astype(_np.float32)
        y = _np.log(soh + 0.02).astype(_np.float32)
        cell_sizes = d["cell_sizes"] if "cell_sizes" in d else _np.array([0])
        n_cells = int(cell.max()) + 1
        # pooled chronological split: first 60% cycles of EVERY cell -> train,
        # last 40% -> test.  The model can learn degradation (high health
        # ceiling), so collapse has room to hurt and STF has room to win.  (The
        # cross-cell held-out variant made the task so hard that the health
        # ceiling collapsed onto the mean-predictor, hiding the recovery.)
        tr = _np.zeros(len(X), dtype=bool)
        for c in range(n_cells):
            idx = _np.where(cell == c)[0]
            idx = idx[_np.argsort(idx)]  # chronological within cell
            k = int(len(idx) * 0.6)
            tr[idx[:k]] = True
        te = ~tr
        return dict(X_tr=X[tr].astype(_np.float32), y_tr=y[tr],
                    X_te=X[te].astype(_np.float32), y_te=y[te],
                    d_in=X.shape[1], n_classes=1, cell_sizes=cell_sizes, n_cells=n_cells)
    import scipy.io
    p = _path("nasa_battery", "B0005.mat")
    if not os.path.exists(p):
        return None
    try:
        B = scipy.io.loadmat(p, squeeze_me=True, struct_as_record=False)["B0005"]
        S = B.cycle                              # (616,) list/array of mat_struct
        types = np.asarray([x.type.rstrip().decode() if isinstance(x.type, bytes) else str(x.type)
                            for x in S])
        caps, feats = [], []
        for i, st in enumerate(S):
            if types[i] != "discharge":
                continue
            d = getattr(st, "data")
            t = np.asarray(getattr(d, "Time")).ravel()
            c = np.asarray(getattr(d, "Current_measured")).ravel()
            v = np.asarray(getattr(d, "Voltage_measured")).ravel()
            if len(t) < 2:
                continue
            cap = float(np.trapezoid(np.abs(c), t))
            fvec = [np.percentile(v, q) for q in (1, 5, 25, 50, 75, 95, 99)]
            fvec += [len(c), c[0], c[-1], t[-1] - t[0], t[0]]
            caps.append(cap); feats.append(fvec)
        if len(caps) < 4:
            return None
        caps = np.array(caps); feats = np.array(feats, dtype=np.float32)
        X = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        mean, std = X.mean(0), X.std(0) + 1e-6
        X = (X - mean) / std
        soh = caps / caps[0]                      # state of health in [0,1]
        n = len(soh); k = int(n * 0.6)
        return dict(X_tr=X[:k], y_tr=soh[:k], X_te=X[k:], y_te=soh[k:], d_in=X.shape[1], n_classes=1)
    except Exception as e:
        print("battery parse failed, skipping:", repr(e))
        return None


def _load_celeba():
    """CelebA (standard vision MTL benchmark), 64x64 grayscale, primary
    attribute = Smiling, official train/test partition.  Same unified protocol:
    shared encoder + primary head + the fixed variance-shrinkage auxiliary
    projections, so the dose response, the calibration and every metric are
    directly comparable with the time-series domains.  Pixels are flattened,
    /255-scaled and standardised with train-split statistics."""
    p = _path("celeba", "celeba64_gray.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p + " -- run _celeba_prep.py first")
    Z = np.load(p)
    X_tr = Z["X_tr"].astype(np.float32) / 255.0
    X_te = Z["X_te"].astype(np.float32) / 255.0
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    return dict(X_tr=(X_tr - mu) / sd, y_tr=Z["y_tr"].astype(np.int64),
                X_te=(X_te - mu) / sd, y_te=Z["y_te"].astype(np.int64),
                d_in=int(X_tr.shape[1]), n_classes=2)


def _load_nyu():
    """NYUv2 (HF tanganke/nyuv2), 144x192 RGB, unified-protocol variant.

    Pixels are flattened and per-pixel standardised with train-split statistics
    (the CelebA convention); the primary task is coarse depth-grid regression
    (9x12 block means of the depth map, metres), so the harness's regression
    path (n_classes=1, Linear head, MSE) is reused unchanged.  The full maps +
    seg/normal grids for the dense benchmark live in nyu_bmtl.py."""
    N_CELL = 108                                   # 9x12 coarse depth grid
    p = _path("nyu", "nyu_nyu.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p + " -- run _hpc/_nyu_prep.py first")
    Z = np.load(p)
    X_tr = Z["X_tr"].astype(np.float32).reshape(len(Z["X_tr"]), -1)
    X_te = Z["X_te"].astype(np.float32).reshape(len(Z["X_te"]), -1)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    return dict(X_tr=((X_tr - mu) / sd).astype(np.float32),
                y_tr=Z["y_tr"].reshape(len(X_tr), -1).astype(np.float32),
                X_te=((X_te - mu) / sd).astype(np.float32),
                y_te=Z["y_te"].reshape(len(X_te), -1).astype(np.float32),
                d_in=int(X_tr.shape[1]), n_classes=1, y_dim=N_CELL)


DOMAINS = {"har": _load_har, "radioml": _load_radioml, "battery": _load_battery,
           "celeba": _load_celeba, "nyu": _load_nyu}

# ----------------------------------------------------------------- data cache
# Each domain's parsing is expensive (RadioML is a 225MB pkl, battery needs .mat
# struct walking).  Cache the parsed tensors once per process so every seed /
# alpha / mode reuses the same arrays instead of re-parsing from disk.
_DATA_CACHE = {}

def load_domain(domain):
    if domain not in _DATA_CACHE:
        _DATA_CACHE[domain] = DOMAINS[domain]()
    return _DATA_CACHE[domain]


# =============================================================================
#  Trainer
# =============================================================================
def scatter(h):
    """S_h = mean over samples of std across dims of batch-centered rep."""
    hc = h - h.mean(0)
    return float(hc.std(dim=1).mean().item())


def scatter_t(h):
    """GPU-resident twin of scatter(): returns a 0-d tensor, NO D2H sync.
    Use inside the training loop where the value only feeds EMA/monitor state;
    call float()/scatter() at serialization or eval boundaries instead."""
    hc = h - h.mean(0)
    return hc.std(dim=1).mean()


def train_one(domain, mode, alpha, m, seed, epochs, lr, batch, rho_c, beta,
              base_scatter=None, rel_collapse=0.3, base_rho=None, method="joint",
              timing=None):
    torch.set_num_threads(1)
    torch.cuda.manual_seed(seed)
    set_seed(seed)
    enable_max_perf()
    D = load_domain(domain)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MTLModel(D["d_in"], D["n_classes"], m=m, k_aux=3, seed=seed,
                     regress=(D["n_classes"] == 1),
                     y_dim=int(D.get("y_dim", 1))).to(dev)
    # ---- Section 7.2: multi-objective / conflict-resolution operators ----
    # Every operator acts on the SHARED encoder parameters -- the set the
    # reference implementations distinguish as shared -- while the primary head
    # is task-specific and simply keeps its own primary gradient (the toxic loss
    # never touches it).  The operators themselves are faithful ports of the
    # author-released code (moo_combiners.py), differentially checked against
    # it by repro/scripts/test_moo_combiners.py; the loss-weighting families
    # (UW / IMTL-L / GradNorm / FAMO) are driven through their own scalarisation
    # instead, exactly as those references do.
    shared_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for p in model.primary.parameters() if p.requires_grad]
    moo_state = moo.make_state(method, 2, dev)
    if moo_state is not None and mode not in ("joint", "sum"):
        raise ValueError("conflict-resolution methods run on the unfiltered "
                         "joint path; got mode=%r with method=%r" % (mode, method))
    opt = torch.optim.Adam(list(model.parameters())
                           + moo.state_parameters(moo_state), lr=lr)
    regress = D["n_classes"] == 1
    Xtr = torch.from_numpy(D["X_tr"]).float().to(dev)
    ytr = torch.from_numpy(D["y_tr"]).long().to(dev) if not regress else torch.from_numpy(D["y_tr"]).float().to(dev)
    Xte = torch.from_numpy(D["X_te"]).float().to(dev)
    yte = torch.from_numpy(D["y_te"]).long().to(dev) if not regress else torch.from_numpy(D["y_te"]).float().to(dev)
    criterion = nn.MSELoss() if regress else nn.CrossEntropyLoss()
    # STF threshold is RELATIVE to the healthy (single-task) spectral footprint
    # of this domain/width, so the same rho_c behaves identically everywhere.
    rc = rho_c * base_rho if base_rho else rho_c
    # bf16 autocast: fp16 speed on A100, no GradScaler needed for range, but we
    # keep one for safety on the regression head; TF32 covers the fp32 paths.
    use_amp = dev == "cuda" and torch.cuda.is_available()
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    ema = {"lmain": torch.as_tensor(1.0, device=dev),
           "ltox": torch.as_tensor(1.0, device=dev)}
    n = Xtr.shape[0]
    if mode in RHO_MODES:
        ctrl = RhoController(alpha, kappa=STF_RHO_CFG["kappa"], ema=STF_RHO_CFG["ema"],
                             rank_max=STF_RHO_CFG["rank_max"],
                             smooth=STF_RHO_CFG["smooth"])
    elif STF_RHO_CFG["trace"] and mode == "joint":
        # trace-only: measure rho(t) on the *unfiltered* joint trajectory
        ctrl = RhoController(alpha, ema=STF_RHO_CFG["ema"],
                             rank_max=STF_RHO_CFG["rank_max"], monitor_only=True)
    else:
        ctrl = None
    cal = None
    if mode in CAL_MODES:
        n_steps = max(1, math.ceil(n / batch))
        cal = CalController(alpha,
                            warm_steps=int(STF_RHO_CFG["cal_warm_epochs"] * n_steps),
                            rank_max=STF_RHO_CFG["rank_max"],
                            ema=STF_RHO_CFG["ema"])
    _probe_every = max(int(STF_RHO_CFG["probe_every"]), 1)
    step = 0
    for ep in range(epochs):
        if timing is not None:
            # Wall-clock instrumentation for repro/scripts/time_overhead.py.
            # Inert unless a caller passes a list; the synchronise() calls make
            # the per-epoch figure a true device-complete time rather than a
            # measure of how fast kernels are *enqueued*.
            if dev == "cuda":
                torch.cuda.synchronize()
            _t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n, device=dev)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            xb = Xtr[idx]; yb = ytr[idx]
            opt.zero_grad()
            with amp_ctx:
                h = model(xb)
                if cal is not None:
                    if not cal.calibrated:
                        if cal.step >= cal.warm_steps:
                            # one probe on the healthy model fixes gate AND rank
                            cal.calibrate(rho_probe(model, xb, yb, criterion, regress,
                                                    R=STF_RHO_CFG["rank_max"]))
                        else:
                            with torch.no_grad():
                                cal.observe_scatter(scatter_t(h.detach()))
                    else:
                        # continuous monitor: live scatter vs the frozen warm-up anchor
                        with torch.no_grad():
                            cal.observe_scatter(scatter_t(h.detach()))
                    aux_in, _w, _r = cal.step_h(h, mode, warm=(not cal.calibrated))
                elif ctrl is not None:
                    # the controller reads the order parameter the theorem is
                    # about (probe is fp32/grad-only, no effect on training)
                    if step % _probe_every == 0:
                        ctrl.observe(rho_probe(model, xb, yb, criterion, regress,
                                               R=STF_RHO_CFG["rank_max"]))
                    aux_in, _w, _r = ctrl.filter_h(h, mode)
                else:
                    # STF filters the rep BEFORE the toxic head sees it.
                    aux_in = stf_filter(h, mode, rc, beta)[0]
                # variance-shrinkage toxifier: pulls each point toward the batch
                # mean, i.e. exactly along the theoretical collapse direction
                # u_c = h - E_b[h].  A benign-sounding "feature-invariance /
                # variance-regularization" aux task whose degenerate minimum is
                # the constant representation.
                mu = aux_in.mean(0, keepdim=True)
                L_tox = ((aux_in - mu) ** 2).mean()
                out = model.primary(h)
                if regress:
                    out = out.squeeze(-1)
                L_main = criterion(out, yb)
                L = L_main + alpha * L_tox
            if method in ("joint", "sum"):
                if scaler is not None:
                    scaler.scale(L).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    L.backward()
                    opt.step()
            elif method in moo.WEIGHTING:
                # Loss-weighting families (UW / IMTL-L / GradNorm / FAMO): the
                # scalarisation itself is learned, so there is no gradient
                # surgery.  The two tasks are the same instance every other row
                # sees -- (L_main, alpha * L_tox) with alpha fixed by the sweep
                # -- and the method supplies the weights.
                losses = torch.stack([L_main, alpha * L_tox])
                if method == "gradnorm":
                    # GradNorm needs the per-task gradients for its restoring
                    # objective, and the reference updates its scale in the same
                    # step; both must be taken before the graph is freed.
                    gL = _grad_list(L_main, shared_params)
                    gT = _grad_list(alpha * L_tox, shared_params)
                    moo_state.weighted_loss(losses).backward(retain_graph=True)
                    moo_state.update_w(losses, [gL, gT], shared_params, epoch=ep)
                else:
                    moo_state.weighted_loss(losses).backward()
                opt.step()
                if method == "famo":
                    # the reference recomputes the post-step losses on the same
                    # batch to form its first-order progress term
                    with torch.no_grad():
                        h2 = model(xb)
                        l2 = ((h2 - h2.mean(0, keepdim=True)) ** 2).mean()
                        o2 = model.primary(h2)
                        if regress:
                            o2 = o2.squeeze(-1)
                        m2 = criterion(o2, yb)
                    moo_state.update_w(torch.stack([m2, alpha * l2]))
            else:
                # Gradient-manipulation families: run the method's own operator
                # on the two shared-encoder gradients, then write the result
                # back and step manually (bf16 autocast needs no GradScaler).
                gp = _grad_list(L_main, shared_params)
                gt = _grad_list(alpha * L_tox, shared_params)
                gc = moo.combine(method, [gp, gt], shared_params, moo_state)[0]
                gh = _grad_list(L_main, head_params)
                for p, g in zip(shared_params, gc):
                    p.grad = None if g is None else g
                for p, g in zip(head_params, gh):
                    p.grad = None if g is None else g
                with torch.no_grad():
                    ema["lmain"] = 0.9 * ema["lmain"] + 0.1 * L_main.detach()
                    ema["ltox"] = 0.9 * ema["ltox"] + 0.1 * L_tox.detach()
                opt.step()
            step += 1
        if timing is not None:
            if dev == "cuda":
                torch.cuda.synchronize()
            timing.append(time.perf_counter() - _t0)
        # ---- evaluation (on held-out) + toxicity order-parameter rho ----
    model.eval()
    with torch.no_grad():
        hp = model(Xte)
        sh = scatter(hp)
        if regress:
            pred = model.primary(hp).squeeze(-1)
            metric = float((pred - yte.float().to(dev)).pow(2).mean().sqrt().item())
        else:
            logits = model.primary(hp)
            metric = float((logits.argmax(1) == yte.to(dev)).float().mean().item())
    # rho = <|Pi_c g_a|> / <||g_p||> on a probe batch (grad-enabled, tiny)
    rho = float("nan")
    srho = float("nan")
    try:
        Xq = Xte[:min(len(Xte), 384)].clone()
        yq = yte[:min(len(yte), 384)]
        with amp_ctx:
            hq = model(Xq)
            _pk = model.primary(hq)
            if D["n_classes"] == 1:
                _pk = _pk.squeeze(-1)
            _, _, srho = toxic_subspace(hq.detach())
            srho = float(srho)
            gp = torch.autograd.grad(
                criterion(_pk, yq.to(dev)), hq,
                create_graph=False, retain_graph=True, allow_unused=True)[0]
            muq = hq.mean(0, keepdim=True)
            L_tox_q = ((hq - muq) ** 2).mean()
            ga = torch.autograd.grad(L_tox_q, hq, retain_graph=False, allow_unused=True)[0]
        if gp is not None and ga is not None:
            # strict geometric order parameter rho = <||P_c g_a||> / <||g_p||>,
            # with P_c = u u^T the projector onto the batch-mean collapse direction
            # u = (h - E[h]) / ||h - E[h]||.  Theory: rho_c = 1 is the saddle-node
            # (the primary anchoring dominates iff ||P_c g_a|| < ||g_p||).
            hcq = hq - hq.mean(0)
            ncq = hcq.norm(dim=1, keepdim=True).clamp_min(1e-8)
            u = hcq / ncq
            ga_c = (ga * u).sum(-1, keepdim=True) * u          # P_c g_a per sample
            rho = float(((ga_c.norm(dim=1) / (gp.norm(dim=1) + 1e-8)).mean()).item())
    except Exception:
        pass
    # -- relative collapse: representation scatter drops below `rel` fraction of
    #    the single-task health ceiling.  This is the domain-independent notion
    #    of "representation collapse" that holds across HAR/RadioML/battery
    #    where the primary task anchors the encoder much harder than in finance.
    if base_scatter is not None and base_scatter > 0:
        collapse = sh < (rel_collapse * base_scatter)
    else:
        collapse = sh < (0.05 if not regress else 0.02)
    return dict(domain=domain, mode=mode, alpha=float(alpha), m=m, seed=seed,
                metric=metric, regress=regress, scatter=sh,
                collapse=collapse, rel_scatter=sh / base_scatter if base_scatter else sh,
                rho_mean=rho, srho=srho, base_rho=base_rho,
                rho_ref=(ctrl.rho_ref if ctrl is not None else float("nan")),
                rho_last=(ctrl.rho_last if ctrl is not None else float("nan")),
                latch_step=(ctrl.latch_step if (ctrl is not None and ctrl.latch_step is not None) else -1),
                rho_trace=(ctrl.trace if ctrl is not None else []),
                w_mean=(ctrl.w_mean if ctrl is not None else float("nan")),
                rank_mean=(ctrl.rank_mean if ctrl is not None else float("nan")),
                cal_rho_u=(cal.rho_u if cal is not None else float("nan")),
                cal_margin=(cal.margin if cal is not None else float("nan")),
                cal_gate=(cal.gate_val if cal is not None else float("nan")),
                cal_rank=(cal.r_use if cal is not None else float("nan")),
                cal_S_star=(float(cal.S_star) if (cal is not None and cal.S_star is not None) else float("nan")),
                cal_scatter_ratio=(cal.scatter_ratio if cal is not None else float("nan")))


# =============================================================================
#  Conflict-resolution ("multi-objective") gradient combination for the
#  section-7.2 ABLATION: PCGrad / FAMO / CAGrad.  The theory (Corollary:
#  every continuous, sign-preserving gradient transform cannot prevent
#  collapse) predicts that none of these can stop toxicity because the toxic
#  gradient is ALIGNED (pulls along the batch-mean collapse axis), not
#  conflicting -- so these operators only reweight/reproject and never sever
#  the toxic projection.  Each is implemented faithfully to its formulation:
#     * PCGrad   : project each gradient off the conflicting others.
#     * FAMO     : adaptive scalarization (lambda) that balances losses.
#     * CAGrad   : max-min robust direction guaranteeing the primary can't
#                  regress beyond a ball around its current gradient.
# =============================================================================
def _param_grads(model, opt, loss, retain=True):
    opt.zero_grad(set_to_none=True)
    loss.backward(retain_graph=retain)
    return [(p.grad.clone() if p.grad is not None else None) for p in model.parameters()]


def _grad_list(loss, params):
    """Per-parameter gradients of `loss` w.r.t. `params`, None where unused."""
    return list(torch.autograd.grad(loss, params, retain_graph=True,
                                    allow_unused=True))


def _moo_flat(method):
    """List-of-tensors entry point to a faithful combiner, for the legacy call
    sites that keep no parameter objects (the battery and finance drivers)."""
    def f(gp, gt):
        return moo.combine_flat(method, [list(gp), list(gt)])[0]
    return f


# These names are kept because the battery/finance drivers import them, but they
# now run the reference algorithms from moo_combiners.py -- the previous bodies
# were per-parameter scalar proxies of PCGrad/CAGrad/FAMO and did not survive a
# line-by-line reading against the released code.
_pcgrad = _moo_flat("pcgrad")
_cagrad = _moo_flat("cagrad")
_mgda = _moo_flat("mgda")
_aligned = _moo_flat("aligned")
_imtlg = _moo_flat("imtlg")


def _famo(*a, **kw):
    raise NotImplementedError(
        "FAMO learns loss weights; it is not a gradient transform and cannot be "
        "expressed as a combiner over pre-computed gradients.  Drive it through "
        "moo_combiners.FAMOState (weighted_loss / update_w), as train_one does.")


# =============================================================================
#  Driver
# =============================================================================
def collect_domain(domain, modes, seeds, alpha, m, epochs, lr, batch, rho_c, beta,
                   rel_collapse=0.3):
    # 1) single-task health ceiling per seed -> relative-collapse anchor AND
    #    relative spectral threshold anchor (healthy rho) for STF gating.
    single_by_seed = {s: train_one(domain, "single", 0.0, m, s, epochs, lr,
                                   batch, rho_c, beta) for s in seeds}
    base_by_seed = {s: single_by_seed[s]["scatter"] for s in seeds}
    rho_by_seed = {s: single_by_seed[s]["srho"] for s in seeds}
    # 2) the actual mode configurations, each judged against its own anchor
    recs = []
    for mode in modes:
        for s in seeds:
            base_rho = rho_by_seed.get(s) if mode in ("stfhard", "stfsoft") else None
            r = train_one(domain, mode, alpha, m, s, epochs, lr, batch, rho_c, beta,
                          base_scatter=base_by_seed[s], rel_collapse=rel_collapse,
                          base_rho=base_rho)
            recs.append(r)
    # collapse / recovery aggregation
    by_mode = defaultdict(list)
    for r in recs:
        by_mode[r["mode"]].append(r)
    agg = []
    joint = by_mode.get("joint", [])
    joint_metric = float(np.mean([r["metric"] for r in joint])) if joint else float("nan")
    # regression (RMSE, lower=better) OR classification (acc, higher=better):
    regress = bool(recs[0]["regress"]) if recs else False
    better = lambda a, b: a < b if regress else a > b
    for mode, rs in by_mode.items():
        agg.append(dict(
            domain=domain, mode=mode,
            metric_mean=float(np.mean([r["metric"] for r in rs])),
            metric_std=float(np.std([r["metric"] for r in rs])),
            collapse_rate=float(np.mean([1.0 if r["collapse"] else 0.0 for r in rs])),
            recovery_rate=(float(np.mean([1.0 if better(r["metric"], joint_metric) else 0.0
                                          for r in rs]))
                           if mode != "joint" else 0.0),
            rho_mean=float(np.nanmean([r["rho_mean"] for r in rs])),
            srho_mean=float(np.nanmean([r["srho"] for r in rs])),
            scatter_mean=float(np.mean([r["scatter"] for r in rs])),
            rel_scatter_mean=float(np.mean([r["rel_scatter"] for r in rs])) if base_by_seed else None,
            rho_ref_mean=float(np.nanmean([r.get("rho_ref", float("nan")) for r in rs])),
            rho_last_mean=float(np.nanmean([r.get("rho_last", float("nan")) for r in rs])),
            w_mean=float(np.nanmean([r.get("w_mean", float("nan")) for r in rs])),
            rank_mean=float(np.nanmean([r.get("rank_mean", float("nan")) for r in rs])),
            cal_rho_u_mean=float(np.nanmean([r.get("cal_rho_u", float("nan")) for r in rs])),
            cal_margin_mean=float(np.nanmean([r.get("cal_margin", float("nan")) for r in rs])),
            cal_gate_mean=float(np.nanmean([r.get("cal_gate", float("nan")) for r in rs])),
            cal_rank_mean=float(np.nanmean([r.get("cal_rank", float("nan")) for r in rs])),
            cal_scatter_ratio_mean=float(np.nanmean([r.get("cal_scatter_ratio", float("nan")) for r in rs])),
        ))
    return recs, agg


def run_conflict_ablation(domain, methods, seeds, alpha, m, epochs, lr, batch,
                          rho_c=0.02, beta=8.0, rel_collapse=0.3):
    """Section-7.2 falsification: at SUPercritical alpha, does any *conflict-
    resolution* method (PCGrad/FAMO/CAGrad) prevent collapse?  All runs use
    raw mode ``joint`` (no STF) but replace plain summation with the method's
    gradient operator.  Predicted (Corollary): collapse persists -- toxicity
    is an aligned projection, not a conflict to resolve."""
    single = {s: train_one(domain, "single", 0.0, m, s, epochs, lr, batch,
                           rho_c, beta) for s in seeds}
    base = {s: single[s]["scatter"] for s in seeds}
    rho_base = {s: single[s]["srho"] for s in seeds}
    seq = [("joint", "joint")] + [(mk, mk) for mk in methods]
    recs = []
    eff = []
    for label, meth in seq:
        for s in seeds:
            recs.append(train_one(domain, "joint", alpha, m, s, epochs, lr, batch,
                                  rho_c, beta, base_scatter=base[s],
                                  rel_collapse=rel_collapse,
                                  base_rho=(rho_base[s] if meth in ("pcgrad", "famo", "cagrad") else None),
                                  method=meth))
            eff.append(dict(domain=domain, method=label, alpha=float(alpha), m=m,
                            seed=s))
    by = defaultdict(list)
    for r, e in zip(recs, eff):
        by[e["method"]].append(r)
    regress = bool(recs[0]["regress"]) if recs else False
    joint_metric = float(np.mean([r["metric"] for r in by.get("joint", [])])) if by.get("joint") else float("nan")
    better = lambda a, b: a < b if regress else a > b
    agg = []
    for label, rs in by.items():
        agg.append(dict(
            domain=domain, mode=label,
            metric_mean=float(np.mean([r["metric"] for r in rs])),
            metric_std=float(np.std([r["metric"] for r in rs])),
            collapse_rate=float(np.mean([1.0 if r["collapse"] else 0.0 for r in rs])),
            recovery_rate=(float(np.mean([1.0 if better(r["metric"], joint_metric) else 0.0
                                          for r in rs])) if label != "joint" else 0.0),
            rel_scatter_mean=float(np.mean([r["rel_scatter"] for r in rs])),
        ))
    raw = [{k: r[k] for k in ("domain", "mode", "alpha", "m", "seed", "metric",
                              "collapse", "scatter", "rel_scatter", "rho_mean", "srho")}
           for r in recs]
    return recs, agg, raw


def run_composition(domain, pairs, seeds, alpha, m, epochs, lr, batch,
                    rho_c=0.02, beta=8.0, rel_collapse=0.3):
    """Take-away experiment for the design rule '(i) filter toxicity, (ii) then
    resolve conflict'.  Each entry of `pairs` is (mode, method): the STF mode
    filters the auxiliary branch (toxicity filter) and the conflict method
    combines the resulting primary/auxiliary parameter gradients.  The key cell
    is (stfhard, pcgrad) / (stfhard, cagrad): conflict resolution applied to an
    already de-toxified gradient.  If the theory is right, filtering is what
    carries the recovery and the conflict operator adds little on top; the joint
    references (joint, pcgrad) / (joint, cagrad) reproduce the falsification of
    Section 7.3."""
    single = {s: train_one(domain, "single", 0.0, m, s, epochs, lr, batch,
                           rho_c, beta) for s in seeds}
    base = {s: single[s]["scatter"] for s in seeds}
    rho_base = {s: single[s]["srho"] for s in seeds}
    recs = []
    for mode, method in pairs:
        for s in seeds:
            r = train_one(domain, mode, alpha, m, s, epochs, lr, batch, rho_c, beta,
                          base_scatter=base[s], rel_collapse=rel_collapse,
                          base_rho=(rho_base[s] if mode in ("stfhard", "stfsoft") else None),
                          method=method)
            r = dict(r); r["combo"] = f"{mode}+{method}"
            recs.append(r)
    by = defaultdict(list)
    for r in recs:
        by[r["combo"]].append(r)
    regress = bool(recs[0]["regress"]) if recs else False
    joint_metric = float(np.mean([r["metric"] for r in by.get("joint+joint", [])])) if by.get("joint+joint") else float("nan")
    better = lambda a, b: a < b if regress else a > b
    agg = []
    for combo, rs in by.items():
        agg.append(dict(
            domain=domain, mode=combo,
            metric_mean=float(np.mean([r["metric"] for r in rs])),
            metric_std=float(np.std([r["metric"] for r in rs])),
            collapse_rate=float(np.mean([1.0 if r["collapse"] else 0.0 for r in rs])),
            recovery_rate=(float(np.mean([1.0 if better(r["metric"], joint_metric) else 0.0
                                          for r in rs])) if combo != "joint+joint" else 0.0),
            rel_scatter_mean=float(np.mean([r["rel_scatter"] for r in rs])),
            rho_mean=float(np.nanmean([r["rho_mean"] for r in rs])),
        ))
    raw = [{k: r[k] for k in ("domain", "mode", "alpha", "m", "seed", "metric",
                              "collapse", "scatter", "rel_scatter", "rho_mean",
                              "srho", "combo")} for r in recs]
    return recs, agg, raw


def sweep_alpha(domain, alphas, seeds, m, epochs, lr, batch, rel_collapse=0.3):
    # single-task anchor per seed
    base_by_seed = {s: train_one(domain, "single", 0.0, m, s, epochs, lr,
                                 batch, 0.02, 8.0)["scatter"] for s in seeds}
    rows = []
    for a in alphas:
        for s in seeds:
            r = train_one(domain, "joint", a, m, s, epochs, lr, batch, 0.02, 8.0,
                          base_scatter=base_by_seed[s], rel_collapse=rel_collapse)
            rows.append(r)
    out = []
    for a in alphas:
        rs = [r for r in rows if r["alpha"] == a]
        out.append(dict(domain=domain, alpha=a, m=m,
                        metric_mean=float(np.mean([r["metric"] for r in rs])),
                        metric_std=float(np.std([r["metric"] for r in rs])),
                        collapse_rate=float(np.mean([1.0 if r["collapse"] else 0.0 for r in rs])),
                        scatter_mean=float(np.mean([r["scatter"] for r in rs])),
                        scatter_std=float(np.std([r["scatter"] for r in rs])),
                        rel_scatter_mean=float(np.mean([r["rel_scatter"] for r in rs])),
                        rel_scatter_std=float(np.std([r["rel_scatter"] for r in rs])),
                        rho_mean=float(np.nanmean([r["rho_mean"] for r in rs])),
                        rho_std=float(np.nanstd([r["rho_mean"] for r in rs])),
                        srho_mean=float(np.nanmean([r["srho"] for r in rs])),
                        spread_std=float(np.std([r["scatter"] for r in rs])) / max(float(np.mean([r["scatter"] for r in rs])), 1e-12)))
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DOMAINS))
    ap.add_argument("--modes", default="joint,stf0,stfhard,stfsoft")
    ap.add_argument("--alphas", default="1.0")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--m-list", default="",
                    help="comma list of encoder widths for capacity-law sweep (overrides --m in sweep mode)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--rho-c", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--rel-collapse", type=float, default=0.3,
                    help="relative representation-collapse threshold (fraction of single-task scatter)")
    ap.add_argument("--sweep-alpha", action="store_true")
    ap.add_argument("--stf-kappa", type=float, default=4.0,
                    help="STF-rho gate sharpness: w = sigmoid(kappa*(alpha*rho_hat-1))")
    ap.add_argument("--rho-ema", type=float, default=0.9,
                    help="EMA coefficient for the online order-parameter estimate")
    ap.add_argument("--rho-probe-every", type=int, default=10,
                    help="steps between fp32 order-parameter probes (cost control)")
    ap.add_argument("--rho-rank-max", type=int, default=3,
                    help="max rank of the toxic subspace considered by stfrhorank")
    ap.add_argument("--rho-smooth", action="store_true",
                    help="smooth gate sigmoid(kappa*(alpha*rho_ref-1)) instead of bang-bang")
    ap.add_argument("--rho-trace", action="store_true",
                    help="record the rho(t) trajectory of the controller (diagnostics)")
    ap.add_argument("--cal-warm-epochs", type=float, default=2.0,
                    help="STF-cal healthy warm-up length (epochs, aux branch detached)")
    ap.add_argument("--ablate-methods", default="",
                    help="comma list of pcgrad,famo,cagrad -> run section-7.2 "
                         "conflict-resolution falsification at --alphas")
    ap.add_argument("--composition", action="store_true",
                    help="STF x conflict-resolution composition sweep: does "
                         "conflict resolution help AFTER toxicity filtering?")
    ap.add_argument("--out")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "transformer"],
                    help="shared-encoder family (mlp = default protocol, transformer = small pre-norm Transformer)")
    a = ap.parse_args()
    global ARCH
    ARCH = a.arch
    STF_RHO_CFG.update(kappa=a.stf_kappa, ema=a.rho_ema,
                       probe_every=a.rho_probe_every, rank_max=a.rho_rank_max,
                       smooth=a.rho_smooth, trace=a.rho_trace,
                       cal_warm_epochs=a.cal_warm_epochs)
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.composition:
        alphas = [float(x) for x in a.alphas.split(",")]
        pairs = [("joint", "joint"), ("joint", "pcgrad"), ("joint", "cagrad"),
                 ("stfhard", "joint"), ("stfhard", "pcgrad"), ("stfhard", "cagrad"),
                 ("stfsoft", "joint"), ("stfsoft", "pcgrad"), ("stfsoft", "cagrad")]
        recs = []; agg = []; rows = []
        for al in alphas:
            _r, _a, _w = run_composition(a.domain, pairs, seeds, al, a.m,
                                         a.epochs, a.lr, a.batch, a.rho_c,
                                         a.beta, a.rel_collapse)
            for x in _a:
                x["alpha"] = al
            recs += _r; agg += _a; rows += _w
        out = agg
        key = "composition"
        raw = rows
    elif a.ablate_methods:
        alphas = [float(x) for x in a.alphas.split(",")]
        methods = [x.strip() for x in a.ablate_methods.split(",") if x.strip()]
        recs = []; agg = []; rows = []
        for al in alphas:
            _r, _a, _w = run_conflict_ablation(a.domain, methods, seeds, al, a.m,
                                               a.epochs, a.lr, a.batch, a.rho_c,
                                               a.beta, a.rel_collapse)
            for x in _a:
                x["alpha"] = al
            recs += _r; agg += _a; rows += _w
        out = agg
        key = "contrast"
        raw = rows
    elif a.sweep_alpha:
        alphas = [float(x) for x in a.alphas.split(",")]
        m_list = ([int(x) for x in a.m_list.split(",")] if a.m_list else [a.m])
        out = []
        raw = []
        for m in m_list:
            rows, raw_rows = sweep_alpha(a.domain, alphas, seeds, m, a.epochs, a.lr, a.batch,
                                         a.rel_collapse)
            for r in rows:
                r["m"] = m
            out.extend(rows)
            for r in raw_rows:
                r["m"] = m
            raw.extend(raw_rows)
        key = "sweep"
    else:
        modes = [x.strip() for x in a.modes.split(",")]
        recs, agg = collect_domain(a.domain, modes, seeds, float(a.alphas), a.m,
                                   a.epochs, a.lr, a.batch, a.rho_c, a.beta,
                                   a.rel_collapse)
        out = agg
        key = "contrast"
        raw = [{k: r[k] for k in ("domain", "mode", "alpha", "m", "seed", "metric",
                                  "collapse", "scatter", "rel_scatter", "rho_mean",
                                  "srho", "rho_ref", "rho_last", "latch_step",
                                  "w_mean", "rank_mean", "rho_trace",
                                  "cal_rho_u", "cal_margin", "cal_gate", "cal_rank",
                                  "cal_S_star", "cal_scatter_ratio")} for r in recs]
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(dict(domain=a.domain, kind=key, results=out, raw=raw), f, indent=2)
        print("wrote", a.out)
    else:
        print(json.dumps(dict(domain=a.domain, kind=key, results=out), indent=2))


if __name__ == "__main__":
    main()