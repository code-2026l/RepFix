"""Controlled test of the two bases used by the STF-cal rank rule.

Section stfcal deletes the top-r principal directions of the centered
representation.  Theorem minimalrank is stated for the toxic operator T, and the
two coincide only when the auxiliary Hessian is diagonal in the representation's
principal basis -- the variance-shrinkage toxifier, whose contraction is
isotropic on H_c.  For a *production* head the toxic directions are set by that
head's own Hessian and need not align with the representation's principal axes,
so the surrogate can leave residual toxic flux.

This script isolates that question with known ground truth.  It builds a
synthetic (g_p, g_a, h) triple whose toxic subspace S is known exactly and is
deliberately placed in the LOW-variance part of the representation, then asks
each basis how well it recovers S:

  basis='pca' : top-r principal directions of the centered representation
                (the paper's surrogate)
  basis='gen' : top-r generalized eigenvectors of (Sigma_a, Sigma_p), i.e. the
                directions of maximal toxic-to-restoring energy ratio

Reported per basis: the principal angles between the returned span and S, and
the rank the calibrated rule would pick.  Run from the repository root:

    python repro/scripts/stfcalg_basis_test.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "code"))

from battery_bmtl_v3 import _toxic_basis  # noqa: E402


def principal_angles(U, V):
    """Largest principal angle (radians) between two subspaces, and the mean of
    the cosines.  Zero angle == identical span."""
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    s = np.linalg.svd(Qu.T @ Qv, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(np.arccos(s).max()), float(s.mean())


def build(d=64, B=512, k=5, seed=0, toxic_ratio=1.0):
    """Return (hc, ga, gp, S) with S the exact toxic subspace.

    The representation carries most of its variance on the FIRST d/2 coordinates;
    S is drawn from the trailing half, so the representation-PCA basis has almost
    no overlap with the toxic directions."""
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    S = Q[:, d - k:]                       # toxic directions: the LOW-variance half
    H_basis = Q[:, : d // 2]               # representation's high-variance half

    hc = (torch.randn(B, d // 2, generator=g) @ (H_basis * 3.0).t()) * 0.3
    gp = torch.randn(B, d, generator=g)                 # isotropic restoring gradient

    coef = torch.randn(B, k, generator=g) * toxic_ratio
    ga = coef @ S.t() + 0.05 * torch.randn(B, d, generator=g)
    return hc, ga, gp, S


def main():
    out = []
    for seed in range(5):
        hc, ga, gp, S = build(seed=seed)
        Sn = S.numpy()
        rec = {"seed": seed}
        for basis in ("pca", "aux", "gen"):
            U, lam = _toxic_basis(hc, ga, gp, R=S.shape[1], basis=basis)
            Un = U.numpy()
            max_ang, mean_cos = principal_angles(Un, Sn)
            # trace(P_U P_S) = sum of squared principal-angle cosines = the
            # fraction of S's energy the returned span captures (1.0 == identical)
            cap = float(np.trace((Un @ Un.T) @ (Sn @ Sn.T)) / S.shape[1])
            rec[basis] = {
                "max_principal_angle_deg": round(float(np.degrees(max_ang)), 2),
                "mean_cos": round(mean_cos, 4),
                "subspace_energy_captured": round(cap, 4),
            }
            if basis == "gen":
                rec[basis]["lam_top"] = [round(float(x), 4)
                                         for x in lam[:S.shape[1] + 2]]
        out.append(rec)
        print(json.dumps(rec))

    # summary
    print()
    for basis in ("pca", "aux", "gen"):
        ang = np.mean([r[basis]["max_principal_angle_deg"] for r in out])
        cap = np.mean([r[basis]["subspace_energy_captured"] for r in out])
        print("%-4s  mean max principal angle = %6.2f deg   "
              "mean energy of S captured = %.3f" % (basis, ang, cap))
    print()
    print("pca angle 90 deg / energy ~0 means the surrogate misses S entirely;")
    print("gen angle ~0 deg / energy ~1 means the generalized basis recovers it.")


if __name__ == "__main__":
    main()
