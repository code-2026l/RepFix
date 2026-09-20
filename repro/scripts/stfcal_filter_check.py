"""What each STF-cal filter mode actually does to the auxiliary-branch embedding.

Section stfcal describes the filter as F = I - P_{S_r}, the orthogonal projector
onto the complement of the r leading principal directions of the centered
representation, latched for the rest of training.  Two filter modes are shipped:

  mode 'cal'     : per-sample direction u = hc/||hc||, returns z - <hc,u> u
  mode 'calrank' : fixed subspace U[:, :r], returns z - (hc U) U^T

This script evaluates both on the same input and compares them with the batch
mean, so the two modes can be told apart from the code that produced the
released numbers rather than from the paper's description of them.

Run from the repository root:

    python repro/scripts/stfcal_filter_check.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "code"))

from battery_bmtl_v3 import BatteryCal  # noqa: E402


def main():
    torch.manual_seed(0)
    B, d, r = 64, 16, 3
    z = torch.randn(B, d)

    cal = BatteryCal(alpha_aux=10.0, warm_epochs=1, rank_max=r, basis="pca")
    cal.U = torch.linalg.qr(torch.randn(d, r))[0]
    cal.rank = r
    cal.gate = 1.0

    hc = z - z.mean(0, keepdim=True)
    mean_rep = z.mean(0, keepdim=True).expand_as(z)

    out_cal = cal.filter_z(z, "cal")
    out_rank = cal.filter_z(z, "calrank")

    def rel(a, b):
        return float((a - b).norm() / b.norm().clamp_min(1e-12))

    print("B=%d d=%d r=%d" % (B, d, r))
    print()
    print("mode 'cal'     : rel. distance to the batch-mean representation "
          "= %.3e" % rel(out_cal, mean_rep))
    print("mode 'cal'     : rel. distance to the rank-r filter       "
          "= %.3e" % rel(out_cal, out_rank))
    print("mode 'calrank' : rel. distance to the batch-mean representation "
          "= %.3e" % rel(out_rank, mean_rep))
    print()
    print("cross-sample scatter of the filtered representation "
          "(0 == constant):")
    for name, o in (("unfiltered", z), ("'cal'", out_cal),
                    ("'calrank'", out_rank)):
        print("  %-12s %.6f" % (name, float(o.std(dim=0).mean())))
    print()
    print("A 'cal' distance to the batch mean of ~0 means mode 'cal' does not")
    print("delete r directions -- it replaces the representation by a constant.")


if __name__ == "__main__":
    main()
