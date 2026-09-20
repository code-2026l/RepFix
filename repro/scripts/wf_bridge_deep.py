"""Deepened fixed-projection bridge (A->B): drive full-market sigma_h to collapse.

The toy testbed collapses because the fixed random-projection head is the only
aux, so Var_b(W h)->0 forces h->const.  On real data the primary signal (the
c2||g_p||Psi restoring term of Thm) resists collapse, so we raise the toxicity:
  * k: more projection rows (more variance-contraction directions),
      from k=2 up to k=64 ~= d_model
  * w: larger fixed_aux weight so the variance term dominates the primary loss.

We watch sigma_h (h.std over the batch) collapse toward 0 AND oos_IC toward 0.
This is the direct "complete collapse on real data" evidence.
"""
import json
import os
import sys
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_repro

FOLD = 0
EPOCHS = 25
PATIENCE = 8
BATCH = 512
DEVICE = "cpu"

# (k, w) grid: widen the toxic subspace and push its weight to extremes
GRID = list(itertools.product([2, 16, 64], [1.0, 10.0, 100.0, 1000.0]))


def main():
    results = []
    for k, w in GRID:
        r = wf_repro.train_fold(FOLD, False, EPOCHS, BATCH, DEVICE, 1e-3,
                                PATIENCE, 0, aux_scale=1.0,
                                fixed_aux_weight=w, fixed_aux_k=k)
        r["fixed_k"], r["fixed_w"] = k, w
        results.append(r)
        print(f"[bridge-deep] k={k:2d} w={w:6.0f} -> sigma_h={r['sigma_h']:.4f} "
              f"oos_ic={r['oos_ic']:+.4f}", flush=True)
    # a healthy anchor (no toxicity) for contrast
    r0 = wf_repro.train_fold(FOLD, False, EPOCHS, BATCH, DEVICE, 1e-3,
                             PATIENCE, 0, aux_scale=1.0,
                             fixed_aux_weight=0.0, fixed_aux_k=2)
    r0["fixed_k"], r0["fixed_w"] = 0, 0.0
    results.append(r0)
    print(f"[bridge-deep] healthy anchor sigma_h={r0['sigma_h']:.4f} "
          f"oos_ic={r0['oos_ic']:+.4f}", flush=True)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "wf_bridge_deep.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("[saved] wf_bridge_deep.json")


if __name__ == "__main__":
    main()