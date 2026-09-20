"""Full collapse-on-real-data, v2: correct cross-sample order parameter.

v1 showed raw h.std() stays ~1 (LayerNorm masks it). Here we measure the
proper order parameters captured by the new train_fold return:
  - sigma_h    : cross-sample scatter magnitude (h centered over the batch)
                 -> 0 iff all test samples map to (near) the same vector
  - collapse   : mean over features of cross-sample std (per-feature scatter)
  - fixed_var  : variance of the fixed-projection outputs; the L_aux objective
                 itself -> 0 iff the encoder collapses the toxic directions

We also shrink d_model to 16 (less expressive room for the restoring term) and
sweep the toxicity weight over a wider, finer ladder, all with the rank loss
(prediction-level trap) enabled, to force the real-data collapse.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_repro

CONFIGS = [
    # w, k  (rank loss on for all)
    (0.0,  2),
    (1.0,  8),
    (5.0, 16),
    (20.0, 64),
    (100., 64),
]

FOLD = 0
EPOCHS = 30
PATIENCE = 10
BATCH = 256
DEVICE = "cpu"


def main():
    results = []
    for w, k in CONFIGS:
        r = wf_repro.train_fold(FOLD, False, EPOCHS, BATCH, DEVICE, 1e-3,
                                PATIENCE, 0, aux_scale=1.0,
                                fixed_aux_weight=w, fixed_aux_k=k,
                                rank_loss=True, rank_beta=1e-3)
        r["fixed_w"], r["fixed_k"] = w, k
        results.append(r)
        print(f"[collapse-v2] w={w:6.1f} k={k:2d} -> sigma_h={r['sigma_h']:.4f} "
              f"collapse={r['collapse']:.4f} fixed_var={r['fixed_var']:.4f} "
              f"oos_ic={r['oos_ic']:+.4f}", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "wf_collapse_v2.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("[saved] wf_collapse_v2.json")


if __name__ == "__main__":
    main()