"""Full collapse-on-real-data bridge (rank trap + fixed projection).

The toy testbed collapses because TWO traps align:
  (L2 prediction trap)  rank loss is zero at a constant output
  (L1 encoder collapse) fixed random-projection head shrinks Var_b(W h)
On real data the primary MSE loss resists collapse (non-zero restoring
gradient at a constant), so we switch the primary head to the SAME pairwise
rank loss as the testbed, restoring the prediction-level trap, and combine it
with the fixed projection head. We sweep the toxicity weight and watch:
  - sigma_h  (encoder output std on test)  -> should -> 0 on collapse
  - oos_ic   (rank IC)                     -> should -> 0 on collapse
  - report train_loss trajectory to see the constant-output fixed point
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_repro

FOLD = 0
EPOCHS = 30
PATIENCE = 10
BATCH = 256        # rank loss is O(B^2); keep B modest
DEVICE = "cpu"
RANK_KIND = "quad"

# (fixed_aux_w, fixed_aux_k): push the toxic subspace harder than before.
# rank_loss=True on ALL runs so the prediction-level trap is present.
CONFIGS = [
    # name, fixed_w, fixed_k, rank_loss, beta
    ("rank_only",            0.0,    2,  True,  1e-3),
    ("rank+fixed_w1_k8",     1.0,    8,  True,  1e-3),
    ("rank+fixed_w5_k16",    5.0,   16,  True,  1e-3),
    ("rank+fixed_w20_k64",  20.0,   64,  True,  1e-3),
    ("rank+fixed_w100_k64",100.0,   64,  True,  1e-3),
]


def main():
    results = []
    for name, w, k, rank, beta in CONFIGS:
        r = wf_repro.train_fold(FOLD, False, EPOCHS, BATCH, DEVICE, 1e-3,
                                PATIENCE, 0, aux_scale=1.0,
                                fixed_aux_weight=w, fixed_aux_k=k,
                                rank_loss=rank, rank_beta=beta)
        r.update({"config": name, "rank_loss": rank, "rank_beta": beta})
        results.append(r)
        print(f"[collapse] {name:22s} -> sigma_h={r['sigma_h']:.4f} "
              f"oos_ic={r['oos_ic']:+.4f} val_ic={r['val_ic']:+.4f}", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "wf_collapse_rank_trap.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("[saved] wf_collapse_rank_trap.json")


if __name__ == "__main__":
    main()