"""Fixed-projection aux bridge (A->B): does the toy mechanism collapse real data?

Adds the toy testbed's non-learnable random-projection variance head
(L_aux = Var_b(W h)) to the full-market StudentModel and sweeps its weight.
If sigma_h collapses as fixed_aux grows, the same phase transition occurs on
real data, bridging the toy testbed (A) to the full-market pipeline (B).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_repro

FIXED_AUX = [0.0, 0.01, 0.1, 1.0, 10.0]
FOLD = 0
EPOCHS = 20
PATIENCE = 6
BATCH = 512
DEVICE = "cpu"


def main():
    results = []
    for w in FIXED_AUX:
        r = wf_repro.train_fold(FOLD, False, EPOCHS, BATCH, DEVICE, 1e-3,
                                PATIENCE, 0, aux_scale=1.0, fixed_aux_weight=w)
        r["fixed_aux"] = w
        results.append(r)
        print(f"[bridge] fixed_aux={w:6.2f} oos_ic={r['oos_ic']:+.4f} "
              f"sigma_h={r['sigma_h']:.4f}", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "wf_fixed_aux_bridge.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("[saved] wf_fixed_aux_bridge.json")


if __name__ == "__main__":
    main()
