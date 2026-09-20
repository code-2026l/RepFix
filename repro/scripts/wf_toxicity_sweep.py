"""Full-market toxicity sweep (B extension): vary aux_scale on fold 0, measure sigma_h.

Bridges the toy-testbed phase transition (A) to the full-market pipeline (B):
if sigma_h collapses as aux_scale grows, the same order-parameter transition
occurs on real data.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_repro

SCALES = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0]
FOLD = 0
EPOCHS = 20
PATIENCE = 6
BATCH = 512
DEVICE = "cpu"


def main():
    results = []
    for s in SCALES:
        r = wf_repro.train_fold(FOLD, False, EPOCHS, BATCH, DEVICE, 1e-3,
                                PATIENCE, 0, aux_scale=s)
        r["aux_scale"] = s
        results.append(r)
        print(f"[sweep] aux_scale={s:6.1f} oos_ic={r['oos_ic']:+.4f} "
              f"sigma_h={r['sigma_h']:.4f}", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "wf_toxicity_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("[saved] wf_toxicity_sweep.json")


if __name__ == "__main__":
    main()
