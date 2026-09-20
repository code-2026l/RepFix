"""STF ablation: does spectral toxicity filtering recover the primary IC
under auxiliary-toxicity collapse, and does it preserve auxiliary-task fit
better than the blunt full-detach (STF-0)?

Grid: {toxicity config} x {off / stf0 / hard / soft}.

  Toxicity is injected by the fixed random-projection head (variance-shrinking
  aux gradient) on top of the pairwise rank loss (prediction-level trap), which
  together collapse the shared encoder when nothing shields it ('off').

  Metric split:
    val_ic / oos_ic  -> primary task quality (must recover under STF)
    aux_val          -> aux-task fit (soft/hard should keep it close to 'off',
                        while STF-0 always sacrifices it)
    sigma_h / collapse -> encoder scatter (order parameter; near-zero = collapsed)

Usage:
  REPFIX_DATA_DIR=<data dir with wf_fold0.npz> python wf_stf_ablation.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_repro

FOLD = 0
EPOCHS = 30
PATIENCE = 8
BATCH = 256            # rank loss is O(B^2); keep B modest
RANK_BETA = 1e-3

# toxicity configs (fixed_proj_w, fixed_proj_k); rank_loss=True on ALL runs so the
# prediction-level trap is present, and the fixed-projection head pushes the
# encoder variance toward zero (the sole driver of collapse in this testbed).
TOXIC_CONFIGS = [
    ("rank_only",        0.0,   2),
    ("rank_w1_k8",       1.0,   8),
    ("rank_w5_k16",      5.0,  16),
    ("rank_w20_k64",    20.0,  64),
    ("rank_w100_k64",  100.0,  64),
]

# (label, detach_flag, stf_mode)
STF_MODES = [
    ("off",  False, None),       # no shielding -> collapse when toxic
    ("stf0", True,  "off"),      # full detach of aux gradients (Detach-Aux-Heads)
    ("hard", False, "hard"),     # threshold-triggered subspace removal
    ("soft", False, "soft"),     # smooth sigmoid gate
]


def torch_is_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


DEVICE = "cuda" if torch_is_cuda() else "cpu"


def main():
    t0 = time.time()
    results = []
    for name, w, k in TOXIC_CONFIGS:
        for label, det, stf in STF_MODES:
            r = wf_repro.train_fold(FOLD, det, EPOCHS, BATCH, DEVICE, 1e-3,
                                    PATIENCE, 0, aux_scale=1.0,
                                    fixed_aux_weight=w, fixed_aux_k=k,
                                    rank_loss=True, rank_beta=RANK_BETA,
                                    stf=stf)
            r.update({"toxic": name, "fixed_w": w, "fixed_k": k, "stf_mode": label})
            results.append(r)
            print(f"[stf] {name:14s} x {label:5s} -> val_ic={r['val_ic']:+.4f} "
                  f"oos_ic={r['oos_ic']:+.4f} sigma_h={r['sigma_h']:.4f} "
                  f"aux_val={r['aux_val']:.4f}", flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "wf_stf_ablation.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()