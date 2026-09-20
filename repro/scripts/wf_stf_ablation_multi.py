"""Multi-seed STF ablation with mean +/- std error bars for the paper table.

Re-runs the single-seed grid (wf_stf_ablation.py) over N_SEEDS independent seeds
and aggregates val-IC / oos-IC / aux / collapse per (toxicity, STF-mode) cell.
The primary task uses the pairwise rank loss (prediction-level trap) plus the
variance-shrinking fixed-projection aux head (encoder collapse). The grid:

    {rank_only, rank_w1_k8, rank_w5_k16, rank_w20_k64, rank_w100_k64}
  x {off, stf0, hard, soft}  x  {seed_0 .. seed_{N_SEEDS-1}}

To parallelize across CPU cores (32-core HPC node), each (config, seed) cell is
submitted as an independent worker and OMP/threadpools are pinned to 1 thread.

Usage:
  OMP_NUM_THREADS=1 python wf_stf_ablation_multi.py --seeds 5 --workers 16 \
      --out ../results/wf_stf_ablation_multiseed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_repro

FOLD = 0
EPOCHS = 30
PATIENCE = 8
BATCH = 256            # rank loss is O(B^2); keep B modest
RANK_BETA = 1e-3

TOXIC_CONFIGS = [
    ("rank_only",        0.0,   2),
    ("rank_w1_k8",       1.0,   8),
    ("rank_w5_k16",      5.0,  16),
    ("rank_w20_k64",    20.0,  64),
    ("rank_w100_k64",  100.0,  64),
]

STF_MODES = [
    ("off",  False, None),       # no shielding -> collapse when toxic
    ("stf0", True,  "off"),      # full detach of aux gradients (Detach-Aux-Heads)
    ("hard", False, "hard"),     # threshold-triggered subspace removal
    ("soft", False, "soft"),     # smooth sigmoid gate
]

DEVICE = "cpu"


def _worker(args):
    name, w, k, label, det, stf, seed = args
    r = wf_repro.train_fold(FOLD, det, EPOCHS, BATCH, DEVICE, 1e-3,
                            PATIENCE, seed, aux_scale=1.0,
                            fixed_aux_weight=w, fixed_aux_k=k,
                            rank_loss=True, rank_beta=RANK_BETA, stf=stf)
    r.update({"toxic": name, "fixed_w": w, "fixed_k": k,
              "stf_mode": label, "seed": seed})
    print(f"[stf] {name:14s} x {label:5s} s{seed} -> val_ic={r['val_ic']:+.4f} "
          f"oos_ic={r['oos_ic']:+.4f} aux_val={r['aux_val']:.4f} "
          f"sigma_h={r['sigma_h']:.4f}", flush=True)
    return r


def _agg(rows, key):
    vals = [r[key] for r in rows]
    m = float(sum(vals) / len(vals))
    sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    return m, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results",
        "wf_stf_ablation_multiseed.json"))
    args = ap.parse_args()

    tasks = []
    for name, w, k in TOXIC_CONFIGS:
        for label, det, stf in STF_MODES:
            for s in range(args.seeds):
                tasks.append((name, w, k, label, det, stf, s))

    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        results = pool.map(_worker, tasks)
    print(f"[done] {len(results)} runs in {time.time() - t0:.0f}s", flush=True)

    # aggregate: cell = (toxic, stf_mode) -> mean/std over seeds
    cells = {}
    for r in results:
        cells.setdefault((r["toxic"], r["stf_mode"]), []).append(r)

    summary = []
    for (name, mode), rows in cells.items():
        summary.append({
            "toxic": name, "stf_mode": mode, "n": len(rows),
            "val_ic_mean": _agg(rows, "val_ic")[0], "val_ic_std": _agg(rows, "val_ic")[1],
            "oos_ic_mean": _agg(rows, "oos_ic")[0], "oos_ic_std": _agg(rows, "oos_ic")[1],
            "aux_val_mean": _agg(rows, "aux_val")[0], "aux_val_std": _agg(rows, "aux_val")[1],
            "collapse_mean": _agg(rows, "collapse")[0], "collapse_std": _agg(rows, "collapse")[1],
            "sigma_h_mean": _agg(rows, "sigma_h")[0], "sigma_h_std": _agg(rows, "sigma_h")[1],
        })
    summary.sort(key=lambda c: (c["toxic"], ["off", "stf0", "hard", "soft"].index(c["stf_mode"])))

    with open(args.out, "w") as f:
        json.dump({"seeds": args.seeds, "cells": summary,
                   "runs": results}, f, indent=2)
    print(f"[saved] {args.out}", flush=True)

    # human-readable CSV
    csv_path = args.out.replace(".json", ".csv")
    with open(csv_path, "w") as f:
        f.write("toxic,stf_mode,n,val_ic_mean,val_ic_std,oos_ic_mean,oos_ic_std,"
                "aux_val_mean,aux_val_std,collapse_mean,collapse_std\n")
        for c in summary:
            f.write(f"{c['toxic']},{c['stf_mode']},{c['n']},"
                    f"{c['val_ic_mean']:.6f},{c['val_ic_std']:.6f},"
                    f"{c['oos_ic_mean']:.6f},{c['oos_ic_std']:.6f},"
                    f"{c['aux_val_mean']:.6f},{c['aux_val_std']:.6f},"
                    f"{c['collapse_mean']:.6f},{c['collapse_std']:.6f}\n")
    print(f"[saved] {csv_path}", flush=True)


if __name__ == "__main__":
    main()