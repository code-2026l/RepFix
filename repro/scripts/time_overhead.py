"""Wall-clock overhead of the STF family and of the gradient-surgery baselines.

Measures the *shipped* training path: every row is a call to
``cross_domain_mtl.train_one`` with its ``timing`` hook enabled, so the numbers
describe the same loop that produces the paper's results rather than a
re-implementation of it.

Why the measurement is designed the way it is
---------------------------------------------
A single A/B pair on a shared GPU is not evidence: throughput drifts with
whatever else is resident, and a one-shot ratio silently absorbs that drift.
Two choices defend against it.

* **Round-robin, not blocked.** Rows are visited rep-by-rep
  (``joint, stf0, ..., pcgrad`` then again), so a slowly varying load is spread
  across every row instead of landing on whichever row ran last.
* **Rotation, not a fixed order** (``--rotate``).  Round-robin alone still gives
  every row the *same* position inside the repetition: ``famo`` is always last
  and ``joint`` always first, so any linear drift or warm-up trend is charged to
  the method rather than to the clock.  ``--rotate`` advances the start of the
  order by one each repetition (a Latin square over the fixed ``ROWS``), so over
  ``reps`` repetitions each row visits each position once; the position is
  recorded per run so the residual spread can be checked.  Reported tables use
  it.
* **Epoch 0 is dropped.** The first epoch carries one-off costs (lazy CUDA
  module loading, cuDNN autotuning, allocator growth) that are identical for
  every row and would otherwise be charged to the row that happens to run
  first. The reported figure is the median over epochs ``1..N-1``.
* **Device-complete timing.** ``torch.cuda.synchronize()`` brackets each epoch,
  so the figure is wall time, not kernel-enqueue time -- without it, a row that
  enqueues more small kernels looks *faster* than it is.
* **Contention is recorded, not assumed away.** GPU utilisation is sampled
  before and after every row and written to the JSON. ``--wait-idle`` blocks
  until the device is quiet, which is what a claim of the form "under X%"
  requires; without it the run is still valid as a *ratio*, but the JSON says
  what the device was doing.

Usage
-----
  python scripts/time_overhead.py --domain har --m 64 --epochs 10 --reps 3 \
      --wait-idle --out results/TIME/overhead_har.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import cross_domain_mtl as cdm  # noqa: E402

# (mode, method) -- mode selects the STF variant, method the gradient combiner.
# "joint"/"joint" is the unfiltered plain-sum reference every ratio is against.
ROWS = [
    ("joint", "joint"),
    ("stf0", "joint"),
    ("stfhard", "joint"),
    ("stfsoft", "joint"),
    ("stfcal", "joint"),
    ("joint", "pcgrad"),
    ("joint", "cagrad"),
    ("joint", "famo"),
]


def gpu_snapshot():
    """(utilisation %, memory used MiB) or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout.strip()
        util, mem = out.splitlines()[0].split(",")
        return {"util_pct": int(util), "mem_used_mib": int(mem)}
    except Exception:
        return None


def wait_idle(max_util=5, need=3, timeout_s=5400, poll_s=30):
    """Block until the GPU reads <= max_util% for `need` consecutive samples."""
    t0 = time.time()
    ok = 0
    while time.time() - t0 < timeout_s:
        s = gpu_snapshot()
        if s is None:
            print("[wait-idle] nvidia-smi unavailable; proceeding", flush=True)
            return
        if s["util_pct"] <= max_util:
            ok += 1
            print(f"[wait-idle] {ok}/{need} quiet samples "
                  f"(util={s['util_pct']}%, mem={s['mem_used_mib']}MiB)", flush=True)
            if ok >= need:
                return
        else:
            ok = 0
            print(f"[wait-idle] busy (util={s['util_pct']}%, "
                  f"mem={s['mem_used_mib']}MiB); waiting", flush=True)
        time.sleep(poll_s)
    print("[wait-idle] timed out; proceeding under contention", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="har")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10,
                    help="epochs per row per rep (epoch 0 is dropped as warm-up)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho-c", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--wait-idle", action="store_true")
    ap.add_argument("--max-util", type=int, default=5)
    ap.add_argument("--rows", default="",
                    help="optional comma-separated subset, e.g. 'joint+joint,"
                         "stfhard+joint' (default: all rows)")
    ap.add_argument("--rotate", action="store_true",
                    help="rotate the row order by one position each repetition "
                         "(a Latin square over the fixed ROWS order).  Without "
                         "this, every row is always measured at the same position "
                         "inside the repetition, so a warm-up or clock-drift "
                         "trend is confounded with the method -- the row that is "
                         "always last absorbs it.  Use it for any reported table.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = ROWS
    if a.rows:
        want = {s.strip() for s in a.rows.split(",") if s.strip()}
        rows = [r for r in ROWS if f"{r[0]}+{r[1]}" in want]
        missing = want - {f"{m}+{k}" for m, k in rows}
        if missing:
            raise SystemExit(f"unknown --rows entries: {sorted(missing)}")
    if not any(f"{m}+{k}" == "joint+joint" for m, k in rows):
        print("[warn] 'joint+joint' is not in --rows; ratios are still computed "
              "against it but it will not be re-measured here", flush=True)

    if a.epochs < 2:
        raise SystemExit("--epochs must be >= 2 (epoch 0 is the dropped warm-up)")

    if a.wait_idle:
        wait_idle(max_util=a.max_util)

    # Warm the domain cache and the CUDA context once, outside every timed row,
    # so no row pays for the first-touch of the data or the allocator.
    print("[warmup] loading domain + building CUDA context", flush=True)
    cdm.load_domain(a.domain)
    _ = torch.zeros(8, device="cuda") if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    per_row = {f"{mo}+{me}": [] for mo, me in rows}
    env_before = gpu_snapshot()

    for rep in range(a.reps):
        # Latin-square rotation: with `--rotate`, the row that is measured first
        # advances by one each repetition, so over `reps` repetitions every row
        # visits every position.  Positions are recorded so the spread across
        # positions can be checked afterwards.
        shift = rep % len(rows) if a.rotate else 0
        order = rows[shift:] + rows[:shift]
        for pos, (mode, method) in enumerate(order):
            key = f"{mode}+{method}"
            snap0 = gpu_snapshot()
            t = []
            t_wall0 = time.perf_counter()
            cdm.train_one(a.domain, mode, a.alpha, a.m, a.seed, a.epochs,
                          a.lr, a.batch, a.rho_c, a.beta,
                          base_scatter=None, rel_collapse=0.3, base_rho=None,
                          method=method, timing=t)
            wall = time.perf_counter() - t_wall0
            snap1 = gpu_snapshot()
            body = t[1:] if len(t) > 1 else t
            per_row[key].append({
                "rep": rep,
                "pos": pos,
                "median_epoch_s": statistics.median(body),
                "all_epoch_s": t,
                "run_wall_s": wall,
                "gpu_before": snap0,
                "gpu_after": snap1,
            })
            print(f"[rep {rep}] {key:<16} median/epoch = "
                  f"{statistics.median(body):.4f}s  (run {wall:.1f}s, "
                  f"gpu {snap0['util_pct'] if snap0 else '?'}% -> "
                  f"{snap1['util_pct'] if snap1 else '?'}%)", flush=True)

    env_after = gpu_snapshot()

    # Aggregate: median over reps of the per-rep median epoch time.
    summary = {}
    for key, runs in per_row.items():
        vals = [r["median_epoch_s"] for r in runs]
        summary[key] = {
            "median_epoch_s": statistics.median(vals),
            "per_rep_median_epoch_s": vals,
            "n_reps": len(vals),
        }
    base = summary["joint+joint"]["median_epoch_s"]
    for key, s in summary.items():
        s["overhead_pct_vs_joint"] = 100.0 * (s["median_epoch_s"] / base - 1.0)
        s["ratio_vs_joint"] = s["median_epoch_s"] / base

    out = {
        "domain": a.domain, "m": a.m, "epochs_per_row": a.epochs, "reps": a.reps,
        "alpha": a.alpha, "batch": a.batch, "seed": a.seed,
        "rotate": bool(a.rotate),
        "rows_measured": [f"{m}+{k}" for m, k in rows],
        "epoch0_dropped_as_warmup": True,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_before_run": env_before, "gpu_after_run": env_after,
        "rows": summary,
        "raw": per_row,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n{:<16} {:>12} {:>12} {:>10}".format(
        "row", "median s/ep", "ratio", "overhead"))
    for key in per_row:
        s = summary[key]
        print("{:<16} {:>12.4f} {:>12.3f} {:>9.1f}%".format(
            key, s["median_epoch_s"], s["ratio_vs_joint"],
            s["overhead_pct_vs_joint"]))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
