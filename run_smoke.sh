#!/usr/bin/env bash
# End-to-end smoke test for the released code.
#
# One unified-protocol cell (HAR, m=32, eight epochs, one seed) at the smallest
# settings that still exercise every path the paper's tables use: the
# variance-shrinkage toxifier, the three spectral-filter variants, and the
# calibrated gate.  It prints each mode's collapse rate and relative scatter so
# the run can be compared against the shipped reference log
# logs/smoke/run_smoke.log.
#
#   bash run_smoke.sh                  # CPU (a GPU is used automatically if present)
#   PY=/path/to/python bash run_smoke.sh
#   OUT=/tmp/mine bash run_smoke.sh    # where the cell JSON is written
#
# The cell is deliberately tiny -- about six seconds on CPU -- but it is long
# enough for the healthy single-task anchor to settle, which the STF-hard and
# STF-soft thresholds are anchored to.  The point is that the pipeline runs end
# to end and the directions come out right -- joint collapses, the STF variants
# do not -- not that eight-epoch numbers match the paper's sixty-epoch ones.
# For the numbers behind a table, replay its `configs/jobs_*.json` entry with
# `code/runner.py`; `results/README.md` maps every result file to its table.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-$(command -v python3 || command -v python || echo python)}
OUT=${OUT:-/tmp/repfix_smoke}
mkdir -p "$OUT"

"$PY" code/cross_domain_mtl.py --domain har --arch mlp \
    --modes joint,stf0,stfhard,stfsoft,stfcal \
    --alphas 100.0 --seeds 0 --m 32 --epochs 8 --cal-warm-epochs 1 \
    --batch 256 --out "$OUT/smoke_har.json"

"$PY" - "$OUT/smoke_har.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
rows = d["results"] if isinstance(d, dict) and "results" in d else d
print()
print("%-10s %14s %16s %10s" % ("mode", "collapse_rate", "rel_scatter", "accuracy"))
for r in rows:
    print("%-10s %14s %16.4f %10.4f" % (
        r.get("mode"), r.get("collapse_rate"),
        r.get("rel_scatter_mean", float("nan")), r.get("metric_mean", float("nan"))))
print()
print("expected shape: joint collapse_rate 1.0 with rel_scatter near 0;")
print("stf0/stfhard/stfsoft/stfcal 0.0 with rel_scatter near 1.")
PYEOF
