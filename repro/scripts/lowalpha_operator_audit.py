"""Small-dose audit of the ten conflict-resolution operators (CPU).

The main paper compares the operators at each domain's supercritical dose.
This driver asks the complementary question raised in review: what happens when
the auxiliary weight is far below the calibrated threshold?  The answer is not
uniform: in the released HAR protocol plain summation is healthy at alpha=0.001
while several default scalarizations contract the representation even there.
That is reported as operator conditioning, not as evidence that the toxifier is
responsible.

USAGE
-----
  export REPFIX_DATA_X=/path/to/data_x
  python repro/scripts/lowalpha_operator_audit.py --domain har \
      --alphas 0.001,0.01,0.1 --methods mgda,imtlg,aligned,nash,famo \
      --seeds 0,1,2 --epochs 60 --m 64 --outdir results/LOWALPHA
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import cross_domain_mtl as cdm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="har")
    ap.add_argument("--alphas", default="0.001,0.01,0.1")
    ap.add_argument("--methods", default="mgda,imtlg,aligned,nash,famo")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--rho-c", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--rel-collapse", type=float, default=0.3)
    ap.add_argument("--outdir", default="results/LOWALPHA")
    a = ap.parse_args()

    outdir = ROOT / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    methods = [x.strip() for x in a.methods.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",")]
    for alpha in [float(x) for x in a.alphas.split(",")]:
        _recs, agg, raw = cdm.run_conflict_ablation(
            a.domain, methods, seeds, alpha, a.m, a.epochs, a.lr, a.batch,
            a.rho_c, a.beta, a.rel_collapse)
        for row in agg:
            row["alpha"] = alpha
        payload = dict(domain=a.domain, kind="lowalpha", methods=methods,
                       seeds=seeds, m=a.m, epochs=a.epochs,
                       alpha=alpha, results=agg, raw=raw)
        path = outdir / ("lowalpha_%s_a%g.json" % (a.domain, alpha))
        path.write_text(json.dumps(payload, indent=2))
        print("wrote %s" % path)
        for row in agg:
            print("  %-9s acc=%.3f collapse=%.2f relS=%.3f" % (
                row["mode"], row["metric_mean"], row["collapse_rate"],
                row["rel_scatter_mean"]))


if __name__ == "__main__":
    main()
