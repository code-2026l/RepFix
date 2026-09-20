# Reproductions

Scripts and write-ups behind the paper's experimental sections. Everything here
depends only on NumPy and PyTorch (no scipy, no numba), so it runs unchanged on CUDA
nodes or on AMD DCU/Slurm nodes.

```
scripts/
  collapse_phase.py          toy testbed: collapse phase diagram and capacity law
                             (--rho-def {norm,percell} selects the order-parameter
                             estimator; see results/README.md)
  nyu_prep.py                NYUv2 parquet -> npz (144x192), unified protocol
  nyu_bmtl.py                NYUv2 dense benchmark: seg / depth / normals, proj and ae tracks
  wf_repro.py                A-share walk-forward StudentModel pipeline
  wf_collapse_v2.py          real-data collapse deepening (rank loss + fixed projection)
  wf_toxicity_sweep.py       auxiliary-scale sweep on real data
  wf_fixed_aux_bridge.py     fixed-projection bridge on real data
  wf_bridge_deep.py          deepened bridge scan
  wf_collapse_rank_trap.py   prediction-level rank trap
  plot_unified_v4.py         THE PAPER FIGURES: figures/unified_all.pdf (Fig. 1) and
                             figures/battery_repair.pdf, from results/ (xcap*, xrep*, R5fin*)
  plot_realdata_collapse.py  legacy real-data collapse figure (superseded, 2026-08-21)
  plot_capacity_law.py       legacy capacity-scaling figure (superseded, 2026-08-21)
  plot_paper_figures.py      legacy multi-panel figure script (superseded, 2026-08-21)
A_capacity_law_analysis.md   legacy capacity-law note, 2026-08-21 -- superseded, see banner
B_pipeline_analysis.md       legacy pipeline note,     2026-08-21 -- superseded, see banner
results/                     experiment JSON outputs
figures/                     paper figures (PDF/PNG)
```

## What these produce

**Real-data collapse.** A full-market A-share StudentModel on walk-forward fold 0 with
a rank-loss primary. Sweeping the toxicity weight of a fixed random-projection head,
`c_aux ∈ {0,1,5,20,100}`, the cross-sample scatter falls by roughly three orders of
magnitude and the projection variance bottoms out near `1e-7` — the representation
collapses on real data, not only in the testbed:

| c_aux | S_h scatter | projection var. | OOS rank-IC |
|:---:|:---:|:---:|:---:|
| 0   | 3.79e-1 | —      | +0.043 |
| 1   | 1.72e-1 | 2.5e-4 | +0.087 |
| 5   | 8.39e-2 | 5.5e-5 | -0.049 |
| 20  | 1.07e-2 | 1.7e-4 | -0.057 |
| 100 | 1.67e-4 | 1.1e-7 | -0.003 |

The light dose (`c_aux=1`) helps — rank-IC rises from +0.043 to +0.087. Past the
transition the collapse basin destroys prediction, as the stability condition
`r = alpha*rho < 1` predicts.

**Capacity law.** The exponent is read under one rule: `rel_scatter` normalised by the
*same seed's* single-task scatter, so a healthy cell sits at `1.0`, and `alpha_c` is
where the mean relative scatter first crosses **`0.5`**. `scripts/capacity_exponent.py`
reproduces every exponent the paper prints from the shipped sweeps. The synthetic
testbed (`results/th_critical.json`, fixed toxic fraction, i.e. the `beta = 1` endpoint)
reads `gamma = 0.01`; the undiluted anchor `gamma_0 = 2.43` comes from the fixed-rank
family `results/SYN5/`, `results/SYN7/` (`scripts/synth_rank_analyze.py`,
`synth_beta_analyze.py`), not from this grid. Estimator, guard and per-domain values:
`../results/README.md`, section "Capacity exponent".
Output: `results/capacity_out_capacity.json`.

> The `alpha_c ≈ 0.01 * m^1.47` / `gamma = 1.53` / `m* ≈ 133` values that appeared here
> previously are the **retired absolute-rule** readings of the nine-point grid
> (`th_capacity.json`). They are not the paper's numbers and are kept only in the
> 2026-08-21 legacy notes below.

**NYUv2 fifth domain.** The unified harness (`code/cross_domain_mtl.py --domain nyu`)
consumes a 9x12 coarse depth grid; the dense benchmark (`nyu_bmtl.py`) predicts 13-class
segmentation, depth, and surface normals from a small conv encoder under either the
controlled `proj` toxifier or the trainable `ae` reconstruction decoder. The calibration
probes use `--configs` that omit the auxiliary head, so the probe is measured on a
detached healthy anchor:

```bash
python scripts/nyu_prep.py
python scripts/nyu_bmtl.py --configs joint3 --aux proj --alpha-aux 1.0 \
    --seeds 0,1,2,3,4,5,6,7,8,9 --epochs 60 --batch 32 --d 128 \
    --cal-warm-epochs 5 --out results/R7/R7_bmtl_joint3.json
```

Per-seed outputs land in `../results/R7/`.

**Order-parameter scan.** `collapse_phase.py --sweep rho --rho-def norm
--seeds 12` reproduces the initialization spectrum quoted in the capacity-law
appendix (ρ = 0.74 ± 0.19, α_c = 1.45 ± 0.37); `--rho-def percell` switches to
the per-cell mean-of-ratios variant. Both write `results/th_rho*.json`.

## Cluster sweep launchers

The `.sh` launchers that drove the multi-job sweeps are not versioned: each one
embeds the deployment prefix of the machine it was submitted from, and shipping
that prefix would leak the author's cluster layout. Their command lines are
reproduced here instead. Every launcher is a plain queue over a driver that
ships in `code/`, so each can be replayed from the repository root.

| Launcher | Driver | Sweep | Output |
|---|---|---|---|
| `xdrcurve_sweep.sh` | `code/cross_domain_mtl.py` | UCI-HAR dose-to-rank curve: ten doses `0.003`--`100` × `--cal-basis {pca,aux}`, `--rho-rank-max 64`, five seeds, 60 epochs | `results/XDRCURVE/` |
| `xdrank_sweep.sh` | `code/cross_domain_mtl.py` | Seven `(domain, alpha)` pairs at `--rho-rank-max {3,64}`, ten seeds (CelebA four) | `results/XDRANK/` |
| `xdbasis_probe.sh` | `code/cross_domain_mtl.py` | Auxiliary-basis runs at `--rho-rank-max 64` on HAR, battery and CelebA, plus the one-cell port check | `results/XDBASIS/` |
| `ll_sweep.sh` | `code/real_aux_mtl.py` | Semi-supervised label fraction: `--alpha {1,30}` × `--label-frac {0.02,0.05,0.2,1.0}`, ten seeds | `results/LOWLAB/` |
| `syn8_sweep.sh` | `code/synth_beta_scan.py` | The seven-width synthetic rank-schedule family: `m ∈ {32,...,256}` × nine `beta`, 21-point `alpha` grid, ten seeds, 800 epochs | `repro/results/SYN8/` |
| `sg_sweep.sh` | `code/battery_bmtl_v3.py` | The MATR flagship at a capped probe depth (`--cal-rank-max 3`) with the rank-rule arms | `results/SG/` |
| `time_matrix.sh` | `repro/scripts/time_overhead.py` | Rotated wall-clock matrix with `--rotate --wait-idle` | `repro/results/TIME/` |

The dose-to-rank curve and the rank-budget contrast in full:

```bash
# dose-to-rank curve, principal basis (the paper's surrogate) and aux basis
for al in 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100; do
  for bs in pca aux; do
    CUDA_VISIBLE_DEVICES= code/cross_domain_mtl.py --domain har \
        --modes joint,stf0,stfcalrank --alphas "$al" --seeds 0,1,2,3,4 \
        --m 64 --epochs 60 --cal-warm-epochs 4 \
        --rho-rank-max 64 --cal-basis "$bs" \
        --out "results/XDRCURVE/har_a${al}_${bs}.json"
  done
done

# the same dose at two probe depths; only --rho-rank-max differs
for r in 3 64; do
  CUDA_VISIBLE_DEVICES= code/cross_domain_mtl.py --domain har \
      --modes joint,stf0,stfcal,stfcalrank --alphas 100 \
      --seeds 0,1,2,3,4,5,6,7,8,9 --m 64 --epochs 60 --cal-warm-epochs 4 \
      --rho-rank-max "$r" --out "results/XDRANK/har_a100.0_r${r}.json"
done
```

The `aux` runs of the first loop also carry the two `stfcalg` arms, which is why
the shipped `XDRCURVE` files record a mode set one longer than the `pca` files.

## Reproduce

Set `REPFIX_DATA_DIR` to your prepared A-share data (the `baseline_fold{N}.npz` fold
files), then from this directory:

```bash
python scripts/wf_collapse_v2.py          # -> results/wf_collapse_v2.json
python scripts/plot_realdata_collapse.py  # results/ -> figures/*
python scripts/plot_capacity_law.py       # results/capacity_out_capacity.json -> figures/*
```

The toy-testbed capacity law (`collapse_phase.py`, 8 seeds) and the real-data collapse
runs take any CUDA device or a Slurm `sbatch`.
