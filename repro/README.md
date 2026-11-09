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
  plot_final_figures.py      paper/figures/unified_all.pdf (Fig. 4) and
                             paper/figures/battery_repair.pdf (Fig. 5), from the recorded
                             results (results/BSCALE, results/R7, results/R5fin, repro/results)
                             Writes paper/figures/; figures/ holds byte-identical copies.
  plot_unified_v4.py         earlier six-panel set (superseded, 2026-09-22); it writes
                             figures/ only and is not the manuscript figure
  plot_realdata_collapse.py  legacy real-data collapse figure (superseded, 2026-08-21)
  plot_capacity_law.py       legacy capacity-scaling figure (superseded, 2026-08-21)
  plot_paper_figures.py      legacy multi-panel figure script (superseded, 2026-08-21)
A_capacity_law_analysis.md   legacy capacity-law note, 2026-08-21 -- superseded, see banner
B_pipeline_analysis.md       legacy pipeline note,     2026-08-21 -- superseded, see banner
results/                     experiment JSON outputs
figures/                     published copies of the paper figures (PDF/PNG)
```

The manuscript has **five** figures and they are split across **two** generators. The
listing above covers only the last two; the first three come from
`paper/revision/build_figures.py`:

| figure | file | generator |
|:---:|---|---|
| 1 | `paper/figures/fig_survival_gap.pdf` | `paper/revision/build_figures.py` |
| 2 | `paper/figures/fig_probe_boundary.pdf` | `paper/revision/build_figures.py` |
| 3 | `paper/figures/fig_diagnostic_intervention.pdf` | `paper/revision/build_figures.py` |
| 4 | `paper/figures/unified_all.pdf` | `repro/scripts/plot_final_figures.py` |
| 5 | `paper/figures/battery_repair.pdf` | `repro/scripts/plot_final_figures.py` |

Run them from the repository root:

```
python paper/revision/build_figures.py     # Figs 1-3, plus paper/revision/evidence_audit.json
python repro/scripts/plot_final_figures.py # Figs 4-5
```

`build_figures.py` also writes `paper/revision/evidence_audit.json`, the per-cell record
dump behind the figure captions and the audited numbers (probe cells, warm-up ratios,
zero-dose drift, the deduplicated dose-scan records, and the source-file SHA-256 list).
Its Figure-3 input is the raw dose-scan log directory `repro/results/_screena/`; the
script fails loudly if that directory is absent rather than emitting blank panels.

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

Current research diagnostics and explicitly experimental candidates are documented
in [ALGORITHM_RESEARCH.md](ALGORITHM_RESEARCH.md). The scripts
`stf_rms_experiment.py`, `toxic_operator_audit.py`, `controlled_rank_scan.py`, and
`battery_representation_audit.py` keep their new outputs separate from the
published runs. Candidate pilot results are not evidence of method superiority.

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
| `syn8_sweep.sh` | `code/synth_beta_scan.py` | 26 completed cells of the planned 63: beta=0 at m=32/48/64/96/128; beta=0.25/0.5/0.75/1 at m=32/48/64/96; beta=0.125 at m=32/48; beta=0.375/0.625/0.875 at m=32. 21 alpha points, ten seeds, 800 epochs | `repro/results/SYN8/` |
| real-head basis sweep | `code/real_aux_mtl.py` | HAR reconstruction, alpha 12/30/100 × `--cal-basis {pca,aux}`, `--rho-rank-max 64`, ten seeds | `results/XDRBASIS_REAL/` |
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

## Audits behind the theory revision

These are separate from the published sweeps and write next to the runs they
audit. They are the artifacts the reproducibility statement names.

| Script | What it establishes | Output |
|---|---|---|
| `synth_beta_joint.py` | joint capacity fit over the 26 SYN8 cells, pivoted at the width whose toxic rank is common to every schedule | stdout |
| `analyze_rank_controls.py` | RS1 half-height crossings and width slopes on the shipped crossing rule | `../results/OPERATOR_AUDIT/rs1_summary.json` |
| `controlled_rank_scan.py` | exact-rank orthonormal control: `A = Q diag(lambda) Q^T` at exactly specified rank under three normalizations. `--self-test` asserts the rank and the amplitude and prints PASS | `../results/CONTROLLED_RANK/` |
| `run_controlled_rank_grid.py` | the twelve-cell control grid (three widths x two ranks x two normalizations) | `../results/CONTROLLED_RANK/` |
| `theory_counterexamples.py` | eight executable counterexamples to two overbroad early claims; `--out` is required | `../results/OPERATOR_AUDIT/counterexamples.json` |
| `reconstruction_basis_audit.py` | reconstruction-gradient energy against deletion rank, principal basis against auxiliary basis | `../results/OPERATOR_AUDIT/har_recon_basis.json` |
| `battery_validation_protocol.py` | cell-disjoint train/validation/test, preprocessing fitted on training cells, observed-only RUL error reported apart from the censored shortfall | `../results/BAT_VALIDATION_V2/` |
| `parameter_guard_har.py`, `parameter_variance_guard.py` | exploratory filter candidates, kept as candidates rather than as evidence of method superiority; `--safety` sets the calibrated attenuation target, and `qgrid_a*.json` / `calsafety_a*_s*.json` are the matched-attenuation sweep behind the appendix's matched-attenuation table | `../results/PARAM_GUARD/` |
| `paired_stats.py` | the paired joint-vs-STFcal tests of Section 5.2: exact two-sided Wilcoxon signed-rank, Cohen `d_z`, the battery SOH-RMSE interval and the Holm correction over four scatter comparisons and one battery SOH comparison, recomputed from the per-seed `raw` records of `results/R3/R3_dose_{har,radioml,battery}_hi.json`, `results/R7/R7_dose_nyu_a1.json` and `results/R6/R6_dose_celeba_hi.json` | stdout |
| `cross_domain_mtl.py --ablate-methods ...`, kernels in `moo_combiners.py` | the ten gradient-manipulation families at each domain's supercritical dose, behind the conflict-baseline table: per-seed `raw` records plus the per-mode aggregate the table prints. The columns sit at different widths --- HAR `m=128` and RadioML `m=64`, battery `m=32` --- which the table caption states. The 129 declared jobs are consolidated into three per-domain files, one record per seed | `repro/results/P02/MOO_har.json`, `MOO_radioml.json`, `MOO_battery.json` |

The exact-rank control is a negative control for the capacity law rather than a
confirmation of it: with the rank and the normalization each held exactly, the
three-width fits carry one residual degree of freedom and the fitted exponent
follows the normalization rather than the rank. The cheapest check of the
construction is `controlled_rank_scan.py --self-test`.

## Paired CPU continuation and fixed readout repairs

The current manuscript also includes a separate CPU-only paired protocol.
Calibration examples are reserved before encoder/head fitting. All branches
clone the complete healthy model and Adam state and reuse future minibatches.
The first 20 runs cover HAR isotropic/projected variance and RadioML SNR; the
next 10 use real reconstruction with fixed HAR doses `1,10,100` and RadioML
doses `30,300,3000`. All results, including the near-null RadioML continuation
control, are retained under `repro/results/CPUCONT/`.

With prepared input data, one numerical rerun is:

```bash
python repro/scripts/paired_readout_continuation.py --domain har --aux recon \
  --seed 0 --warm-epochs 60 --cont-epochs 30 --doses 1,10,100 \
  --data-root /path/to/data_x --out repro/results/CPUCONT/example/u000.json
python repro/scripts/checkpoint_repair_audit.py \
  --record repro/results/CPUCONT/example/u000.json --data-root /path/to/data_x \
  --out repro/results/CPUCONT/example/repair.json
```

`checkpoint_repair_audit.py` first reproduces the original healthy/frozen
metrics, then compares mean, global scale, their combination, per-feature
statistics, paired diagonal regression, scaled Procrustes, covariance alignment,
and paired ridge. All maps fit fitting-split features only, with no new labels
or test-statistic fitting. Ridge is fixed at `1e-3`; the covariance eigenvalue
floor is `1e-6` of the largest eigenvalue. The paired estimators need old/new
features of the same anchors. They are diagnostic post-hoc baselines, not a
reimplementation of jointly trained BC-Aligner.

`reconstruction_task_audit.py --record <record.json> --out <audit.json>` evaluates
the saved healthy/final decoders without fitting, with `REPFIX_DATA_X` set to
the prepared data root. Its default MSE uses the declared auxiliary forward
filter and frozen gate in fixed 256-example batches; raw-head MSE is also saved.
`radioml_split_audit.py --out <audit.json>` checks all prepared cache tensors
against the original dictionary's documented seeded 50/50 partition, read-only.

To regenerate the manuscript numbers, tables, and figure from the complete
shipped numerical records without datasets or model checkpoints:

```bash
python repro/scripts/summarize_readout_mechanism.py
python -m pytest repro/scripts/test_readout_alignment.py repro/scripts/test_scale_boundaries.py -q
python repro/scripts/build_cpu_paper.py
```

`verify_cpu_evidence.py` and `verify_readout_mechanism.py` additionally verify
model/optimizer checkpoint hashes when the full training runs have been
regenerated at their recorded paths. The released verification JSON records
the checks on the original saved numerical artifacts; large trained states and
raw datasets are not distributed. Analysis and PDF compilation are local.

## Waveform CNN and unlabeled consumer decisions

The next extension holds the prepared RadioML inputs/split, task heads, and
paired engine fixed while replacing the encoder with two convolution/pooling
blocks and a width-64 feature layer. Five seeds use 25 healthy plus 15
continuation epochs, reconstruction doses `30,300,3000`, joint/detached arms,
and primary-only continuation. No STF-cal CNN result is claimed.

```bash
python repro/scripts/cnn_reconstruction_continuation.py --seed 0 \
  --data-root /path/to/data_x --out repro/results/CPUCONT/frontier/c000.json
```

With `REPFIX_DATA_X` set to the same input root, the independent audit is:

```bash
python repro/scripts/checkpoint_consumer_audit.py \
  --record repro/results/CPUCONT/frontier/c000.json --family-size 3015 \
  --out repro/results/CPUCONT/cnn_consumer/n000.json
```

The fixed bank contains all 300 earlier endpoints plus 35 CNN endpoints, each
with nine repair candidates. Every corpus supplies 1,024 uniformly sampled
unlabeled monitor inputs; the rest is scored afterward. Monitor selection
requires healthy reference predictions. Paired repair fitting separately needs
paired old/new training anchors. A hypergeometric tail bound with total bank
error probability 0.05 supports relative-loss budgets 0.05 and 0.10. Selection
uses only the certificate and fixed repair order, with abstention when no arm
passes. Disagreement control is a standard baseline, not absolute accuracy
estimation, future-step prediction, or a new population guarantee.

The release keeps candidate-level bounds, all choices/abstentions and remainder
scores, the CNN grid, and exact source/checkpoint links. Reproduce their local
summary and figures without raw data or trained checkpoints:

```bash
python repro/scripts/summarize_consumer_frontier.py
python repro/scripts/render_paper_figures.py
python -m pytest repro/scripts/test_consumer_risk.py repro/scripts/test_readout_alignment.py repro/scripts/test_scale_boundaries.py -q
python repro/scripts/build_cpu_paper.py
```

`verify_consumer_frontier.py` additionally needs the locally regenerated model
states. Its verification JSON is shipped. The finite-population tests enumerate
every possible disagreement count on small corpora, rather than merely checking
the inversion implementation against itself. Branch counts are descriptive;
multiple branches of the same seed are not independent trials.

### Publication figures and head-state boundary

`render_paper_figures.py` regenerates all thirteen included figures locally from
the shipped records, including editable SVGs. It uses a final 5.5-inch canvas,
checks label bounds and vector fonts, and verifies that legacy text styling
does not change plotted numerical artists. The manifest contains physical
sizes, source/output hashes and plotted-data hashes. The opening real-HAR panel
uses full-test repair scores from `CPUCONT/recon_repairs/`, with all 450 map
hashes checked against the fixed consumer bank; remainder scores are not mixed
into it. Paired linear alignment needs paired training-feature anchors.

`auxiliary_state_factorial.py` runs the separately declared dose-30 SNR control:
warm/initial auxiliary weights by retained/reset auxiliary Adam state, fixed
encoder/primary state and future batches, five archived references. Its 20
endpoints include five exact archived replays and 15 new conditional paths.
It is outside the fixed consumer certificate bank. All four cells retain zero
scatter-collapse endpoints; no significant frozen-accuracy ordering is claimed.
Records, seed-paired summaries and checkpoint verification are shipped under
`CPUCONT/head_state_first/`, `CPUCONT/head_state/`, `head_state_summary.json`
and `head_state_verification.json`. Running `summarize_head_state.py` again
performs full model/Adam checks and therefore requires the large local trained
states, which are excluded from the source ZIP. The generated tables/macros
and numerical summaries are included for a data-free local LaTeX rebuild.

### Predeclared real-auxiliary benefit and exact consumer control

`docs/plans/2026-11-09-prospective-benefit-compatibility.md` fixes the study before
calculation. `benefit_development_choice.json` freezes the validation choice
before confirmation. Development seeds 200--202 cover the six declared HAR /
RadioML dose candidates and compute no test outcomes. Confirmation seeds
300--309 use the selected low-label SNR dose-1 recipe with ordinary/guarded
detachment/joint branches, shared model/Adam states and future orders.

`readout_nullspace.py` constructs the projector from the normalized healthy
head, including feature scales. `benefit_compatibility_cpu.py` performs only
CPU numerical work. `verify_benefit_consumers_cpu.py` independently reconstructs
the forty saved endpoint consumers. The separate fixed audit bank has 40
consumers and does not inherit the earlier 3,015-candidate guarantee. Protection
is fixed before continuation; anchor-drift logs are not an adaptive controller.

The local state-dependent verifier/analysis is
`summarize_benefit_compatibility.py`. Its verification JSON confirms 104 model /
optimizer states, all five criteria, and exact independent replay. Trained
states and raw datasets are excluded from the source ZIP. The complete
numerical records, all negative development candidates, frozen selection,
generated tables/macros and figure inputs are included. `select_benefit_configuration.py`
deliberately refuses to overwrite an existing frozen choice or choose again
after confirmation. It is not a post-hoc tuning command.

The guard needs the frozen reference encoder at inference. Its measured CPU
batch latency is 2.37x ordinary inference, and it emits one shared 64-vector.
Dual routing is a distinct, explicitly discussed baseline. Full-column-rank
heads leave no nullspace feature freedom; primary gains outside this fixed
low-label protocol remain unestablished. See the latest revision report for
the exact statistical scope and all costs.

```bash
python repro/scripts/render_paper_figures.py
python -m pytest repro/scripts/test_consumer_risk.py repro/scripts/test_readout_alignment.py repro/scripts/test_scale_boundaries.py repro/scripts/test_readout_nullspace.py -q
python repro/scripts/build_cpu_paper.py
```

## Reproduce

### Current manuscript evidence audit (2026-09-21)

Run from the repository root:

```bash
python repro/scripts/current_evidence_audit.py
python repro/scripts/real_basis_evidence_audit.py
python repro/scripts/finance_fold_audit.py
python repro/scripts/summarize_filter_semantics.py
python repro/scripts/filter_semantics_audit.py --self-test
python repro/scripts/plot_final_figures.py
```

The first two scripts regenerate `paper/battery_evidence_table.tex` and
`paper/real_basis_evidence_table.tex`; both are required manuscript inputs.
The last one rebuilds the two manuscript figures from the recorded results and writes
them to `paper/figures/`.
The JSON audits in `results/OPERATOR_AUDIT/` record source hashes and distinguish
test covariance from legacy validation coordinate variance, nominal rank from
gate-active rank, and fold-level comparisons from pooled financial fold-seed
cells. The corrected battery intervals measure initialization variation on one
fixed cell split, not new-cell population uncertainty.

`results/FILTER_SEMANTICS/` retains an alpha=12 three-seed pilot and an alpha=100
ten-seed matched comparison in two disjoint seed shards. Both projection maps
share the same Jacobian and calibrated basis/rank; only the auxiliary forward
value differs. Backward-only filtering improves reconstruction error, decreases
scatter, and does not significantly improve primary accuracy. These are
exploratory results, not a claim of algorithm superiority. To rerun the fixed
ten-seed configuration in one file:

```bash
python repro/scripts/filter_semantics_audit.py --alpha 100 --seeds 0,1,2,3,4,5,6,7,8,9 --out results/FILTER_SEMANTICS/replication_a100_10s.json
```

The real-head probe now disables autocast for its covariance/residual algebra
as well as gradient extraction. Previously, an enclosing bf16 training context
could make the auxiliary-basis eigendecomposition fail on CUDA. The semantic
self-test checks both bases under enclosing bf16 against fp32, in addition to
forward values, Jacobians, head gradients and the surviving mean channel.

### Original data-dependent reproduction

Set `REPFIX_DATA_DIR` to your prepared A-share data (the `baseline_fold{N}.npz` fold
files), then from this directory:

```bash
python scripts/wf_collapse_v2.py          # -> results/wf_collapse_v2.json
python scripts/plot_realdata_collapse.py  # results/ -> figures/*
python scripts/plot_capacity_law.py       # results/capacity_out_capacity.json -> figures/*
```

The toy-testbed capacity law (`collapse_phase.py`, 8 seeds) and the real-data collapse
runs take any CUDA device or a Slurm `sbatch`.
