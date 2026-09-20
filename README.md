# RepFix — Spectral Toxicity Filtering for Multi-Task Representation Learning

Code release for *"Gradient Toxicity: When Auxiliary Heads Collapse the Shared
Encoder"* (ICLR 2027, double-blind submission).

Multi-task learning usually worries about gradient **conflict** — auxiliary gradients
that oppose the primary task. This work studies the opposite regime, **gradient
toxicity**: auxiliary heads with a *constant attractor* (reconstruction, distribution
matching, view invariance) push the shared encoder along one variance-contracting axis,
so their gradients are mutually *aligned* rather than opposed. The encoder then falls
through a saddle-node transition into a degenerate fixed point, and no directional
gradient surgery sees it coming.

The order parameter is

    rho = ||Pi_c g_a|| / ||g_p||,

with `Pi_c` the projector onto the batch-centred representation direction. The encoder
collapses once the operating point `alpha * rho` crosses 1. The **Spectral Toxicity
Filter (STF)** deletes only the toxic projection and keeps the rest of the auxiliary
gradient; **STF-cal** derives both its on/off gate and its deletion rank from a single
probe taken on a short detached warm-up, so there is nothing to tune by hand and no
online feedback loop.

## Layout

```
code/
  cross_domain_mtl.py     unified-protocol harness: variance-shrinkage toxifier,
                          STF-0/hard/soft, STF-cal/rank, MLP + small Transformer
                          encoders (--arch), collapse / scatter / rho diagnostics
  real_aux_mtl.py         real auxiliary heads (input reconstruction, subject-ID probe,
                          I/Q reconstruction, SNR regression) under the same controller
  battery_bmtl_v3.py      BatteryMTL flagship: joint SOH/RUL over a TSMixer+S4 encoder,
                          censoring-aware (Tobit) RUL supervision, tri-weight regime gate
  runner.py               job-queue runner; skips any cell whose output already exists
  moo_combiners.py        faithful ports of the reference multi-objective combiners
                          (PCGrad, CAGrad, FAMO, MGDA, Aligned-MTL, IMTL-G/L,
                          Nash-MTL, UW, GradNorm), shared by every driver
  # discovery-domain stack (Appendix A): a production A-share StudentModel with an MDN
  # distributional head, an autoencoder, and a view-contrastive head on a shared encoder
  model.py                student encoder/heads (TSMixer + light SSM + skewed-t MDN)
  light_ssm_v2.py         Mamba-2/SSD state-space block
  skewed_t_mdn.py         skewed-t mixture-density head
  losses.py               multi-objective training loss
  train.py, train_data.py walk-forward training loop and data pipeline
  calibration.py          rolling conformal calibration (TCP / ACI)
  wf_repro.py             walk-forward reproduction driver
  data_loader.py, eval.py, inference.py   backend data loading and evaluation
configs/
  jobs_*.json             one entry per (domain, width, mode, seed) cell
repro/scripts/
  nyu_prep.py             NYUv2 (HF tanganke/nyuv2) parquet -> npz, 144x192
  nyu_bmtl.py             NYUv2 dense benchmark: conv encoder, seg/depth/normal heads,
                          proj and ae auxiliary tracks, probe-only calibration
  collapse_phase.py       toy testbed: phase diagram and capacity law
  wf_*.py, baseline_repro.py, plot_*.py    real-data finance experiments and figures
  moo_combiners.py        copy of code/moo_combiners.py, kept beside its test
  test_moo_combiners.py   differential + invariant test: our PCGrad is diffed
                          against the vendored reference, every combiner is
                          checked for shape/finiteness and for the "tasks agree
                          -> stay on the common direction" property
  _ref_pcgrad.py          vendored BSD-3 copy of WeiChengTseng/Pytorch-PCGrad
                          pcgrad.py, used only as that differential reference
results/
  R2/ .. R6/              raw per-seed JSON behind the paper's tables
  R7/                     NYUv2 dose grid, width calibration, and dense benchmark
logs/
  runner/                 job-queue ledgers: the job list, every launch with its
                          pid, every exit code, and the wall-clock per cell
  cert/                   stdout of the finite-batch certificate runs
  smoke/                  reference output of run_smoke.sh
```

Result JSONs keep per-seed values, not just aggregates, so every paired test in the
paper can be recomputed from them. `_ref_pcgrad.py` is the only third-party file
in the tree; it is BSD-3-Clause (Wei-Cheng Tseng, 2021) and is vendored verbatim
with the license retained. Long trace arrays are stripped to keep the release
small; they regenerate from the code. `logs/README.md` says what the ledgers
record and which deployment paths were substituted on export.

## Running

Python >= 3.10, PyTorch >= 2.0, numpy, scipy. All runs in the paper used a single A100
80 GB; the unified protocol is small enough for a consumer GPU (bf16 autocast optional).

The quickest check that the code runs and the effect is real is the smoke test:

```bash
bash run_smoke.sh
```

One cell, about six seconds on CPU. It prints each mode's collapse rate and relative
scatter, and `logs/smoke/run_smoke.log` is the reference output: `joint` collapses,
the four STF variants do not.

One unified-protocol cell:

```bash
python code/cross_domain_mtl.py --domain har --arch mlp \
    --modes joint,stf0,stfhard,stfsoft,stfcal,stfcalrank \
    --alphas 100.0 --seeds 0,1,2,3,4,5,6,7,8,9 --m 64 \
    --epochs 60 --cal-warm-epochs 4 --rho-trace \
    --out results/har_dose_hi.json
```

`--domain {har,radioml,battery,celeba,nyu}` selects the validation domain and
`--arch {mlp,transformer}` the encoder. The `--modes` flag is the intervention:
`joint` is plain summation, `stf0` is full `.detach()`, `stfhard` and `stfsoft` are the
thresholded and differentiable spectral filters, and `stfcal`/`stfcalrank` use the
one-probe calibrated gate and rank. `configs/jobs_*.json` enumerates every cell behind
the paper's tables; `runner.py --jobs <json> --slots N` executes them.

A real-auxiliary cell:

```bash
python code/real_aux_mtl.py --domain radioml --aux recon --alpha 3000.0 \
    --m 64 --seeds 0,1,2,3,4,5,6,7,8,9 --epochs 25 --out results/rml_aux.json
```

The battery flagship and the A-share walk-forward follow the same pattern; see the
`R2*`, `R5bat*`, and `R5fin*` entries in `configs/jobs_ALL2.json`.

## Multi-task baselines (Section 7.2)

Multi-task learning's standard answer to interfering gradients is to resolve
*conflict*: project one task's gradient off another (PCGrad), solve a min-norm
Pareto combination (MGDA, CAGrad), align the per-task gradients to a common
geometry (Aligned-MTL, IMTL, Nash-MTL), or learn scalar loss weights (UW,
GradNorm, FAMO). `code/moo_combiners.py` ports each of these from its authors'
release, line for line, and names the file it came from in the routine's
docstring; nothing is a proxy or a structured re-implementation of the idea.

All of them act on the *directions* or the *scalar weights* of the per-task
gradients, so none of them can see an auxiliary gradient that agrees with the
primary one along a variance-contracting axis. Running the full family through
the same harness as plain summation at each domain's supercritical dose gives
Table 6 of the paper: every family retains the 1.0 collapse rate of plain
summation in every domain, and on HAR seven of ten fall *below* summation's
accuracy, four of them to near chance. Stretching the auxiliary weight does not
change this; deleting the toxic projection does, which is the whole point.

```bash
python code/cross_domain_mtl.py --domain har \
    --ablate-methods pcgrad,cagrad,famo,uw,gradnorm,imtll,imtlg,nash,aligned,mgda \
    --alphas 0.1 --seeds 0 --epochs 60 --m 128 --out results/MOO/har_s0.json
python code/battery_bmtl_v3.py --dataset MATR --agg pcgrad --alpha-aux 10 \
    --d 128 --epochs 150 --seeds 0 --out results/E2aFAITH/E2aF_MATR_pcgrad_s0.json
python code/wf_conflict_ablate.py --fold 0 --method nash --out results/E2dMOO/fin_fold0_nash.json
python repro/scripts/test_moo_combiners.py
```

`configs/jobs_P02_moo.json` is the full manifest behind the four tables
(the three-domain family sweep, the numeric battery baselines, the
filter x conflict composition grid, and the discovery-domain folds).
`test_moo_combiners.py` diffs our PCGrad against the vendored reference on the
same random problem and the same parameter set, and checks each combiner for
the invariant that matters here: when the two tasks agree, the output must stay
on their common direction rather than inventing a resolution.


The failure was first observed on a production equity-ranking model built from the
`model.py` stack: a pairwise-rank primary head sharing an encoder with an MDN
distributional head, an autoencoder, and a view-contrastive head. Out of sample the
model drifted to a near-constant ranking while each individual objective still reported
a low training loss — the signature of collapse onto the constant attractor, invisible
to the scalar losses. Appendix A of the paper gives the details; `repro/scripts/`
reproduces the controlled version, and `repro/A_capacity_law_analysis.md` and
`repro/B_pipeline_analysis.md` document the two write-ups behind it.

```bash
python repro/scripts/wf_collapse_v2.py      # real-data collapse sweep
python repro/scripts/wf_stf_ablation.py     # STF-0 / hard / soft on the rank-loss path
python repro/scripts/wf_toxicity_sweep.py   # auxiliary-weight sweep
```

## NYUv2 dense benchmark

`repro/scripts/nyu_bmtl.py` is the fifth-domain harness: a four-block convolutional
encoder with linear heads for 13-class segmentation, depth, and surface normals, plus
two auxiliary tracks — `proj` (the controlled variance-shrinkage toxifier) and `ae` (a
trainable reconstruction decoder, i.e. the production-style head). `joint3` runs the
three primary tasks with no auxiliary head and is the reference row.

```bash
python repro/scripts/nyu_prep.py                      # parquet -> data_x/nyu/nyu_nyu.npz
python repro/scripts/nyu_bmtl.py --configs joint3 --aux proj --alpha-aux 1.0 \
    --seeds 0,1,2,3,4,5,6,7,8,9 --epochs 60 --batch 32 --d 128 \
    --cal-warm-epochs 5 --out results/R7/R7_bmtl_joint3.json
```

`results/R7/` holds the nine-point dose grid (`R7_dose_nyu_a*.json`), the width
calibration probes (`R7_calib_nyu_m*.json`), and the three-task benchmark. The harness
reads the auxiliary track through `CFG_AUX`, so a config that is absent there runs
aux-free.

## Reference points

A collapsed encoder is easy to miss. The collapse criterion is relative cross-sample
scatter `S_h < 0.3 * S_single`, anchored per seed to that seed's own single-task
ceiling, and on some domains the primary metric never registers the failure — HAR
accuracy stays near 0.95 on a collapsed representation, and NYUv2 depth RMSE actually
*improves*. Both are discussed in the paper's Scope and Failure Modes section, together
with the regime where the estimated projector saturates and STF-0 (full deletion) is
the correct choice.

The toxifier is a variance-shrinkage projection over three seeded random projections,
`L = Var_b[W_k h]` — a benign-sounding "feature invariance" objective whose exact
constant attractor makes it a controlled stand-in for the real heads. All datasets are
public (UCI-HAR, RadioML2016.10A, NASA PCoE and MATR battery archives, A-share daily
bars, CelebA, NYUv2).
