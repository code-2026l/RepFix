# results/ — raw per-seed artefacts

Every JSON here is a raw run output: the aggregate means, and the per-seed
records behind them wherever the runner kept them. Each directory below names
the table or figure it backs. One file does not retain per-seed records
(`B10/MATR_eolonly_10s.json`, see its row below); everything else can be
recomputed from the shipped data alone.

## Layout

| Directory | Backs | Notes |
|---|---|---|
| `B4/` | `tab:bmtl`, ceiling row | Four-seed flagship battery runs, MATR and NASA. `single_ceiling.soh.rmse_soh` and `single_ceiling.rul.mae_rul` are the single-task ceilings printed in `tab:bmtl` (MATR 0.51 / 694.6, NASA 9.17 / 18.5). The same files also carry four-seed `joint`/`stf0` rows; those are **not** the rows in the table, which are ten-seed. |
| `B10/` | `tab:bmtl` rows, `tab:cens_ablation` | Ten-seed flagship battery runs. `NASA_main_10s.json` supplies every NASA row of `tab:bmtl` except the ceiling; `MATR_main_10s.json` supplies MATR `joint`, `stf0`, `stf0_w`; `MATR_toxvar_10s.json` the MATR toxifier row; `NASA_toxvar_10s.json` and `NASA_stfcal_10s.json` their rows. `MATR_nocens_10s.json` and `MATR_eolonly_10s.json` are the two legacy supervision variants of `tab:cens_ablation`. Note the two are not equally reproducible: `MATR_nocens_10s.json` keeps its ten `per_seed` records, so the four-seed legacy row of `tab:cens_ablation` can be recomputed from it, whereas `MATR_eolonly_10s.json` stores only the ten-seed aggregate — its four-seed row (`1.53 ± 0.13` / `0.53 ± 0.03`) cannot be regenerated from this release, and the ten-seed values differ (`1.59 ± 0.17` / `0.55 ± 0.04`). |
| `BAT_VALIDATION_V2/` | `tab:bmtl_corrected` | The corrected battery protocol: cell-disjoint train/validation/test splits, preprocessing fitted on training cells only, checkpoint selection on validation, observed-only RUL error reported apart from the censored shortfall. `MATR_10s150.json` and `NASA_10s150.json` hold ten seeds × five modes (`single_soh`, `joint`, `stf0`, `stfcal`, `stfcalrank`) at 150 epochs; `MATR_3s150.json` is the three-seed pilot. Each row's `test` block carries `soh_rmse` and `rul_observed_mae` (the two metric columns of the table), its `validation` block the `scatter` column, and its `representation` block the test-split covariance diagnostics. The table is MATR only; the NASA file is the companion run, where `stf0` also beats `joint` on both tasks but the calibrated gate stays shut in all ten seeds, so `stfcal` falls back to `joint`. Driver: `repro/scripts/battery_validation_protocol.py`. |
| `E2a/` | `tab:conflict_num` | Per-seed battery runs of the faithful combiner ports (`code/moo_combiners.py`) at the supercritical dose of that table, five seeds per `(dataset, combiner)` cell. `sum` is plain summation; the others are PCGrad, FAMO and CAGrad at their published defaults. |
| `../repro/results/P02/MOO_*.json` | `tab:conflict` | The ten gradient-manipulation families at each domain's supercritical dose, ten seeds each: `MOO_har.json` (`alpha=0.1`, `m=128`), `MOO_radioml.json` (`alpha=10000`, `m=64`) and `MOO_battery.json` (`alpha=3000`, `m=32` --- the width the printed battery column was run at). Each file holds the ten per-seed records, every one carrying both the per-mode `results` aggregate (`metric_mean`, `collapse_rate`, `rel_scatter_mean`) and the per-`(mode, seed)` `raw` rows (`alpha`, `m`, `rho_mean`, collapse flag); a table cell is the mean of `collapse_rate` over the ten seeds and, in parentheses, the mean of `metric_mean`. These are the consolidated form of the 129-job `configs/jobs_P02_moo.json` (see the manifest note below); a wider `m=64` battery variant of the same sweep exists only as an untracked scratch file and is not what the table prints. |
| `../repro/results/CERT/` | `tab:certificate` | The frozen-anchor sampling check of Proposition~\ref{prop:calibration}. `calib_cert_<domain>_m64[_iid].json` re-runs the shipped probe on `reps=2000` resampled batches per size `B ∈ {16,…,4096}` at a frozen healthy anchor (`m=64`, four detached warm-up epochs, five seeds). `fit.cv_ps.slope` and `fit.sd_ps.slope` are the two slopes quoted in the text (`-0.500` for the coefficient of variation and `-0.494` for the standard deviation on HAR); `rows` holds the per-`B` `rho-hat`, its spread, and the measured and Bernstein-envelope probabilities. The `_iid` files draw cells with replacement, the others without. Driver: `repro/scripts/calib_cert_b.py`. |
| `E2b/` | `tab:aux_sweep` | The auxiliary-weight (dose--response) sweep on MATR at seed 0: three widths `d ∈ {32,128,256}` × four weights `λ_aux ∈ {0.01, 0.1, 1, 10}`, twelve files, one seed each. Every file carries the `joint` row, the dose-free `stf0` row (identical across λ, since the detached arm never sees `λ_aux`), and both single-task ceilings; together they are every number in `tab:aux_sweep`. Manifest: `configs/jobs_E2b.json`; driver: `code/battery_bmtl_v3.py`. |
| `E2b2/` | the degenerate-encoder observation of Section~\ref{sec:validation} | The same flagship at the theory-consistent toxifier (`--tox-var`, `u_c = h - E[h]`) instead of real auxiliary heads, because on the narrow width of `E2b/` the failure mode there is aux-feature domination and the variance-collapse transition is not bracketed. Three widths `d ∈ {32,128,256}` × four weights `α_aux ∈ {1,3,10,30}`, twelve files, one seed each, same recipe as `E2b/` (150 epochs, batch 256). The `d=256, α_aux=30` cell is the source of the main-text sentence that a fully degenerate encoder can beat detachment on SOH-RMSE: `joint` scores `0.8242` at relative scatter `0.00523`, against `stf0` at `1.5148` --- the same `stf0` row `E2b/` carries, since the detached arm never sees `α_aux`. Manifest: `configs/jobs_E2b2.json`; driver: `code/battery_bmtl_v3.py`. |
| `R2/` | cross-check | Earlier four-seed flagship runs. Superseded by `B4/` and `R5bat/`, kept because the four-seed ceiling line is reproduced from them. |
| `R3/` | calibration and dose scans | One-probe calibration at five widths for battery, HAR and RadioML, plus a high/low dose contrast per domain. |
| `R4/` | Transformer-encoder runs | `former` names the Transformer encoder, not a superseded grid: these are the calibration, dose and conflict runs behind the Transformer claims of Section~\ref{sec:validation} (RadioML at `alpha=3000`) and Appendix~\ref{app:moved} (HAR and battery at `alpha=100`). They are parallel to `R3/`, which holds the MLP runs. |
| `R5aux/` | `tab:real_aux` | Real auxiliary heads — HAR input-reconstruction, HAR subject-ID probe, RadioML I/Q-reconstruction, RadioML SNR regression — at `m=64`, ten seeds. |
| `R5bat/` | `tab:bmtl` | Ten-seed flagship battery runs. `R5bat_MATR_10s.json` supplies the MATR `stfcal` row of `tab:bmtl` and repeats the MATR `joint`/`stf0` rows of `B10/`; `R5bat_NASA_10s.json` is the companion ten-seed NASA run. |
| `R5fin/` | `tab:finance_repair` | A-share walk-forward folds: ten folds × ten seeds × four modes, reported as out-of-sample rank-IC. |
| `R6/` | CelebA toxicity check | Calibration at four widths and a high-dose grid for the CelebA (Smiling) instance. |
| `R7/` | `tab:nyu_dose`, `tab:nyu_bmtl`, NYUv2 capacity | NYUv2 dense benchmark: calibration probes at three widths (the fourth, `m=256`, is in `NYUCAP/`), the dose response, and the three-task comparison for the `proj` and `ae` auxiliary branches. |
| `NYUCAP/` | NYUv2 capacity exponent (Appendix~\ref{app:nyu}) | The NYUv2 α sweeps at the four widths of the capacity law (`m ∈ {32,64,128,256}`), ten seeds each, plus the `m=256` one-probe calibration that completes the series `R7/` stopped at `m=128`. Manifest: `configs/jobs_NYUCAP.json`; driver: `code/cross_domain_mtl.py`. The sweeps exist to put the two α_c estimators on one protocol, and they do not agree: the probe (`cal_rho_u`) gives `γ = 0.27` over four widths, while the scatter half-height crossing returns `γ = −0.77` — the opposite sign — because the half-height point fires while no seed has collapsed (`0.069` at `m=64`, where the collapse rate is `0/10`, against the probe's `α_c = 0.498`, where it is `10/10`). The `m=64` sweep reproduces the `R7/` dose curve at `m=64` to floating-point equality at all nine shared α, so this is the published protocol at more widths, not a second one. |
| `NYULO/` | the grid-extent control for `NYUCAP/` (Appendix~\ref{app:nyu}) | The same four widths and ten seeds as `NYUCAP/`, on a grid extended four decades lower (`α` from `10^{-6}` instead of `10^{-2}`, 17 points). It answers the one objection the crossing's inversion invites — that the half-height point is an artefact of where the grid starts. It is not: the extended grid returns the scatter to its anchor (`0.99`–`1.01` at `α=10^{-6}`, against `0.64`–`0.83` at `α=0.01`) and leaves all four crossings bit-identical to `NYUCAP/` (`0.1136`, `0.0689`, `0.0356`, `0.0239`), and the twelve shared α reproduce to `0.000e+00`. Manifest: `configs/jobs_NYULO.json`; driver: `code/cross_domain_mtl.py`. |
| `RHO/` | Appendix~\ref{app:online} | The per-step probe along the joint trajectory. HAR at `m ∈ {32,64,256}` and battery at `m ∈ {64,256}`, five seeds per cell, run with `--rho-trace`, which records `rho_u` every ten optimiser steps without acting on it (the controller is in monitor mode, so the training path is untouched). Each `xcap_<domain>_trace_m<m>.json` covers the doses that span the crossing (six on HAR, five on battery); each `xcap_<domain>_zero_m<m>.json` is the `α = 0` control, where the toxifier is inert and the probe nevertheless rises by the same orders of magnitude — which is why the appendix reads the rise as the primary's convergence rather than as toxicity. The monitor is verified not to perturb: the aggregate rows of these files reproduce the `BSCALE/` cells they overlap exactly (28 matched cells, `max\|Δ\| = 0` over metric, collapse rate, scatter and `ρ̂`). Analysis: `repro/scripts/rho_trace_analyze.py`. Manifests: `configs/jobs_RHO.json`, `configs/jobs_RHOZ.json`; driver: `code/cross_domain_mtl.py`. |
| `WARM/` | Appendix~\ref{app:warmup} | The probe read as a function of the warm-up length. The shipped calibration recipe (stfcal, `alpha=1e-9`, 40 epochs, ten seeds) at HAR and battery over five widths `32`--`512` and at NYUv2 over four, with **only** `--cal-warm-epochs` varied over `{1,2,4,8,16,32}`. The `t4` column is the shipped configuration and reproduces it exactly: `max\|delta\| = 0` across all `100` HAR and battery cells against `R3/`, and `rho-hat = 2.258/2.008/1.637/1.285` on NYUv2 against the values of Appendix~\ref{app:nyu}. The read rises with the warm-up --- five decades on HAR (`0.63` to `6.9e4` at `m=32`), a factor of `2.2` on NYUv2 --- so the probe exponent is a function of the read time. Manifests: `configs/jobs_WARM.json` (all four domains) and `configs/jobs_WARMN.json` (the NYUv2 slice, split out to run ahead of the much slower RadioML slice); driver: `code/cross_domain_mtl.py`. |
| `WARMU/` | Appendix~\ref{app:warmup} | Warm-up robustness of the calibrated gate. The shipped HAR repair recipe (`m=64`, 60 epochs, ten seeds, `joint`/`stf0`/`stfcal` in one run) at `alpha = 0.03` and `alpha = 100`, with the same sweep of the read time. At `alpha = 0.03` the gate stays shut in all ten seeds for `W <= 2`, and `stfcal` then collapses in `10/10` at the unfiltered joint metric; from `W = 4` on the gate opens in every seed and the collapse rate is `0/10`. At `alpha = 100` the gate opens for every read and the repair is independent of `W`. Manifest: `configs/jobs_WARMU.json`; driver: `code/cross_domain_mtl.py`. |
| `../repro/results/P03/` | `tab:reg` | Anti-collapse regulariser sweep at the Table-10 doses: VICReg and Barlow Twins on the shared representation over nine weights, the `joint`/`stf0`/`stfhard` reference arms measured in the same runs, and a fine weight grid around each clearing weight. Ten seeds per cell on HAR, battery and RadioML; the fine grid covers HAR and battery only, so the RadioML threshold is located only up to the coarse grid. Manifests: `configs/jobs_P03_reg.json`; driver: `code/cross_domain_mtl_reg.py`. |
| `../repro/results/TIME/` | `tab:overhead` | Rotated wall-clock matrix: five domains × four widths (`m ∈ {32,64,128,256}`), one file per cell, produced by `repro/scripts/time_overhead.py` with `--rotate` and `--wait-idle`. `tab:overhead` prints the UCI-HAR row; the other sixteen cells are the cross-domain check reported in the text beside it, and `repro/scripts/time_cross.py` re-derives both from these files. Every file records the device state before and after the run, the per-repetition medians, and the rotation position of each row, so the guards the table relies on are auditable rather than asserted. |
| `../repro/results/xcap*.json`, `xrep*.json` | `fig:unified`, `fig:battery_repair` | The unified-protocol runs behind Figure 1(b–e) and Figure 2. `xcap*` are α sweeps over widths `m ∈ {32,64,128,256}` in `joint` mode (the scatter curves and the capacity law); `xrep*` are repair contrasts of `joint`/`stf0`/`stfhard`/`stfsoft` at one supercritical dose per domain (the repair panels). Driver: `code/cross_domain_mtl.py`; the per-run settings are recorded in each file's `raw` records (domain, `m`, `alpha`, `seed`, `mode`). `xrep2_battery.json` is the NASA domain at `m=128`, `alpha=10000`, 20 seeds per mode, scored on log-SOH RMSE — the same metric as the battery column of `tab:conflict` — and is the source of both Figure 2 and the battery panel of Figure 1; it is **not** the MATR flagship of `tab:bmtl`, which reports SOH-RMSE% for the TSMixer+SSM model. |
| `BSCALE/` | capacity-exponent resolution (Appendix~\ref{app:proofs}) | The extended capacity sweep on the two domains the appendix extends: nine widths `m ∈ {16,24,32,48,64,96,128,192,256}` × the HAR and battery α grids × five seeds, one file per `(domain, width)`. Same driver, seeds, epochs and grids as the `xcap*` row below, and every cell the two grids share reproduces to floating-point equality (largest disagreement `2.2e-16`), so the extended grid is the published one at finer width resolution, not a second protocol. The appendix reads its bound on how much of the exponent is resolved from these files. Manifest: `configs/jobs_BSCALE.json`; driver: `code/cross_domain_mtl.py`. |
| `../repro/results/E2c/` | RadioML capacity exponent (Appendix~\ref{app:proofs}) | The three RadioML α passes that `xcap2_radioml.json` is pooled from, at `m ∈ {32,64,128,256}`: `cap` (α 1..1000, seeds 0–2), `cap2` (α 0.003..1, seeds 0–2) and `cap3` (α 0.1..0.8 densified about the transition, seeds 0–4), plus `E2c_radioml_conflict.json` for the contrast-type ablation. No single pass spans both the healthy branch and the collapsed floor, which is why the half-height rule needs them pooled: `repro/scripts/build_rml_sweep_forfig.py` pools them (precedence `cap3 > cap2 > cap`) into `xcap2_radioml.json`, and `repro/scripts/analyze_radioml_cap.py` recomputes the crossings and the exponent from these files alone. Manifests: `configs/jobs_E2c.json`, `jobs_E2c_rml2.json`, `jobs_E2c_rml3.json`; driver: `code/cross_domain_mtl.py`. |
| `../repro/results/SYN5/`, `SYN7/` | the capacity exponent `γ₀ = 2.43` and the β axis | The synthetic rank-schedule families. `SYN5/rank_m<m>_r<r>.json` holds the **fixed-rank** family (generator rank `r` independent of width; the two-head combined span can reach `min(m, 2r)`) at four widths × six ranks; `SYN7/beta<b>_m<m>.json` holds the schedule `r(m) = 32 (m/64)^β` at four widths × five β. Analysis: `repro/scripts/synth_rank_analyze.py` and `synth_beta_analyze.py`, which print the anchor sweep, the exponent at each anchor, and the absolute-rule control. |

| `XDRCURVE/` | `tab:doserank` (Appendix~\ref{app:general}) | The dose-to-rank curve. UCI-HAR at `m=64`, five seeds, `--rho-rank-max 64`, ten doses spanning `α ∈ [0.003, 100]`, each run twice: once in the representation's principal basis (the paper's surrogate) and once in the auxiliary-gradient eigenbasis. These files are every number in `tab:doserank`, including the rank the rule returns at each dose (`cal_rank_mean`), the gate (`cal_gate_mean`), and the two relative-scatter columns. They are also the source of the statement that the two bases move the rank by at most 0.6 directions **for the variance-shrinkage toxifier only**. The reported filtered scatter is the `stfcalrank` arm, not the fallback `stfcal`. Settings are recorded per run in each file's `raw` records; driver `code/cross_domain_mtl.py --cal-basis {pca,aux}`. |
| `XDRANK/` | probe-depth sensitivity (Appendix~\ref{app:general}) | The same seven cells at rank budgets `3` and `64`, the two runs of a cell differing in nothing but `--rho-rank-max`. Shows the cap is a binding constraint at `α=100` on HAR: `stfcalrank` reaches relative scatter `0.7992` at `R=3` against `0.9456` at `R=64`, while `joint` and `stf0` are bit-identical across the two budgets and `stfcal` ignores the budget entirely (`0.9933` both), which is the deployed fallback of `alg:stfcal` behaving as the comment beside it says. |
| `XDBASIS/` | basis portability of `tab:doserank` | The auxiliary-basis runs at `R=64` on HAR, battery and CelebA, including `har_a100.0_aux_r64.json`, plus `portcheck_har_r3_pca.json`. The portcheck re-runs one `XDRANK` cell through the `--cal-basis` code path and records the comparison: 76 fields agree to the last bit (`VERDICT: port is bit-identical`), so adding the basis selector did not perturb the principal-basis results that every shipped `stfcal`/`stfcalrank` number was produced with. |
| `LOWLAB/` | not cited | The semi-supervised test of whether minimal intervention converts into a primary-metric payoff. UCI-HAR with the benign reconstruction head, `α ∈ {1, 30}` × label fraction `{0.02, 0.05, 0.2, 1.0}`, ten seeds, `joint`/`stf0`/`stfcal`/`stfcalrank`. It does not convert: under label scarcity `stfcal` and `stf0` are indistinguishable (`0.8629 ± 0.0135` against `0.8627 ± 0.0148` at `f=0.02`, both above `joint`'s `0.8574 ± 0.0169`), and at full labels `joint` leads. The experiment is shipped because the driver's `--label-frac` flag is what produced it and the released behaviour is its `f=1.0` setting; the paper's minimal-intervention claim is about how much is deleted (`tab:doserank`), not about a metric gain, so this file set neither supports nor contradicts it. |

| `XDRBASIS_REAL/` | `tab:realbasis` | Six real HAR reconstruction runs: alpha 12/30/100 × pca/aux, width and probe depth 64, ten seeds. Rank-arm accuracy, relative scatter and deleted rank come from `results` and `raw`; fallback `stfcal` is basis-independent. |
| `../repro/results/SYN8/` | conditional joint capacity fit | 26 completed cells, ten seeds: beta=0 at widths 32/48/64/96/128; beta=0.25/0.5/0.75/1 at 32/48/64/96; beta=0.125 at 32/48; beta=0.375/0.625/0.875 at 32. `synth_beta_joint.py` reports conditional cell-level OLS intervals, with shared seeds and identical pivot curves; no resolved collapsed floor at widths >=96. |

## Manifest paths

`OPERATOR_AUDIT/` contains exact Hessian/rank checks, not training results.
`STFRMS/` contains exploratory CPU candidates with matched local baselines;
the short pilot and the 3-seed/60-epoch runs are distinct protocols. Neither
candidate establishes primary-task superiority; see `repro/ALGORITHM_RESEARCH.md`.

The `configs/jobs_*.json` manifests are the job definitions as they were run.
Their `script` values resolve inside this release (`code/` for the drivers,
`repro/scripts/` for the testbed and finance scripts), so a manifest can be
replayed from the repository root with `code/runner.py`. Their `out`/`log`
paths record the original cluster layout, which grouped runs by campaign
(`results/E2c`, `results/REG`, ...); the shipped tree consolidates those into
the directories mapped above, so treat this file, not the `out` path, as the
authoritative map from a result file to the table or figure it backs.

Five drivers ship in both trees — `cross_domain_mtl.py`, `moo_combiners.py`,
`real_aux_mtl.py`, `wf_conflict_ablate.py` and `wf_repro.py` — because `code/`
holds the experiment drivers while `repro/scripts/` is the directory the
released reproduction scripts import from, and the two roles were populated at
different times. The pairs are byte-identical in this release, so either copy
is authoritative; `md5sum` them if you want to check that rather than take it
on trust.

### Declared outputs this tree does not carry

Eleven of the `configs/jobs_*.json` manifests declare outputs the released tree
does not carry file-for-file; the rest are complete. The audit is reproducible:
`repro/scripts/audit_manifest_scripts.py` walks every manifest, resolves each
`script` against the tracked tree and checks each `out` basename against the
shipped files. It reports that **every referenced driver ships**, and that the
eleven exceptions — 393 absent `out` paths in total — fall into three groups,
none of which leaves a table or figure unsourced:

* **Consolidated into a per-domain file.** `jobs_P03_reg.json` (110 jobs) ships
  as the three `repro/results/P03/REG_*.json` files, one per domain, each
  keeping its per-seed records; `jobs_P02_moo.json` (129 jobs) ships as the six
  `repro/results/P02/*.json` files; the three `jobs_E2e_compose.json` outputs
  are inside `repro/results/P02/E2eF.json`.
* **Superseded by a shipped pass.** Four manifests declare the first calibration
  grid — twelve cells at four widths — and its six dose pairs: `jobs_R1.json`,
  the R1 block of `jobs_ALL.json`, and the same R1 block repeated in
  `jobs_R1R2.json` (whose other two jobs, the flagship `R2_*`, do ship);
  `results/R3/` replaces all three, at five widths. `jobs_battery_cap2.json` and
  `jobs_battery_cap3.json` declare battery capacity sweeps at four widths (the
  latter writing to `results/BCAP5/`); the nine-width `results/BSCALE/` replaces
  them, and neither declares anything the paper cites. `E2c_battery_cap.json`
  (declared by `jobs_E2c.json` and by its lite duplicate `jobs_E2c_lite.json`)
  is a subcritical battery sweep on that same four-width grid, likewise replaced
  by `results/BSCALE/`. The four real-auxiliary runs declared by both
  `jobs_E2c.json` and `jobs_E2c_lite.json` (`E2d_har_recon`, `E2d_har_subj`,
  `E2d_rml_recon`, `E2d_rml_snr`, five seeds each) are the five-seed
  predecessors of the ten-seed `results/R5aux/` files that `tab:real_aux` is
  computed from.
* **Not completed.** `jobs_WARM.json` swept four domains; HAR, battery and
  NYUv2 finished and RadioML did not, so its thirty RadioML cells are absent.
  Appendix~\ref{app:warmup} rests on the three domains that finished and makes
  no claim about RadioML.

The eleven flagged manifests, with the group each falls in, are `jobs_ALL`
(superseded), `jobs_E2c` and `jobs_E2c_lite` (superseded), `jobs_E2e_compose`
(consolidated), `jobs_P02_moo` and `jobs_P03_reg` (consolidated), `jobs_R1` and
`jobs_R1R2` (superseded), `jobs_WARM` (not completed), and
`jobs_battery_cap2` and `jobs_battery_cap3` (superseded).

Every directory a table or figure is computed from is mapped above and present.

## Capacity exponent

The real-domain exponents are read with one rule, and `repro/scripts/capacity_exponent.py`
applies it to the files above:

    python repro/scripts/capacity_exponent.py results/BSCALE
    python repro/scripts/capacity_exponent.py repro/results/xcap2_radioml.json

`rel_scatter` is normalised by the *same seed's* single-task scatter, so a
healthy cell sits at `1.0` and `α_c` is where the mean relative scatter first
crosses **`0.5`** (log-α interpolation between neighbours). A width whose
smallest α still reads below `0.5` never reaches the healthy branch; its
crossing would be an artefact of where the grid starts, so the script prints
and drops it rather than fitting it. The synthetic testbed is the exception —
its healthy level is not `1.0` by construction, so its anchor is read off the
curve (see the section below).

The tool reproduces every exponent the paper prints. From `BSCALE/`:
`γ_HAR = +0.3831` (`R² = 0.5962`, 95% CI `[+0.101,+0.665]`, n=9) and
`γ_bat = −0.0566` (`R² = 0.1560`, CI `[−0.174,+0.061]`, n=9); restricted to the
four main-text widths, `+0.6645` (`R² = 0.8563`) and `+0.0957` (`R² = 0.4301`).
From `xcap2_radioml.json`: `γ_RML = +0.0232` (`R² = 0.0036`, CI `[−1.158,+1.205]`,
n=4), with crossings `{0.1735, 0.3949, 0.2415, 0.2156}` and a max/min movement
of `2.28×` across the eightfold width range. Reading the threshold relative to
each curve's own first point instead (`0.5·rel0`) is *not* the paper's rule:
the anchors are `0.946`–`1.069` on HAR and `0.998`–`1.001` on battery, and the
variant gives HAR `+0.51` at four widths (`+0.35` at nine), battery `+0.099`
(`−0.056`), RadioML `+0.047` — versus `+0.66`, `+0.38`, `+0.096`, `−0.057` and
`+0.023` under the paper's rule.

## Why the ceiling row is four-seed

The ten-seed single-task ceilings exist too, but they are not interchangeable
with the four-seed ones. The SOH ceiling barely moves (MATR 0.5085 → 0.5030,
NASA 9.1709 → 9.1976), whereas the NASA RUL ceiling moves from 18.5 to 27.5 —
which would place it *above* the collapsed joint model's 17.5 and make the word
"ceiling" meaningless. The ten-seed RUL-only baseline is unstable; the
four-seed value is the one the table reports, and its caption states the seed
count rather than leaving it implicit.

## Synthetic testbed

`../repro/results/th_*.json` hold the synthetic capacity-law runs:
`th_critical.json` is the 17-point log-spaced α grid over six widths that
defines the synthetic testbed of Appendix C — the exponent quoted for that
testbed is read from it under the relative half-height rule, while the retired
absolute rule reads the same curves as γ = 1.53. The undiluted anchor
γ₀ = 2.43 is **not** from this file: it comes from the fixed-rank family, i.e.
the `SYN5/`, `SYN7/` row above. `th_capacity.json` and `th_alpha.json` are the
earlier nine-point grids. The order-parameter scans split by estimator of
Definition 3's ratio: `th_rho.json` (8 seeds) and `th_rho12.json` (12 seeds —
the run behind the ρ = 0.74 ± 0.19 quoted in Appendix A) apply the paper's norm
ratio ‖Π_c g_a‖ / ‖g_p‖, whereas `th_rho_v2.json` (8 seeds) applies a per-cell
mean-of-ratios variant that reads about ten times larger on the same seeds;
`collapse_phase.py --rho-def {norm,percell}` selects between the two.
