# Mechanism audit and experimental improvement

Status: research candidate, not a validated performance improvement. Existing
published runs are preserved; new experiments use distinct output paths.

## 1. Capacity exponents are not determined by a rank-norm exponent alone

Write the local critical dose as alpha_c = restoration / contraction.
If restoration scales as m^a and contraction as m^b r^c, with r ~ m^beta,
then gamma = a - b - c beta. The undiluted value is gamma_0 = a - b.
The advertised gamma = gamma_0(1-beta) additionally requires c = gamma_0.
It does not follow from rank growth alone. Even if the contraction rate is
the squared toxic norm and that norm scales as r^p, c=2p only with all other
factors fixed. gamma_0=2p then additionally requires a-b=2p. Measuring
gamma_0 does not identify p without these assumptions.

For lambda_i = C i^(-(1+s)) with a rank-independent C and s>-1:

| Quantity over ranks 1..r | Scaling |
| --- | --- |
| Operator norm | C, independent of r |
| Trace, s>0 / s=0 / -1<s<0 | O(1) / log r / r^(-s) |
| Frobenius norm, s>-1/2 / s=-1/2 / -1<s<-1/2 | O(1) / sqrt(log r) / r^(-s-1/2) |

Thus even the same power-law spectrum implies different p depending on the
norm. For s>0 all three stay bounded, not sqrt(r). Width-dependent amplitude,
restoration spectrum, eigenvector alignment and covariance must be specified
before a width exponent can be obtained. The general spectral criterion uses
R^(-1/2) A R^(-1/2), so the spectrum of A alone is insufficient.

### Exact generator correction

The actual loss averages over both k heads and r auxiliary coordinates.
For batch-centered H, its sample-normalized Hessian is
A = 2/(k r) sum_j W_j W_j^T. With W entries of variance 1/r,
E[A] = (2/r) I. Hence changing r changes the curvature scale as well as rank;
norm preservation of W alone does not preserve the *averaged loss* gradient.
With k=2 the algebraic rank is min(m,2r), not r. At r=32 the m=32 and
m=64 points are saturated; the whole width series is not a fixed-effective-rank
experiment. Participation rank need not equal algebraic rank either.
`toxic_operator_audit.py` verifies the formula against autograd at 24 cells.

Changing --rho-rank-max only changes a probe; it cannot repair this generator.
For a clean follow-up use a single orthonormal subspace Q with a truly fixed
rank below the smallest width, explicit curvature amplitudes, and vary rank
and amplitude separately. Compare fixed eigenvalue, fixed trace and fixed
Frobenius normalizations. Report censored thresholds if collapse is unobserved.
The existing RS1 experiment is useful but cannot alone remove normalization
and effective-rank confounding.

## 2. Basis choice and a precise, implementable rank rule

For a linear squared-error decoder, g_a = D^T(Dh-x) (up to loss scaling).
Representation PCA sees Cov(h); the gradient second moment sees
D^T E[delta delta^T] D. These need not share eigenvectors. Positive decoder
Hessian D^T D does not alone imply a constant attractor: the input-dependent
forcing term D^T x also matters. Residual-error orthogonality is a hypothesis,
not established by a flat residual-rank curve. Measure energy captured by each
subspace, residual-error covariance, and decoder Jacobian before claiming it.
The auxiliary-gradient second moment is not the auxiliary Hessian.

Let v_i = g_ai/(||g_pi||+epsilon), C = mean_i v_i v_i^T, and eigenvalues
lambda_1 >= ... >= lambda_m >= 0. For an orthogonal rank-r deletion projector P,
mean_i ||(I-P)v_i||^2 = tr((I-P)C). By the variational eigenvalue principle,
the minimum is sum_{j>r} lambda_j, attained by top-r eigenvectors. Therefore
r* = min{r: alpha sqrt(sum_{j>r} lambda_j) < 1} is the minimum deletion rank
for this **empirical RMS bound**. Jensen upper-bounds the old mean residual by
this RMS quantity. This is not a proof of optimal rank for the mean residual,
the true toxic Hessian, parameter-space updates, or future training batches.

`stf_rms_experiment.py` implements this candidate. Gate and rank use the same
quantity. A capped unresolved probe falls back to full detachment explicitly.
The filter preserves forward h and applies (I-P) only to the auxiliary
encoder gradient. The primary branch and auxiliary-head input are unchanged.
The original arm filters forward features too; comparing the two is a causal
ablation, not just a basis comparison. The self-test checks forward identity,
gradient projection, residual-tail equality, Jensen, alternative subspaces,
zero-rank behavior and capped-probe fallback.

Alternatives: (a) generalized eigenvectors relative to primary curvature can
protect valuable primary directions, but require stable curvature estimates
and a metric-consistent projector; (b) a variance-barrier controller addresses
representation loss directly, but adds online feedback and needs separate
validation. Start with the RMS candidate because its stated objective and
implementation can be matched exactly; do not present it as automatically
superior or as a whole-trajectory safety certificate.

## 3. MATR: task error and collapse are different measurements

The published raw scatter values are joint=0.148642, stf0=0.1745,
stfcal=0.1194. The same-run primary-only anchor is 0.433528, yielding ratios
of means about 0.343/0.403/0.275. These are not mean per-seed normalized values.
More seriously, battery `rep_scatter` computes z.var(dim=1), variance across
feature coordinates within a sample. A constant encoder h(x)=c with unequal
coordinates has positive coordinate variance despite complete across-example
collapse. Neither the raw value nor dividing by its anchor fixes that issue.

Required new evaluation: between-example centered covariance, trace, normalized
trace against the same-seed single-task anchor, effective rank, prediction
variance, and SOH/RUL errors. The evaluation code must record which checkpoint
and axis it measures. Existing scalar logs cannot reconstruct these statistics.
Cap32 can show whether greater probe depth helps its recorded metrics, but
cannot settle representation collapse without new representation diagnostics.
Likewise the saturated-endpoint corollary describes a conditional possibility;
it is not evidence that MATR's measured operator is saturated.

## 4. Literature and defensible novelty

- [ATTITTUD, ICLR 2021](https://arxiv.org/abs/2108.11346): decomposes auxiliary
  updates by their influence on the primary task, using gradient subspaces.
  Subspace manipulation itself is established prior work.
- [GradOPS, WSDM 2025](https://arxiv.org/abs/2503.03438): projects against
  task-gradient subspaces to resolve conflicts. Different target from a
  verified constant-attractor failure, but projection alone is not novel.
- [Dimensional collapse, ICLR 2022](https://arxiv.org/abs/2110.09348): collapse
  and spectral mechanisms already have theory in contrastive learning.
- [VICReg, ICLR 2022](https://arxiv.org/abs/2105.04906): explicit variance and
  covariance regularization provides an essential anti-collapse baseline.

The plausible contribution is a verified conflict-free failure mechanism plus
a clearly scoped minimal-intervention objective and real-head evidence.
Do not claim all prior work lacks collapse theory or spectral analysis.
Empirical acceptance requires matched seeds, meaningful partial deletion,
preserved representation statistics, primary error, auxiliary utility, and
runtime, with failures retained. A pilot win is not a publication result.

The author's [DeTox/ProFS project](https://uppaal.github.io/projects/profs/profs.html)
identifies DeTox as an earlier version of the ICLR 2025 model-editing paper.
Its toxicity is language toxicity, not auxiliary-induced representation collapse;
compare the spectral technique without conflating the two meanings.
[Neural Collapse in Multi-Task Learning, ICLR 2026](https://openreview.net/pdf?id=M4t2JUMlfI)
studies class-feature/classifier ETF geometry, which is also distinct from a
constant encoder. It prevents claiming that MTL collapse has no prior theory.

## 5. Remaining theorem audit items (not solved by an algorithm gain)

- A largest generalized eigenvalue exceeding a threshold witnesses at least
  one contracting mode; it does not alone establish collapse of every mode.
  For diagonal R=I and A=diag(2,0.5), alpha=1 contracts the first coordinate
  and expands the second under dh/dt=(R-alpha A)h. Full collapse needs all
  active modes to contract, or an explicit nonlinear coupling argument.
- The minimal-rank operator-norm result is valid for its stated PSD operator,
  but relating it to the general restoration criterion requires whitened
  coordinates or R=I. Euclidean orthogonality changes under whitening.
- FT is not necessarily symmetric PSD for an arbitrary linear filter F.
  A singular-value norm need not characterize stability of a nonsymmetric
  dynamical system. Broad filter characterizations need stronger hypotheses.
- A batch concentration bound for one statistic does not identify a different
  Hessian eigenvalue or certify its evolution for the whole training trajectory.
- A scalar fold's standard deviation and the synthetic aggregate files cannot
  recover independent per-seed crossing uncertainty. Shared pivot curves must
  not be counted as independent replications in an inferential claim.

These are substantive scope issues. Resolving them, and the checkpoint-selection
protocol, is necessary before calling the manuscript submission-ready.

## 6. Completed tests and follow-up protocol

`results/OPERATOR_AUDIT/exact.json` records the exact generator audit. For fixed
r=32, m=32/64/128/256 gives algebraic ranks 32/64/64/64 but participation ranks
22.05/31.75/42.99/50.95 and operator norms 0.168/0.259/0.339/0.557. Neither
effective rank nor curvature is held fixed. `controlled_rank_scan.py` provides
a separate orthonormal generator with fixed-eigenvalue, fixed-trace and
fixed-Frobenius alternatives; its exact Hessian is tested independently.

SYN8 pivot deduplication (`synth_beta_joint.py --dedup-pivot`) retains 22 cells,
giving gamma_0=2.566, nominal CI [1.980,3.151] at anchor 3e-4. This is a
sensitivity check, not a cure for shared-seed or threshold-definition issues.

The one-seed 12-epoch RMS pilot was encouraging, but the completed 3-seed,
60-epoch CPU run at HAR reconstruction alpha=12 does **not** support promoting
the candidate:

| Arm | Accuracy | Auxiliary MSE | Relative legacy scatter | Deleted feature rank |
| --- | ---: | ---: | ---: | ---: |
| joint | 0.957018 | 0.014927 | 0.757634 | 0 |
| stf0 | 0.948648 | 0.023656 | 0.997227 | 64 (full detach) |
| stfrms | 0.944237 | 0.016307 | 0.890346 | 20 |

Source: `results/STFRMS/har_recon_a12_3s60.json`. These CPU results must not
be merged with published GPU/bf16 aggregates. RMS retains auxiliary utility
but loses primary accuracy to both matched baselines. Record this negative
result rather than choosing the favorable short run.

The next diagnostic variant, `--variant variance`, applies the minimum
Frobenius-norm correction satisfying <H_centered,g_aux> <= 0 on the current
batch. The correction is max(0,<H_centered,g_aux>)/||H_centered||^2 times
H_centered. Its forward pass is identical. This is one direction in flattened
batch space, **not** a rank-one feature projector; feature-rank and calibration
gate fields are consequently null in its JSON. Its bound is instantaneous
in feature space, not a guarantee under encoder Jacobians or Adam updates.

The battery follow-up uses `battery_representation_audit.py` at full probe
depth 128 with joint/stf0/stfcal/stfcalrank, one seed, 150 epochs, after the
currently occupied GPU becomes idle. It adds across-example covariance and
prediction-variance measurements at the driver's selected checkpoint while
preserving the original run and RNG stream. It is a diagnostic pilot; final
claims require a separate validation split and multi-seed replication.

The variance variant's completed 3-seed/60-epoch CPU result is accuracy
0.948309, auxiliary MSE 0.015792, relative legacy scatter 1.820051
(`results/STFRMS/har_variance_a12_3s60.json`). Against full detachment this
retains similar mean primary accuracy (difference -0.000339) and reduces
auxiliary MSE by 33.2%, but joint remains more accurate (0.957018).
Three seeds do not establish non-inferiority, and larger scatter is not by
itself better representation. This is a trade-off signal, not a winning method.
