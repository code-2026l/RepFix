# Predeclared real-auxiliary benefit and consumer-compatibility study

This plan is fixed before the new development and confirmation calculations.
It follows the user's explicit request to reopen numerical work after the
previous bounded round. All authoring/analysis/figures/article compilation stay
local; only numerical execution uses the existing bounded CPU runner remotely.

## Construction and its price

For the healthy standardized affine head, let A = W0 diag(1/s0), and let P be
the orthogonal projector onto null(A). Evaluate the guarded representation as
h0(x) + P [h_theta(x)-h0(x)] (column-vector convention). Its healthy logits are
identical to the reference for every input in exact arithmetic. Training may
still learn directions outside the head's row space. This is an elementary
output-preserving projection control, not a claim to invent projection or
backward-compatible learning. Relevant primary literature includes Shen et al.
(CVPR 2020) and Farajtabar et al. (AISTATS 2020).

The frozen reference encoder is retained at inference; both encoder evaluations
and the projection are needed. Training can cache reference fitting features.
The unchanged current primary head and auxiliary use the guarded representation.
Report this cost, projector rank, numerical logit error and actual disagreements.
Do not call this a single-encoder drop-in repair or a cheap warning algorithm.

## Development, fixed candidate grid

- Two real objectives: reconstruction of the 561-dimensional HAR input, using
  10% of fitting primary labels; RadioML SNR regression using 20% primary labels.
- Same official HAR / audited prepared RadioML split and 20% independent
  calibration reservation as the earlier paired CPU studies. The calibration
  split is development validation here, not an exchangeability claim.
- Development seeds 200, 201, 202. HAR: warm 60, continue 30 epochs, doses
  1, 10, 100. RadioML: warm 25, continue 25, doses 0.1, 1, 10. Width 64,
  batch 256, Adam 1e-3. Use the original MLP and real auxiliary heads.
- Four branch types from identical healthy model/Adam state and future orders:
  ordinary detach D, ordinary joint J, guarded detach GD, guarded joint GJ.
  D/GD train their detached auxiliary at dose 1 in development (encoder/primary
  updates do not depend on this dose); J/GJ cover the fixed grid. No checkpoint
  selection: always use the declared endpoint. Confirmatory D/GD match the
  selected dose. All primary fits and replacement healthy heads see only the
  labeled fitting subset. Auxiliary targets use the fitting inputs/metadata.
- Development computes validation outcomes only, not held-out test metrics.
  Across each candidate's three seeds, compute mean J-D, GJ-GD and GJ-D current
  task accuracy. Choose the candidate maximizing the minimum of these three
  means. Ties within 1e-12 prefer HAR, then the smaller dose. Select once; record
  the whole grid and the selected configuration before any confirmation run.
  A nonpositive best development score does not authorize a new grid.

## Independent confirmation and success criteria

- Ten new seeds 300..309, one frozen selected dataset/label fraction/dose/recipe,
  all four branches. Score the full fixed test corpus only after the endpoint.
- Report paired two-sided 95% t intervals across seeds, conditional on the one
  fixed dataset/split/recipe. No endpoint or seed omissions.
- Full target requires all of: J-D lower interval >0; GJ-GD lower interval >0;
  GJ-D lower interval >0; GJ-J lower interval >-0.005; guarded fixed-consumer
  logit error <1e-8 and zero prediction disagreement on the scored corpus.
  These are a conjunction of criteria; report each separately. If only guarded
  auxiliary benefit is established, say so rather than claiming preservation
  of an independently established unrestricted benefit.
- A separate fixed bank of 40 confirmatory endpoint consumers uses a uniformly
  sampled 1,024-input unlabeled test monitor, total delta 0.05 and relative
  classification-error budget 0.05. Membership is independent of the trained
  classifiers. This bank does not inherit the earlier 3,015-candidate guarantee.
  Training uses no test monitor, labels, predictions or membership. Report
  bounds and abstentions for ordinary and guarded arms. NLL/ECE equality of the
  guarded consumer follows from preserved logits, not the disagreement bound.
- Log training-anchor consumer drift prospectively at fixed epochs, with a
  predeclared 0.05 prediction-disagreement threshold. This is a diagnostic trace,
  not a claimed warning lead time or a monitor-adaptive training guarantee.

## Resource and integrity controls

Smoke checks on small data verify implementation before the fixed development
grid. Keep smoke outcomes separate. At most four global single-thread workers,
8 GiB per job / 24 GiB aggregate, 1800 CPU s / 2400 wall s per job and 5400 wall s
per campaign, nice 10, host-memory and disk guards. No GPU. New remote directories
are opaque. Pull records/logs promptly. Preserve source/initial-state/order hashes,
complete model/Adam checkpoints, projector checks and the plan hash. Retain all
null/adverse results. No outcome-dependent numerical extension after confirmation.
