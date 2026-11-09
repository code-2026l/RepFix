# Nonlinear Compatible Interface Implementation Plan

> Execution: sequential implementation/review in the existing isolated local
> worktree. The user explicitly requests another research round after the
> frozen-trunk result. This is a new hypothesis and separately declared study,
> not an extension of the earlier confirmation or a pooled analysis.

**Goal:** Test whether a trainable narrow nonlinear channel can retain real
auxiliary primary benefit and exact old-consumer outputs at less cost than a
full reference-anchored two-encoder control.

**Architecture:** Retain the healthy width-64 base f0 and add a narrow two-ReLU
MLP g of hidden width k. Emit h=f0(x)+P g(x), with P the orthogonal projector
onto the kernel of A=W0 diag(1/s0). Train the side MLP and current heads; the
base stays fixed. P g can learn nonlinear functions while A P g=0 on every
input in exact arithmetic. Fuse equal-depth base/side paths into one width
64+k MLP using stacked first weights, a block-diagonal second matrix, and a
concatenated final matrix [B0, P Bg]. This retains the frozen base weights
inside a wider encoder; it does not discard that computation or preserve the
original parameter count. Side-tuning, projection and branch merging are
established ideas; this is a diagnostic/compatibility control, not a new
adaptation algorithm.

**Tech Stack:** local PyTorch/Python/SciPy/Matplotlib/LaTeX; bounded remote CPU
training and independent saved-state replay only. No GPU, article uploads,
remote writing/plotting/compilation, GitHub push or OpenReview action.

---

## 1. Existing gaps and interpretation

The earlier two-encoder result establishes useful auxiliary learning with
unchanged affine-consumer outputs in one low-label SNR recipe. The compiled
frozen-trunk control establishes cheap compatibility, but loses that SNR
benefit; the matched unconstrained frozen-trunk control attributes most of
the loss to freezing. Neither result establishes a new optimizer, arbitrary
nonlinear-consumer compatibility, broad-architecture gains, online warning
lead time, or population generalization. This study targets nonlinear learning
and an explicit deployment budget; those other gaps remain separate.

## 2. Native precision, data and fixed recipes

- The entire feature encoder (base and side) uses float64 in every arm, as do
  classification logits and the healthy normalized head. The primary and
  auxiliary head parameters remain float32; auxiliary input is cast to float32.
  The healthy consumer is fitted to this native interface before continuation.
  All-double encoder arithmetic keeps fusion errors below the declared strict
  logit tolerance when matrix shapes change. This precision change prevents
  pooling with earlier fp32/mixed-precision confirmations.
- Original width-64 three-affine-layer/two-ReLU MLP, Adam 1e-3, batch 256.
  A stratified 20% training reservation occurs before any model/head fit.
  Only the seed-fixed labeled fitting subset supplies primary loss or healthy
  readout labels. All fitting inputs/available auxiliary targets may supply the
  real auxiliary. Validation is only for declared development selection.
- RadioML: audited prepared split, SNR regression, 20% primary fitting labels,
  dose 1, warm 25 / continuation 25 epochs. Side widths 16 and 32.
- HAR: official held-out-subject test split, input reconstruction, **1%** fitting
  primary labels, doses 1 and 10, warm 60 / continuation 30. Side widths 16/32.
  The lower label fraction is a new declared regime, not the previous 10% one.
  The mask is a fixed random subset with seed+501, as in the earlier studies;
  failure to cover every primary class is reported as an execution failure,
  never fixed by selecting a favorable seed.
- The side's first layer copies the first k healthy hidden units; its second
  layer copies the k-by-k healthy block and bias. Its final map and bias are
  zero at initialization. Thus every branch starts at the healthy feature
  function, and side prefix gradients emerge after the final map updates.
- Clone healthy model/Adam states for full-network branches. For side models,
  preserve the shared base/head Adam states in their original parameter order;
  new side parameters start with empty Adam state. Paired constrained and
  unconstrained side branches have identical side initialization and future
  minibatches. The frozen base parameters and their Adam states are verified
  unchanged. All endpoints and epochs are fixed.

## 3. Development: fixed grid and selection before confirmation

Development seeds 500, 501, 502. No development test inputs, labels or metrics
are evaluated. One healthy warm-up per task/seed is shared across candidates.
For each task/dose/width candidate use full-network detach/joint D/J, narrow
unconstrained side detach/joint UD/UJ, and constrained side TD/TJ. D and J are
shared across widths; D's primary trajectory is shared across doses (its
detached auxiliary uses dose 1), with dose-matched full J outcomes.

The four declared validation contrasts are J-D, TJ-TD, TJ-D and TJ-J. For each
candidate compute the minimum of their three-seed means, adding 0.005 to the
last contrast to encode the half-point tolerance. Select the maximum score
per task; ties within 1e-12 prefer smaller width, then smaller dose. Confirm a
task only if its best score is strictly positive. Freeze the complete grid and
both decisions in a locally committed JSON before any confirmation. Nonpositive
scores do not authorize grid expansion. Retain all null and adverse candidates.

## 4. Independent confirmation and conjunction of targets

For each selected task use ten new seeds 600--609 and the selected width/dose,
with ten paired branches:

- D/J: full-network detachment/joint continuation.
- CD/CJ: the earlier frozen-trunk constrained-last-affine detach/joint control,
  now under the same new native precision and healthy references.
- GD/GJ: full-network reference-anchored output projection, detach/joint.
- UD/UJ: narrow nonlinear side with no output constraint, detach/joint.
- TD/TJ: narrow nonlinear side projected into ker(A), detach/joint.

The full benefit/compatibility target per task requires: positive lower paired
two-sided 95% t limits for J-D, TJ-TD and TJ-D; TJ-J lower limit above -0.005;
TD/TJ logit error <1e-8 with zero changed old predictions; and fused/unfused
feature difference <1e-10. Report all criteria, not only the successful ones.
Intervals describe seed/mask variation conditional on this fixed split/recipe.
As a corroborating real-objective criterion, require the paired 95% upper limit
for TJ-minus-TD test auxiliary MSE to be below zero. This requirement is fixed
before the first development calculation; implementation-only smoke outcomes
are excluded from selection and claims.

Explanatory contrasts: UJ-UD (unconstrained-side auxiliary benefit), TJ-UJ
(constraint cost at matched capacity), TJ-CJ (added nonlinear-channel effect),
TJ-GJ (comparison with the full two-encoder compatible control), CJ-CD and GJ-GD.
Report current/old accuracy, NLL/ECE, auxiliary error, trace and rank for all arms.
No task, seed or checkpoint is dropped after seeing confirmation outcomes.

An additional deployment budget target requires the **upper** descriptive 95%
limit of the mean per-seed median fused/ordinary CPU latency ratio to be below
1.6 at batch 1024, and fused/final-head latency below the measured full GJ
two-encoder path. Record all ratios even if the target fails. At batch 1 and
1024 use five warm-ups and 21 alternating timed repeats, one CPU thread.
Time the encoder plus current head in every path. Report widths, dense deployed
parameter counts/bytes, explicit base retention, and the block-diagonal zeros;
do not call the wider model the same-size original encoder.

The separate fixed confirmation consumer family reserves 200 candidates
(at most two tasks x ten seeds x ten endpoints), delta 0.05, independent uniform
1024-input test monitor per seed, and relative classification-error budget
0.05. Use delta/200 even if one task is not selected. Training/selection never
uses test-monitor information. The bound covers this finite family/corpus, not
new populations or future unconstrained training. Preserved logits/NLL/ECE are
structural, not consequences of the disagreement bound.

Fixed fitting-anchor diagnostics at epochs 0,1 and every fifth epoch use a
0.05 disagreement threshold. This is an always-on structural control, not a
new claim about an adaptive warning policy or warning lead time.

## 5. Implementation and meaningful checks

Create `repro/scripts/nonlinear_interface.py`: native double encoder, narrow
channel construction, common-Adam extension, exact equal-depth branch fusion.
Create `test_nonlinear_interface.py`: arbitrary nonlinear side changes preserve
the normalized affine consumer, multi-step Adam changes the nonlinear prefix,
fusion agrees for arbitrary inputs and two widths, all dense zero blocks/base
weights retain their values, full-rank consumer blocks updates, and an
unconstrained or base-drift negative control breaks the identity.

Create `nonlinear_interface_cpu.py` with immutable source/plan/split/mask/state/
minibatch hashes and complete model/Adam/deployed states. Small real-data smoke
jobs use one epoch and reduced data; their outcomes are excluded from claims.
Source authoring stays local. Commit tested numerical implementation before
development; commit the fixed selection before confirmation.

Create `select_nonlinear_configuration.py` locally; refuse overwriting the
choice or selecting again after confirmation. Create independent CPU replay
`verify_nonlinear_consumers_cpu.py`, reconstructing the full saved family from
tensors without using the experimental fusion/feature helpers. Local analysis
`summarize_nonlinear_interface.py` validates all states and every planned
contrast, reports failures and costs, and generates manuscript tables/macros.

## 6. Article review, visual presentation and delivery

Preserve old positive/negative results within their original scopes. Add new
evidence only at the strength it supports. Keep the main contribution centered
on diagnosing readout validity and its intervention costs, not algorithmic
novelty. State cross-task and prospective-policy limitations. Render the
nonlinear-learning/legacy-output/deployment-cost comparison with all seed points
and use the established editable perspective-plane design for any mechanism
diagram. All plotting/statistics/writing/LaTeX stay local. Verify nine actual
main-text pages and all references; deliver a fresh source/PDF artifact after
an independent figure/test/LaTeX rebuild. Do not overwrite previous deliverables.

## 7. Resource limits and stopping rule

Only numerical CPU modules are uploaded to new opaque UUID directories. At
most four global single-thread jobs, nice 10, 8 GiB/job and 24 GiB aggregate
RSS, 1800 CPU s / 2400 wall s per job, 5400 wall s per campaign, host-memory and
disk guards. Pull records/logs promptly and completed state files, verify their
SHA-256, and check for no remaining owned jobs. Never affect unrelated jobs.
No outcome-dependent expansion after the declared confirmation. A later user
request can authorize a separately declared study. Partial success, adverse
results and unmet latency targets must be reported plainly.
