# Compiled Single-Encoder Compatibility Implementation Plan

> Execution: implement sequentially in the existing isolated local worktree,
> using the executing-plans review/check workflow. The user has explicitly
> authorized continued research and execution; no additional handoff is needed.

**Goal:** Determine whether the exact retained-affine-output control can be
folded into one encoder while retaining independently measured real-auxiliary
primary benefit, and measure the price of freezing the nonlinear trunk.

**Architecture:** Write the healthy encoder as h0(x) = B0 phi0(x) + b0, where
phi0 is the deterministic frozen nonlinear trunk and B0 is its last affine
map. With A = W0 diag(1/s0) and orthogonal P onto ker(A), train
B = B0 + P (Braw - B0), b = b0 + P (braw - b0). Fold B,b before deployment.
The compatible displacement is exact for every input in real arithmetic,
without an inference reference branch or an inference projection. This is an
elementary constrained-affine control, not a claim to invent nullspace
learning, backward compatibility, parameter merging or a general deep update.

**Tech Stack:** local Python/PyTorch/SciPy/Matplotlib/LaTeX; the existing
bounded remote CPU runner performs numerical training and independent replay.

---

## Fixed study, before new calculations

No new development grid, test-selected checkpoint or dose tuning is allowed.
The previous validation grid and ten-seed confirmation are already observed.
This is a new, separately declared follow-up, not a relabeling of that study.

- Two tasks: RadioML modulation classification + real SNR regression, 20% of
  fitting primary labels, dose 1; HAR activity classification + real 561-D
  input reconstruction, 10% labels, dose 10. RadioML reuses the previous frozen
  choice; HAR uses the positive max-min candidate in the previously archived
  three-seed validation grid. HAR was not selected for the earlier confirmation.
- Ten new seeds per task: 400--409. Retain the official HAR split and audited
  prepared RadioML split. Reserve a stratified 20% of the original training
  corpus before any model/head fit. The label mask is seed-fixed; unlabeled
  fitting examples participate in auxiliary losses only. No held-out test or
  validation input is used for training, healthy readout fitting, or folding.
- Width 64, original three-affine-layer MLP; batch 256, Adam 1e-3. HAR warm
  60 / continue 30 epochs, RadioML warm 25 / continue 25. Endpoint fixed.
- In every arm, the last encoder affine map and classification computation
  are float64; the nonlinear trunk and auxiliary head are float32. The healthy
  reference and healthy normalized consumer are defined in this same native
  mixed-precision interface before continuation. This differs explicitly from
  the earlier encoder-fp32 study and is not pooled with it.
- Six paired branches: full-network detach D / joint J; frozen-trunk,
  unconstrained-last-affine detach LD / joint LJ; frozen-trunk,
  constrained-last-affine detach CD / joint CJ. Clone the healthy model and
  complete Adam state for each branch, use identical future minibatch orders.
  Frozen parameters stay in the optimizer parameter ordering but do not update.
- The projection is fixed before continuation, based on the normalized healthy
  affine consumer, not just W0. Constrain the final bias as well as its weight.
  The deployed compiled state contains one trunk and one updated affine map;
  no healthy final-map copy or projector is retained in that encoder.
- Log a fixed 512-fitting-input diagnostic at epochs 0, 1 and every fifth epoch:
  changed legacy predictions, max logit error, threshold 0.05 disagreement.
  This always-on structural intervention does not establish adaptive warning
  performance or a warning lead time.

## Declared comparisons and decision rules

Report each task separately, all ten seeds. Two-sided paired 95% t intervals
describe seed variation conditional on each fixed dataset/split/recipe.
They do not establish broad population or architecture generalization.

For each task, the complete empirical target is the conjunction of:

1. J-D lower paired interval > 0 (a real unrestricted auxiliary benefit).
2. CJ-CD lower interval > 0 (a real auxiliary benefit inside the compiled control).
3. CJ-D lower interval > 0 (total benefit over ordinary detachment).
4. CJ-J lower interval > -0.005 (predeclared half-percentage-point tolerance).
5. CD and CJ preserve old logits to max error < 1e-8 and change zero test
   predictions; folding preserves training-time guarded features < 1e-10.

Report LD-D and LJ-J (freezing cost), CJ-LJ and CD-LD (constraint cost), and
LJ-LD as planned explanatory contrasts regardless of signs. A preserved
consumer alone does not pass the benefit criteria. If a task fails the
conjunction, retain and explain the failure without expanding the grid.

Additionally measure the deployment claim: one compiled encoder, identical
affine count and parameter shapes to the mixed-precision baseline, no projector
or reference parameters in the deployed encoder, and batch-1 / batch-1024 CPU
latency. Five warm-ups, 21 alternating timed repeats per path, median per seed;
report paired ratios and 95% intervals. Compare to the ordinary one-encoder path
and a two-encoder projection path of the same shapes. No throughput claim
outside the measured host/thread/batch scope; parameter precision is reported.

A separate fixed family of 120 consumers (2 tasks x 10 seeds x 6 endpoints)
uses delta 0.05, a uniform 1024-input test monitor independent of training and
budget 0.05. It is separate from both previous consumer families. Training
never sees the monitor or test outcomes. Report exact finite-corpus bounds,
abstentions and actual scored-corpus differences without calling abstention
observed harm. Complete logits, NLL and ECE preservation are a structural
identity, not a consequence of that classification-disagreement bound.

## Task 1: Algebra and meaningful implementation tests

Create `repro/scripts/compiled_readout.py` and
`repro/scripts/test_compiled_readout.py`. Verify normalized-head and bias
constraints on arbitrary inputs, Adam updates in the parameterization,
same-shape folding with identical features, absence of reference/projector
parameters after folding, useful gradients, and a negative trunk-drift control.
Run this targeted suite locally before real-data smoke execution.

## Task 2: Numerical driver and integrity

Create `repro/scripts/compiled_interface_cpu.py`. Reuse audited loaders,
masked primary fitting, normalized healthy head, consumer metrics and bounded
runner, while leaving the old training engine unchanged. Save complete
model/Adam states, compiled states, source/plan/split/mask/order hashes,
diagnostic traces, all validation and test outcomes, and deployment timings.
Small-data, one-epoch smoke runs only verify code and are excluded from claims.
Commit the fixed design before launching any such numerical calculation.

## Task 3: Independent replay and local analysis

Create `repro/scripts/verify_compiled_consumers_cpu.py` with an independent
folded forward reconstruction, and replay all 120 endpoint consumers.
Create `repro/scripts/summarize_compiled_interface.py` locally, validate all
source/plan/state/order identities, and report every criterion and contrast.
Never revise doses, seeds, label fractions, precision or endpoints from the
confirmation outcomes. Preserve all null and adverse results.

## Task 4: Article, figures and clean delivery

If evidence warrants, add a compact compiled-control result to the main text
and the exact construction, precision, cost and complete per-task contrasts in
the appendix. Otherwise document the failed target as a measured limitation.
Use the established perspective-plane vector style, with an explicit frozen
trunk and an inference folding step. Preserve old numerical evidence. All
authoring, statistics, drawing and compilation remain local. Render/inspect
the complete PDF, keep nine actual main-text pages, and deliver a new uniquely
named PDF/source package after a clean figure/test/LaTeX rebuild.

## Resource and execution limits

Only CPU numerical modules are uploaded; no article, plots or authoring tools.
Use new opaque remote directory names. At most four global single-thread jobs,
8 GiB/job and 24 GiB aggregate RSS, 1800 CPU seconds / 2400 wall seconds per job,
5400 wall seconds/campaign, nice 10, host-memory/disk guards. Pull useful
records/logs promptly and all model states on completion, verify SHA-256, and
check there are no remaining owned jobs. Never terminate unrelated work.
No GPU, GitHub push, or OpenReview action. No outcome-dependent numerical
extension after the fixed confirmation. User changes can authorize a later
separately declared study.
