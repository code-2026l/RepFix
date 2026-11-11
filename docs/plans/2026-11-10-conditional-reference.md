# Conditional Reference Computation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether consumer-aware selective reference execution preserves the
current learning gain and legacy decisions at a lower measured cost than always
executing the reference suffix.

**Architecture:** Fit an unlabeled cheap legacy-logit reader on frozen updated
features. Route inputs near the old consumers' decision boundaries, or predicted
to disagree, to the retained reference suffix. Primary outputs stay unchanged.
Compare boundary routing with confidence and learned disagreement baselines.

**Tech Stack:** CPU PyTorch, NumPy/SciPy, existing immutable suffix checkpoints,
TorchScript for actual serving, local analysis and LaTeX.

## Scientific positioning

This is a follow-up after the original vision test outcomes were opened. A
development-only screen precedes any routing test outcome. Distillation,
cascades and selective prediction are established components, not new inventions.
The question is whether the geometry of a *specified set of legacy decisions*
predicts the reference computation needed to preserve them, while keeping the
same FP32 current function. A positive result needs a matched conventional gate
comparison and actual latency; a smaller referral fraction alone is insufficient.

Directly overlapping prior work must enter the manuscript's related work:

- Duan, *Do Activation Monitors Survive Model Updates?* (2026),
  https://arxiv.org/html/2606.15980 : frozen-probe staleness and label-free repair.
- Schumann et al., *Backward Compatibility During Data Updates by Weight
  Interpolation* (2023), https://arxiv.org/abs/2301.10546 .
- *When Does Confidence-Based Cascade Deferral Suffice?* (NeurIPS 2023),
  https://papers.nips.cc/paper/2023/file/1f09e1ee5035a4c3fe38a5681cae5815-Paper-Conference.pdf .

## Task 1: Mathematics and meaningful implementation checks

Create `repro/scripts/consumer_routing.py` and
`repro/scripts/test_consumer_routing.py`.

For cheap logits z, use the minimum winning-class gap and absolute distance of
winning-class log-odds to either frozen confidence threshold. If this margin
exceeds the oscillation (max minus min) of reference-minus-cheap logits, all
three consumers agree. A common logit offset cancels. This is a conditional
pointwise statement; a learned estimate of oscillation is not a pointwise bound.
Use the existing finite-corpus monitor to certify the actual routed decisions.

An exact-reference fallback can change only predictions on the referred set.
Use the 7 nonempty error patterns for the three consumers to compute the oracle
minimum referral mass by linear programming. Report its assumptions explicitly:
unit reference cost, divisible population mass, fixed cheap reader. Batch latency
is measured separately. Exhaustive small cases check this bound and the margin
condition; a deliberately wrong residual estimate must not receive a certificate.

## Task 2: Fixed development screen

Create `repro/scripts/conditional_reference_cpu.py`. Reuse all six immutable U
development endpoints (three continuation seeds per backbone), the 2048 isolated
anchors, and 5000 validation inputs from the completed repair-control artifacts.
No new primary or coarse labels enter fitting or routing. Labels are used only
for descriptive scoring after candidates are fixed. The old affine head and its
two thresholds remain fixed.

The same seed-independent anchor permutation, RNG 6100, partitions anchors into
1024 reader-fit, 512 gate-fit, and 512 threshold-calibration inputs. Fit:

1. Affine ridge to healthy logits, standardized features, lambda 1e-3.
2. The same affine map plus a width-64 GELU residual MLP, AdamW 1e-3, weight
   decay 1e-4, batch 128, 100 epochs, zero-initialized output, logit MSE normalized
   by target variance. No validation-selected epoch or architecture.

For each reader evaluate four deterministic gates, all using no new labels:

- old-task confidence (refer low maximum softmax);
- distance to the nearest class or confidence-threshold boundary;
- that distance divided by predicted logit-error oscillation (log-residual ridge
  on the 512 gate-fit features, ridge 1.0);
- standard learned any-consumer-disagreement prediction (logistic L2 1e-2,
  maximum 100 LBFGS iterations, on the same 512 gate-fit examples).

Thresholds are score quantiles on the 512 calibration anchors at referral targets
0, .05, .10, .20, .30, .40, .50, .60, .80, 1.0. Ties refer conservatively; the
zero/all endpoints are explicit. Both approximation and reference outputs are
saved, and all 80 candidates per endpoint are reported. No oracle label selects
a gate. The LP oracle is descriptive only.

Advance a reader/gate/threshold family only if all three development endpoints
have maximum class/selective disagreement <=1%, mean referral <=50%, and
current predictions are unchanged. Rank by mean referral, then affine reader,
then gate order confidence/boundary/normalized-boundary/learned-disagreement,
then lower target referral. Report the best passing confidence baseline alongside
any consumer-aware choice, including when that baseline wins. If no family
passes, stop this route without test confirmation or favorable-example claims.

## Task 3: Conditional continuation

Only after a passing family is frozen, implement actual image-to-output dynamic
reference serving and verify batching/index restoration, all consumer decisions,
and current FP32 outputs. Confirm on held-out routing endpoints with unchanged
monitor budgets, and measure isolated latency against equivalently optimized
current-only, always-reference, and confidence-gated programs. Existing saved
confirmation endpoints are a retrospective serving follow-up, not new training.

Independent split/reference replication is a separate next study. Its partition,
label masks, seed hierarchy, control set and criteria must be fixed in an addendum
before computing those outcomes; no existing test score selects them. Do not
infer foundation-scale auxiliary learning from CIFAR supervised transfer.

## Execution and delivery

Author all files locally. Upload numerical code only into opaque UUID roots.
Use at most four 12-thread CPU workers, empty CUDA visibility, no fixed RSS cap,
and a 256 GiB host-available-memory reserve. Check current load, memory and disk
before launching. Keep at least 100 GiB disk free when adding new feature caches.
Pull useful records/logs/states promptly and verify hashes. Do not alter or delete
the previous release. Integrate only completed evidence, then locally rebuild
and verify the paper, figures and source/supplementary/submission packages.
