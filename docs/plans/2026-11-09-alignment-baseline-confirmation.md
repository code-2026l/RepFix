# Alignment Baseline and Fresh Confirmation Implementation Plan

**Goal:** Test whether established joint feature alignment explains the current
positive construction, and independently confirm the previously retrospective
HAR reader fit only if its development gate passes.

**Architecture:** Reuse the frozen healthy development checkpoints from the
protected-head study, with identical optimizer moments and future minibatches.
Compare a jointly learned linear feature map with the same map trained against
legacy logits. Add training-only OLS fits to the existing direct-head candidates.
Freeze choices before any new confirmation seed is trained.

**Tech stack:** Locally authored PyTorch numerical drivers, bounded remote CPU
execution, local analysis, writing, vector figures and LaTeX. No GPU use.

## Design and scientific scope

The alternatives considered are a third-corpus repetition, an affine
recoverability diagnostic on saved features, and a matched strong-baseline
comparison. The last has priority: BC-Aligner already jointly trains a light
backward map and encoder (Hu et al., KDD 2022). Its single-update alignment
objective is the relevant baseline here. Model stitching and logit distillation
are also established methods. No new algorithm is claimed for these operations.

The current HAR OLS 10/10 result is retrospective. Original HAR 9/10 and SNR
0/10 counts remain unchanged. This plan is committed before the new baseline
outcomes; it is a public fixed protocol, not an independent registry entry.

## Task 1: matched development comparison

Create `repro/scripts/joint_alignment_cpu.py` and
`repro/scripts/alignment_objectives.py`, with numerical tests in
`test_alignment_objectives.py`.

Use original development seeds 800--802 on HAR and RadioML, and their saved
healthy model, readout and Adam state. Preserve original splits and training
order. HAR uses dose 100, 10% primary labels, and 30 continuation epochs after
the saved 60-epoch warm-up. SNR uses dose 1, 20% labels, and 25 continuation
epochs after the saved 25-epoch warm-up. Adam is 1e-3, batch 256, four threads.
Development uses validation only and never loads official test inputs.

Two new families use a bias-free 64x64 map M initialized to identity:

1. `feature_map`: primary + auxiliary + lambda * mean((M h - h0)^2).
   Grid: 0.01, 0.1, 1, 10, 100. This is the single-update linear alignment
   objective of BC-Aligner, divided by feature width. Its coefficient in the
   paper's per-example squared-norm convention is lambda/64. There is no
   historical composition at one update, and no multi-update claim is made.
2. `logit_map`: primary + auxiliary + lambda * mean((A M h + B - z0)^2).
   Grid: 0.1, 1, 10. M has exactly the same parameterization as family 1;
   the distinction is the alignment target. A and B are frozen healthy maps.

The original direct learned head at its original selected lambda (HAR 0.1,
SNR 10) remains the distillation comparator, rather than a newly named method.
Fit standardized float64 OLS to its final frozen training features and healthy
training logits, using the previous fixed SVD tolerance. Cast the resulting
reader to float32 for serving; no precision selection or coefficient search.
Report float64 versus float32 logit differences and prediction changes.

For each map family, select the smallest lambda with mean validation legacy
disagreement <= 0.05; otherwise select the largest. Neither accuracy nor
auxiliary gain is used in this selection. Inspect utility only AFTER selection.
No choice is revised if the utility gate fails.

For each selected map and direct-head-plus-OLS candidate, the development gate
requires every seed's empirical validation disagreement <= 0.05 and:

- HAR: mean reconstruction-MSE ratio <= 0.95 of detachment and mean primary
  accuracy within 0.005 of detachment.
- SNR: paired mean primary gain over detachment has a descriptive 95% t interval
  strictly above zero and is within 0.005 of the unconstrained joint gain.

Retain every candidate and adverse outcome. A failed family opens no test
confirmation. No extra lambda, different head or changed recipe follows.

## Task 2: independent training confirmation, conditional on the gate

Create `confirm_joint_alignment_cpu.py`,
`select_joint_alignment.py` and `verify_joint_alignment_cpu.py`.

For a corpus with at least one passing candidate, run fresh seeds 1300--1309.
These do not occur in existing released experiment records. Train new healthy
checkpoints; keep all splits, architecture, label budgets, doses, epochs and
minibatch pairing as above. Include detached and ordinary joint branches, the
original direct protected head at its fixed lambda, and each map family that
passes development at its frozen selected lambda. Always report the fixed
OLS-float32 reader on the direct protected features if that candidate passes.
No map fitting or reader fitting uses validation/test inputs or labels.

The fixed deployment bank is at most 160 consumers: two corpora x ten seeds x
four protected reader choices x two monitor budgets. Keep tail delta 0.05/160
even when fewer candidates confirm. The four protected readers are the original
online direct head, its OLS-float32 fit, selected feature map and selected logit
map. Detached/joint legacy changes are descriptive controls, not selected safe
candidates.

Use nested uniform monitors of 1024 and 2048 inputs, RNG seed
7390001 + seed + (100000 if HAR else 0), each capped at population size minus
one. Primary acceptance uses 1024, budget 0.05. The 2048 result reports an
additional monitoring-data cost and cannot rescue a failed primary criterion.

For HAR, success requires all ten empirical disagreements <= 0.05 AND all ten
1024-monitor finite-population upper bounds <= 0.05, reconstruction-MSE ratio's
paired 95% interval upper endpoint <= 0.95, and primary difference's paired
95% interval lower endpoint >= -0.005 relative to detachment. For SNR, require
the same compatibility criteria, positive primary-gain lower endpoint, and a
paired primary difference to joint whose lower endpoint is >= -0.005.

Both corpora additionally require one encoder call returning current AND legacy
outputs, no reference encoder at inference, changed nonlinear-layer tensors,
mean ratios of per-seed median latency <= 1.20 at batch one and <= 1.10 at
batch 1024. Benchmark only after training queues finish, with four threads,
five warmups and 51 alternating repetitions. Report parameter/storage counts.
Fold a learned map into the legacy head before serving, and record prediction
differences from the unfused arithmetic; no bitwise identity is assumed.

Independently reload every saved consumer and check model/optimizer hashes,
split/order hashes, metrics, bounds, OLS fitting conditions and serving precision.
The new confirmation does not overwrite any earlier study's decisions.

## Resource and publication controls

Smoke-check algebra and serialization locally before numerical launch. A smoke
uses limited fitting inputs, no test data, and is excluded from scientific
selection. At most two workers and four numerical threads per worker; at most
8 GiB RSS per job, 24 GiB aggregate, 900 seconds wall / 3600 CPU seconds per job
and 3600 seconds per campaign. Use opaque remote directories and empty CUDA
visibility. Keep authoring, aggregation, writing and rendering local; retrieve
numerical logs/states promptly and verify hashes. Do not touch unrelated work.

After results, update only claims supported by the comparison. A known baseline
that matches or improves the construction is a scientific result, not a reason
to hide that baseline. Rebuild the paper and packages locally, retain nine main
pages, and update the existing GitHub draft PR and organized delivery folder.
