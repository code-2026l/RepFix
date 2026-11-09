# Protected-head integration and claim audit

This revision preserves the supplied experiments in commits `80320da` and
`da57cb4`, including all adverse results. It adds fixed retrospective controls,
independent numerical replay, corrected two-output timing, and manuscript edits.

## Scientific assessment

The HAR arm is a useful positive construction for approximate prediction
retention with one encoder. It does not establish exact retention of the old
function: maximum logit error is 4.05, and the original certificate requirement
passes 9 of 10 seeds. The paired primary difference is +0.44 percentage points
[-0.03, 0.91], while reconstruction MSE is 0.767 of detachment. Thus the
primary gain is not resolved, and the original all-seed conjunction fails.

The SNR arm loses 1.29 points [-1.78, -0.80] and certifies no seed. This does
not prove information erasure. The loss jointly trains the encoder and reader;
linear accessibility, encoder regularization, and optimization are separate
possible explanations. Distillation and Learning without Forgetting are
acknowledged as methodological precedents in the paper.

## Fixed retrospective control

The plan `docs/plans/2026-11-09-protected-head-audit.md` was fixed after the
original outcomes were known and before the new fits. All ten saved seeds per
corpus are included. No encoder or current-head updates occur. Both OLS and
ridge (coefficient 0.001) fit the healthy training logits on training features
without labels. No validation selection or further tuning follows.

A fresh monitor and total delta 0.05 cover 140 fixed readers. HAR protected
features plus OLS yield 1.44% mean disagreement and 10/10 new certificates;
unconstrained joint features plus OLS yield 3.35% and 1/10. On SNR the protected
OLS reader yields 4.04% and 0/10. These bounds do not replace the original 9/10
and 0/10 verdicts. The HAR result supports a contribution from joint encoder
regularization; it is not a fresh training confirmation or a general solution.

## Replay and cost

All 60 original legacy-consumer metric sets replay exactly from saved tensors.
A separate process reloads 120 fitted readers and 20 online heads without
refitting: all metrics and bounds replay exactly; maximum convex stationarity
residual is 8.50e-10. The numerical records are in
`repro/results/CPUCONT/protected_head_audit/` and
`protected_head_reader_replay/`.

The original timing closures measured the same current-only operation for two
paths. Those frozen records are retained. The corrected isolated benchmark
serves current and legacy outputs from one encoder call. At batch 1024:

| Corpus | online protected head / current only | float64 OLS / current only |
|---|---:|---:|
| HAR | 1.054 | 1.181 |
| SNR | 1.076 | 1.243 |

These ratios are means of per-seed medians with four CPU threads. Encoder and
online-head inference use float32, while the current-head diagnostic and the
stored OLS reader evaluate logits in float64. They price these implementations,
not an optimized deployment kernel. Parameters alone do not imply free inference.

One timing invocation initially matched progress snapshots as complete records
and stopped before calculation. The corrected wrapper filters those files;
the failed ledger is retained locally. All subsequent numerical jobs completed.
The queues are finished with no running jobs; peak aggregate RSS was 2.55 GB
in decimal units. All 80 original checkpoint files (112,267,980 bytes) were
retrieved locally and checked against remote SHA-256 digests. The compact
release excludes model tensors and raw datasets.

## Other corrections

- Vision bf16 seed-900 comparisons now use their own detached control; FP32
  three-seed means are reported separately.
- Pure-scale vision replay contracts standardized features about their healthy
  mean. It is no longer described as the raw-feature critical-scale test.
- Input-moment correction and new-to-old ridge are labeled by their implemented
  operations, rather than full AdaBN or the original LDC prototype update.
- Frozen source versions and the SNR-only protocol prefixes are retained with
  their recorded digests. Historical vision byte-representation hashes are
  verified using their original convention.
- The supplied plan filenames dated November 11 are retained as labels. Plan
  and result commits alone are not an independently registered timestamp.

## Next research gate

A stronger contribution needs a new task and backbone with independently
confirmed retained utility and decision compatibility, explicit access to
teacher logits or anchors, and measured two-output cost. The present pretrained
ResNet-18 screens did not reproduce the HAR accuracy failure. A new large run
is warranted only after a small development screen finds a useful operating
regime against ordinary distillation and post-hoc reader fits. The paper makes
no foundation-model or universal-scale-law claim.
