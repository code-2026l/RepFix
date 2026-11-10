# Reference replication and failed auxiliary/routing screens

The new positive evidence concerns transfer across independently fitted ViT
references, label masks and fit/validation partitions. Four reference groups,
with three continuation seeds each, give +1.39 percentage points over the richer
frozen-feature probe H, 95% interval [0.80, 1.97]. Against each reference's highest
mean F/N/H control, the gain is +1.35 [0.88, 1.81]. Intervals use the four reference
means with three degrees of freedom. Shared ImageNet pretraining remains fixed.
Official test inputs stay closed; this is a training-population validation
confirmation with newly sampled partitions, not a new benchmark-test result.

The real visual auxiliary does not qualify. At weights 0.01/0.1/1, ResNet-18's
primary gains over CE-only detachment are 0.00/+0.06/-0.10 points; ViT's are
+0.12/+0.04/-0.52. Dose-matched reconstruction MSE ratios improve, but neither
backbone meets the joint primary/MSE gate. No grid expansion or auxiliary test
confirmation follows. The independent CE-only ViT confirmation is an explicit
fallback in the original protocol, triggered by its +2.38-point pilot gain over H.

Selective reference routing also stops at development. All six saved visual
development endpoints evaluate two readers, four gates and ten cutoffs. The
least mean referral satisfying all three consumers' one-percent disagreement
gate is 76.98% for ResNet and 78.40% for ViT, above the fixed 50% limit. Confidence
referral is 77.62% and 78.40%. The affine-reader fractional oracle is 43.67% and
29.39%; it fixes that reader and assumes unit fallback cost. It is not a latency
lower bound. No test or dynamic-serving claim follows from the failed screen.

The first ResNet auxiliary attempt is invalid because decoder optimizer moments
could alias across branches. A deep-copy correction, regression check and
unchanged warm-state hash accompany the reissue. Neither its numerical outcomes
nor a premature-cache ViT failure enters scientific selection. Missing-dependency
preparation failures are also retained. Data, seeds, losses, rates, grids and
stopping rules stay fixed. Exact executed source revisions remain available.

Separate CPU saved-state processes replay fourteen valid study records and
sixty-two branch outputs. All class and fixed-threshold decisions agree with
healthy references; maximum old-logit rounding error is 1.05e-5. Current logits
exactly match saved outputs during the independent CPU execution. Returned
decisions, legacy logits and per-image reconstruction errors are checked locally.
All 480 routing policies are independently recalculated locally as well.

Twenty-nine returned tensor files pass SHA-256 and finite-value checks: 472
tensors, including 365 floating tensors. Nine numerical queues are stopped.
The numerical workspace occupies about 18.33 GiB, mainly reproducible prefix
caches; 131.08 MiB of checked records and states are recovered locally. CPU
workers have no fixed RSS cap; the protocol retains a 256 GiB host-available
reserve and at most four twelve-thread jobs. No GPU computation is used.
Writing, analysis, figures, compilation and packaging are local.

The manuscript now directly cites activation-monitor staleness, functional
alignment pitfalls, backward-compatible weight interpolation and established
cascade deferral. Its scientific case rests on the scale intervention and
consumer-specific fidelity/utility/cost evidence, not priority for generic
staleness, distillation or unlabeled alignment. Broader real auxiliary benefit,
different pretraining and an effective online intervention remain open.

The 27th figure separates development screening from reference-level confirmation
and shows all continuation seeds. All 26 earlier figures retain their numerical
artist signatures. The manuscript remains nine main pages and 92 pages overall.
The compact source/supplement archives include complete records and executable
review checks; raw datasets, large caches and model tensors are stored separately.
