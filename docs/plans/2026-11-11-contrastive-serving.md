# Cross-view benefit: fixed native serving follow-through

Only run if the development-selected weight passes the four-reference utility
confirmation. Keep failures and the full independent replay regardless.

Use the already fixed ViT U ABI choices in reference_precision_choice.json;
do not search a new implementation after seeing these cost outcomes. At batch1,
current is monolithic FP32, chosen reference is monolithic BF16-products-only,
and full precision is modular FP32 reference. At batch128, current is modular
FP32, chosen reference autocasts only the old suffix, and full precision is
monolithic FP32. Prefix and current path stay FP32 throughout. Keep the exact
healthy-function construction separate from its approximate served reference.

Audit all twelve confirmation endpoints from saved weights on the5000 clean
validation images with a physically rerun prefix. Compare current and detached
readers on that same prefix; keep the primary change from cached evaluation.
For each selected ABI, evaluate complete outputs in128-image chunks; also run
128 fixed individual inputs through the batch1 programs and report their
differences. Do not describe the latter as a full-corpus batch1 fidelity check.
Report class and two fixed-threshold decisions for full FP32 and chosen
reference programs against healthy FP32. A new uniform1024 monitor (seed7300)
gets alpha .05/(12 endpoints*2 variants*2 ABIs*3 consumers). The finite-corpus
remainder budget is5%, separate from the observed1% disagreement target.
All selected programs must retain identical current classification, with
FP32 current-logit error separately reported; otherwise no unchanged-current
claim. Report every disagreement and logit deviation, not only certification.

Measure native serving sequentially after owned training/replay queues finish.
Use the seed6310 endpoint of each of four fitted references,12 threads,
batches1/128,12 warmups and31 alternating repetitions of the three frozen
paths. Denominator is the fixed fastest current-only ABI from prior development.
Retain all absolute times/repetitions; average ratios within reference and use
a t interval across the four references. Limits remain1.20/1.10, including
upper interval limits. Model construction, preprocessing and tracing are timed
separately and excluded from forward latency, as in the preceding study.
This is one host and one pretrained checkpoint, not a hardware-wide guarantee.

Before any retrieval outcomes, add a secondary utility measure: bidirectional
top1 paired-view retrieval among all5000 validation images. Use the same fixed
healthy normalizer and unit-normalized suffix features as NT-Xent. Evaluate
joint and stored D on the two cached prefix views. This tests instance-pair
recovery, not semantic retrieval. Report four-reference paired differences;
retain all outcomes, and do not change any primary gate using this secondary
post-training measure.
Also apply the unchanged healthy coarse head directly to the updated suffix
features, as a descriptive need-for-protection control. Record its class and
threshold disagreement, negative flips and feature-trace ratio. It does not
enter implementation selection or the certificate family.

No official-test inputs, new labels for compatibility or GPU. Code, analysis,
figures, manuscript integration and builds stay local. Remote roots stay opaque;
CPU execution has no per-process RSS cap and retains the256GiB host reserve.
