# Fixed-corpus, prediction-assisted audit

The previous Gaussian estimator has accurate MAE but no distribution-free
deployment guarantee. This follow-up tests whether its ranking improves exact
finite-population audit efficiency, and compares established model-assisted
mean estimators. It introduces no new claim to PPI, stratification or confidence
inversion. Preserve manuscript release6545564 and all previous scientific results.

All calculations are local on twelve recovered ViT output arrays, four fitted
references with three continuations. These saved models have been previously
studied; the new audit outcomes have not. This is a prospective analysis of
saved models, not a new model/benchmark confirmation. Clean validation is primary;
fixed augmented inputs are a stress test. No official test inputs or true labels.

Freeze the risk predictor before any audit sample: isotropic Gaussian residuals
estimated from the1024 map-fitting anchors' in-sample residuals, not new monitor
outcomes. Use64 common Gaussian draws (seed7100) on every cheap validation logit.
The model may be biased; it only chooses a sampling design. Its healthy anchor
outputs are already included in the1024 initial paired-map access cost.

Compare uniform sampling with four strata by predicted maximum consumer error,
at fixed population quantiles0--.50--.80--.95--1. Assign budgets64/128/256/512/1024
by a20% population-proportional,80% Neyman allocation using predicted Bernoulli
variance, with at least8 samples per stratum. Allocation uses predicted risk only.
Use nested random orders7100+replicate,20 audit repetitions per endpoint.
A fixed rank-reversed predictor is a stress control; it must not receive any
inferential validity from its Gaussian model.

For each stratum and consumer invert the exact hypergeometric distribution for
both lower and upper total error counts. Weighted sums cover full-corpus risk.
The uniform comparator uses the same full-corpus target and exact inversion.
Union allocation: total .05 over3 methods x5 looks x3 consumers x2 tails, with
an extra factor4 for the stratified methods. These are simultaneous bounds for
one audit run, not a joint claim over twenty repeated research simulations.
Candidate maps and strata are fixed independently of audit membership.

At budgets1%,5%,10%, reject only if a consumer lower bound exceeds the budget;
accept only if every upper bound meets it; otherwise abstain. Primary utility
target is correct rejection of the1% budget: mean first-decision queries must
fall>=20% versus uniform and reference-level paired95% saving interval(df3)
must exclude zero. Unresolved by1024 is recorded as2048 for this declared
censored-cost score, with unresolved fraction also reported. The exact observed
corpus risk determines correct/incorrect decisions only after inference.

For point estimation additionally compare direct counts, fixed-power PPI
mean correction, and a full-corpus survey-regression power-tuned mean with
lambda=clip(Cov(Y,q)/Var(q),0,1). The latter is the transductive mean form,
not a claim to reproduce all PPI++ implementations or its asymptotic intervals.
These are secondary comparators and never supply exact certificates. PPI/PPI++
primary references: https://arxiv.org/abs/2301.09633 and
https://arxiv.org/abs/2311.01453 . No new safety or superiority claim if the
fixed utility gate fails; do not retune on these outcomes.
