# Confirmed small-budget diagnostic improvement

The new result is a reduction in reference-query cost for estimating a repaired
consumer's disagreement. It does not repair the consumer, certify deployment,
or establish a new covariance-direction mechanism. It supplements the manuscript's
separation of decodability, consumer decisions and preservation cost.

## Frozen confirmation

After exploratory development on six earlier endpoints, the estimator and its
criteria were frozen before risk outcomes on twelve existing ViT endpoints:
four fitted references/partitions, three continuation seeds each. The fitted
ImageNet backbone remains shared. Each new reference has1024 map-fitting anchors
and a separate1024-anchor monitor pool. Additional reference-query budgets are
32/64/128/256. Twenty monitor subsets measure each fixed endpoint's sampling
variation. Statistical replication uses four reference means, not240 resamples.
Official test inputs are never loaded.

For the repaired cheap logits Z, estimate the healthy-minus-cheap logit residual
mean/covariance on the monitor. Propagate this Gaussian residual through the
three fixed consumers on512 inexpensive current outputs, averaging64 draws.
Compare with direct disagreement counting on exactly the same reference queries.
An isotropic model keeps the same residual mean and covariance trace.

| Extra monitor queries | Direct MAE (points) | Covariance propagation MAE | Paired reduction,95% interval |
|---:|---:|---:|---:|
|32|4.773|0.509|4.264 [3.148,5.380]|
|64|2.964|0.502|2.462 [1.573,3.352]|
|128|2.172|0.501|1.671 [0.556,2.786]|
|256|1.519|0.486|1.033 [0.101,1.966]|

The predeclared primary32/64 criteria pass: at least20% MAE reduction and a
positive reference-level lower95% limit for the improvement. Reductions are89.3%
and83.1%. The actual maximum disagreements are6.20--7.96%, above the1% repair
acceptance limit. This estimates failed-repair risk accurately; it does not make
those repairs acceptable. These are saved-model risk confirmations, not newly
trained-model or benchmark-test confirmations.

## Strong controls and stress test

The isotropic model obtains0.508/0.499-point MAE at32/64 queries. Its difference
from full covariance is unresolved; the evidence does not support novelty for
covariance directions, nor prove that the true error distribution is isotropic.
The useful result is statistical propagation through the specified consumers.
Conditional-expectation variance reduction and Gaussian error propagation are
established tools; the claim here is their measured value for this update audit.

With fixed augmented validation views and clean monitor anchors, covariance MAE
is0.690/0.683 points against4.228/2.434 for direct counts. Their paired improvement
intervals are[2.277,4.800] and[0.355,3.147] points. At128/256 the augmented-view
intervals cross zero; no uniformly superior high-budget result follows.

An independent implementation reproduces every forecast and all three consumer
rates. Averaging absolute errors across the three consumers, rather than taking
their maximum, retains the benefit: clean32/64 MAEs are0.772/0.779 points against
3.488/2.514. Thus the result is not solely an artifact of maximizing noisy counts.
Zero-residual and common-offset checks also pass.

## What makes the direction useful, and what remains open

A small number of binary disagreement observations has high sampling variance.
The many cheap logits reveal how much mass lies near each consumer's boundary;
the residual model supplies perturbations. Combining these quantities can be
more precise than direct counting. This is an empirical, assumption-based
estimate. Identical marginal logit/residual distributions can yield opposite
disagreement under different dependence; the explicit binary counterexample in
RISK_PROPAGATION_MECHANISM_20261110.md rules out a universal guarantee.

The1024 healthy map-fitting outputs are an initial access cost, potentially
cached across subsequent updates. The32--256 budgets count additional monitor
queries.512 cheap current outputs and statistical processing also cost time;
end-to-end latency was not measured. Do not interpret incremental query savings
as equal total runtime savings or as a cheaper distribution-free certificate.
Prediction-powered inference is related established work on using cheap
predictions with a small exact sample: Angelopoulos et al., Science2023,
https://arxiv.org/abs/2301.09633 . That comparison is relevant for a subsequent
study of uncertainty calibration; this experiment reports point-estimation MAE.

The numerical jobs execute only saved-state CPU forwards. One metadata-only
queue-contract failure is retained and reissued without changing the numerical
recipe. All twelve final outputs are recovered with SHA256. Statistical analysis,
replay and writing are local. The independently verified manuscript release is
preserved while this new evidence receives its separate review package.
