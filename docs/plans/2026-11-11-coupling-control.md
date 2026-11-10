# Residual coupling and audit efficiency: mechanism follow-up

The fixed certified-audit comparison failed its query-efficiency target.
Diagnose this result rather than tune its strata on the same outcomes.

1. Record correlations between the fixed risk predictor and each binary error,
   and the oracle residual variance after linear control-variate correction.
   This diagnoses available variance reduction; it is retrospective and does
   not select a predictor or supply a new guarantee.
2. From each existing clean/augmented saved endpoint, uniformly select256
   validation examples with RNG7200. Center both cheap and reference logits;
   retain the empirical residual multiset exactly. For each of the three
   consumers form the256x256 binary disagreement cost of assigning residual j
   to cheap logit i. Minimum/maximum linear assignments give sharp empirical
   coupling extrema for that consumer under uniform marginals. The Birkhoff
   theorem/linear assignment are established tools, not new inventions.
3. Evaluate Gaussian propagation using that exact residual multiset under the
   paired, minimum and maximum assignments. The estimator must stay identical
   because its marginal inputs stay fixed. Verify all marginal means/covariances
   and the numerical forecast invariant. These are constructed permutation
   controls, not trained networks, future-population bounds or new labels.

No acceptance/advancement criterion is selected from this control. Report all
endpoints and all consumers, including narrow intervals. Its purpose is to
measure the dependence information discarded by the marginal error model.
The actual observed error remains separate from the constructed extrema.

All work is local CPU on recovered arrays. Preserve concurrent external edits
to the manuscript; commit only this research's explicitly named files.
