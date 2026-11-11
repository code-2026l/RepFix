# Small-budget consumer-risk estimation

Question: can combining cheap served-logit geometry with a small sample of
old/new logit residuals estimate disagreement more accurately than directly
counting monitor errors at equal healthy-reference query cost?

The previous worst-case oscillation bound is rejected. This study instead tests
distributional error propagation, an established statistical idea. It observes
an already updated model and does not predict future training or information loss.

Exploratory development: all six recovered vision endpoints, same1024 map-fitting
anchors and1024 isolated residual-monitor anchors (permutation6500). Direct
centered-logit ridge .001 is fixed. Budgets32/64/128/256, twenty random monitor
subsets per budget (RNG6700+replicate), shared across all estimators.

Predict the maximum of the three consumer disagreement rates. Compare direct
monitor counts, Gaussian residual propagation with10% covariance shrinkage to
isotropic, and isotropic Gaussian propagation. The latter uses the same residual
mean and trace. Use512 uniformly sampled cheap validation logits,64 fixed
Gaussian draws; average decision changes without accessing healthy validation
logits. Actual5000-input healthy decisions are used only after predictions.
No calibration coefficient is fit to validation outcomes. All grids are retained.

Advance only if the covariance estimator reduces overall MAE by>=20% versus
direct counts at both32 and64 queries, and does not worsen average MAE at128/256.
Report gains against isotropic propagation separately. Six endpoints share one
reference per backbone; resampled monitors are not independent trained models.
Passing triggers a frozen confirmation on the twelve existing independent fitted
ViT references/continuations, not a new training or benchmark-test claim. Failure
stops this distributional estimator and preserves its outcome. No posterior
retuning on the same outcomes. Current paper and final packages stay unchanged.

All development analysis is local CPU. If needed, CPU host is used only to
reconstruct saved confirmation activations under existing resource policy.
