# Fixed transfer of cached-anchor risk estimation to new auxiliary updates

The earlier CE-only endpoints are development for this control. Their cached
anchors have mean ridge leverage .741; leave-one-out isotropic propagation has
clean MAE .488 points versus .508 with32 extra reference queries. The paired
improvement interval includes zero. These outcomes select a standard statistical
control, not a new ridge/PRESS identity or an independent confirmation.

Before inspecting any repaired-consumer risk on the new contrastive endpoints,
freeze the following transfer check. Use all twelve weight-.1 confirmation
models, four existing fitted references and three continuation seeds each.
Their primary accuracies may already be known; their new alignment/risk outcomes
have not been exported or examined. This is prospective outcome-blind auditing
of new auxiliary updates, with shared reference fits and ImageNet pretraining;
it is not a new dataset or independently pretrained-backbone study.

Fit the same ridge map from current768-dimensional features to centered old
20-class logits, coefficient .001, on the same first1024 anchors (RNG6500).
Use the fixed-design leave-one-out formula and retain all original fit-anchor
controls. Gaussian propagation uses512 current-only validation outputs and64
common draws, seed6700, with an isotropic covariance estimate. This estimator
uses no other old-model outputs. Initial1024 paired outputs remain a cost.

Compare against the preceding32/64 additional-query residual estimators on the
other1024 anchor indices,20 fixed nested repetitions, with the original seeds.
Evaluate the maximum class/coverage80/coverage50 disagreement on5000 clean
validation examples. Clean is primary; the fixed augmented view is a stress
test. Export full reference logits only for evaluation, after all predictors
are fixed. No semantic labels are used for estimation or compatibility.

Primary transfer gate: the four-reference mean absolute error of zero-new-query
leave-one-out isotropic propagation is<=1 point, and the upper endpoint of its
paired two-sided95% t interval for MAE minus32-query isotropic MAE is<=.25 points.
Average three continuations and nested monitor resamples before reference-level
inference. Keep all methods and both views. The .25-point noninferiority margin
is fixed here using the CE development evidence before the new outcomes.
No changing the visual primary-benefit gate, no estimating a finite-corpus
certificate from leave-one-out residuals, and no claiming zero total access cost.

Reuse the native-audit feature forwards and add only anchor reconstruction and
ridge algebra. Numerical CPU execution only; all analysis/writing remains local.
