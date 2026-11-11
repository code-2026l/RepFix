# Repair-risk forecast: exploratory result and stopping decision

All six cached visual development endpoints were evaluated under the fixed new
anchor split. No encoder training, official test access or remote computation
was needed. The result is retrospective: these validation populations and
endpoints had already been scored in earlier studies.

The direct centered-logit ridge reader is fit on1024 anchors; the other1024
anchors forecast the maximum class/selective disagreement on5000 validation
inputs. Mean absolute forecast errors across six endpoints are:

| Predictor | MAE, percentage points |
|---|---:|
| In-sample fitted-anchor disagreement |10.42|
| Direct holdout-anchor disagreement |1.54|
| Existing margin/oscillation sufficient-event bound |66.53|

The geometry forecast fails both predeclared requirements: MAE<=2 points and
at least25% improvement over direct holdout. It receives no larger confirmation.
A conservative sufficient condition is not a calibrated estimate of failure
probability. Its looseness must not be presented as evidence of information loss.

The simple holdout comparison is useful operationally and consistent with the
paper's existing finite-corpus monitor, but is not a new prediction algorithm.
These six runs share one reference per backbone; their MAE does not establish
cross-reference or cross-task prediction. No claim that risk is predictable
before training follows: the procedure observes paired activations after update.

Actual repaired maximum disagreements are28.14--28.44% for ResNet and
12.48--12.94% for ViT. Every endpoint still fails the original1% consumer gate.
The direct-logit fit and different anchor partition are a distinct exploratory
recipe, not a reversal of earlier complete feature-alignment controls.

The current manuscript and its verified submission package stay unchanged.
The scientific lesson is to require added value over a direct held-out consumer
check before treating a geometric score as a contribution. No larger run is
justified by this failed screen alone.
