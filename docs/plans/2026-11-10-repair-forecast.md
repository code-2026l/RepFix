# Retrospective repair-risk forecast screen

Goal: predict affine legacy-repair disagreement before executing the updated
encoder over an evaluation corpus. This is after an update, not a forecast of
a future training trajectory. Cached development activations allow a cheap
retrospective screen without training or official test access.

Use all six saved ResNet/ViT development endpoints. Freeze their existing
reference coarse heads and thresholds. Anchor permutation6500: first1024 fit
a direct centered-logit ridge reader (.001 standardized ridge); remaining1024
forecast error. Actual outcomes are5000 original validation inputs. Existing
outcomes were previously observed: this is exploratory, not independent
confirmation. No labels enter fitting, forecasts or selection.

Compare actual held-out consumer disagreement with (a) anchor in-sample error,
(b) direct anchor holdout error, (c) the holdout fraction whose reference
decision margin is <= logit-error oscillation. The last is the existing
pointwise sufficient-margin condition, used as a conservative empirical
forecast, not a population certificate. Also record standardized feature MSE
and centered-logit MSE. No free fit of calibration weights to these six outcomes.

Before inspecting results, require the geometry forecast to reduce endpoint
MAE by>=25% over direct holdout error and have absolute MAE<=.02, or stop it
as a proposed accuracy predictor. Conservative bounding is assessed separately;
an upper estimate is not automatically a better predictor. No deployable
success, cross-domain generalization or new method follows from this screen.

If geometry is uncompetitive, do not launch a larger confirmation. Preserve
records and use the result to narrow the next research question. All analysis
here is local CPU using previously recovered immutable states.
