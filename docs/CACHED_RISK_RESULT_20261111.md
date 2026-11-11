# Disagreement forecasts from cached fitting anchors

The ridge repair already uses 1,024 pairs of current features and healthy
logits. Fixed-design leave-one-out residuals yield more accurate forecasts
than fitted residuals on the saved endpoints. The calculation holds the feature normalizer
and ridge penalty fixed and includes the unpenalized intercept in leverage.
The row-deletion identity was checked against independent refits.
[Fixed procedure](plans/2026-11-11-cached-anchor-risk.md)

On twelve saved supervised endpoints, mean leverage is 0.741. Clean
validation MAE is 4.85 percentage points for fitted decision counts,
1.09 for leave-one-out counts and 0.49 for isotropic propagation of
leave-one-out residuals. The established comparator using 32 additional
reference outputs gives 0.51 points. These outcomes guided the fixed
transfer protocol. [Supervised control](../repro/results/CPUCONT/cached_anchor_risk_20261111.json)

The transfer protocol fixes the estimator and gate before inspecting
repair risk on twelve cross-view continuations. The current models use
new auxiliary updates, with the same four fitted references and shared
pretraining. Each confidence interval uses four reference means.
[Transfer protocol](plans/2026-11-11-cached-risk-transfer.md)

| View | Cached leave-one-out MAE | 32 additional queries MAE | Paired difference and 95% CI |
|---|---:|---:|---:|
| Clean | 0.47 | 0.49 | −0.02 [−0.27, 0.22] |
| Augmented | 1.48 | 1.28 | +0.20 [0.10, 0.31] |

The clean result meets both fixed criteria. Its MAE is below one point
and the upper interval endpoint for the paired difference is below
0.25 points. The augmented result has higher error. Every row in the
comparison accounts for the original 1,024 paired fitting outputs.
[Complete transfer results](../repro/results/CPUCONT/cached_risk_transfer_20261111.json)

The repaired clean outputs still disagree on 7.92–9.30% of examples
under the worst audited consumer. Exact certification uses an
independently sampled monitor. The separate audit study records the
slower stratified certificate, the model-assisted estimation controls
and the constructed dependence counterexample.
[Exact audit and dependence](AUDIT_BOUNDARY_RESULT_20261111.md)
