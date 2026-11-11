# Accurate mean forecasts do not identify which decisions change

The fixed prediction-assisted audit failed its efficiency target. Four fitted
ViT-B/32 references, each with three saved continuations, were audited on clean
and fixed augmented validation inputs. These are new analyses of saved models,
not new training runs or official-test confirmation. Reference-level intervals
average the three continuations and twenty nested resamples before inference.

| Inputs | Uniform queries | Stratified queries | Paired saving, 95% interval |
|---|---:|---:|---:|
| Clean | 78.67 | 129.60 | −50.93 [−63.43, −38.43] |
| Augmented | 76.00 | 123.73 | −47.73 [−70.05, −25.41] |

The target here is rejection of a 1% full-corpus disagreement budget. No run was
censored at that budget. At the secondary 5% and 10% budgets, many runs remained
unresolved after 1024 queries; the declared score codes these as 2048. That code
is a ranking penalty, not 2048 observed queries. The summary JSON separately
records every unresolved fraction. The initial 1024 paired mapping anchors are
an additional, cacheable reference-access cost.

Each stratum uses exact hypergeometric count inversion, with a union allocation
over methods, query looks, consumers and tails. Predictions choose strata and
sample allocation; Gaussian calibration does not confer validity. Across 1440
simulated audit runs, seven had at least one uncovered bound (seven of 7200
pointwise bounds). No certified decision was wrong. These are empirical checks
under the declared sampling design, not a claim of zero noncoverage.

Established mean correction gives a modest improvement under matched access.
At 64 queries, clean-input mean per-consumer MAE is 2.530 points for direct
counting, 2.374 for fixed-power PPI correction and 2.419 for the power-tuned
survey-regression form. At 1024 queries the values are 0.509, 0.470 and 0.470.
The power-tuned estimator is a transductive mean adaptation, not a replication
of every PPI++ implementation or its asymptotic confidence intervals.
See [PPI](https://arxiv.org/abs/2301.09633) and
[PPI++](https://arxiv.org/abs/2311.01453). This study's query population and
predictor differ from the preceding small-budget forecast study; their MAEs
should not be treated as a direct head-to-head comparison.

Two controls examine the distinction between mean prediction and localization.
The fixed fit-anchor predictor has mean squared correlation 0.127 with binary
consumer errors. Even an oracle Gaussian law fitted to all validation-reference
residuals reaches only 0.244 on clean inputs (0.256 on augmented inputs). The
oracle is retrospective and pays for all reference outputs; it is not an
operational low-budget method. Both residual-law estimation and missing
sample-level dependence remain plausible contributors.
Four strata also split tail probabilities more finely; the controls do not
attribute the entire query penalty to residual dependence.

For a separate coupling control, each endpoint supplies 256 fixed examples and
its exact residual multiset. Minimum and maximum linear assignments preserve
that multiset, hence all marginal moments, while changing which sample receives
which residual. The sharp per-consumer disagreement range averages 39.49 points
on clean inputs and 38.72 on augmented inputs; the narrowest ranges are 28.52
and 30.47 points. The marginal Gaussian forecast is numerically unchanged.
These extrema are constructed permutations, not trained-network outcomes or
future-population bounds. Extrema for different consumers need not share an
assignment. Linear assignment and coupling nonidentifiability are established
ideas; the contribution here is the measured control on these saved outputs.

The evidence supports a narrower statement: a useful corpus-level estimate can
discard the dependence needed for targeted auditing. It does not support faster
certification, safe repair, a new distribution-free estimator, or a universal
failure law. The manuscript's remaining empirical priority is a confirmed net
auxiliary benefit on a stronger visual backbone, with an independently replayed
legacy function and measured serving cost.

The standalone figure shows (a) reference-level audit-query means and four group
points, (b) clean-corpus mean-estimation errors, (c) class-consumer coupling
extrema and observed pairings for all twelve endpoints, and (d) predictor/error
correlations for four reference groups. Numerical source arrays, plans, scripts
and all outcomes accompany the evidence archive. All calculations, figure
rendering and writing for these controls ran locally. Concurrent manuscript
edits were preserved separately.
