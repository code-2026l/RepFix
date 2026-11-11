# Does targeted computation matter?

The fixed confidence route retains more of the updated teacher than a uniform random subset using exactly the same number of additional suffix evaluations.

Across four fitted-reference groups, each averaging three continuations:

| Quantity | Mean [paired 95% interval] |
| --- | --- |
| Accuracy advantage over exact random-routing expectation | +0.633 pp [0.512, 0.755] |
| Student/teacher differences captured by the gate | 99.33% [99.06, 99.59] |
| Remaining teacher disagreement, fixed route | 0.041% [0.023, 0.058] |
| Remaining teacher disagreement, random expectation | 4.095% [3.848, 4.343] |

The random baseline is computed exactly: a uniform subset of k of n inputs includes every input with probability k/n. Expected accuracy is the corresponding mixture of student and teacher accuracy. Expected teacher disagreement is (1-k/n) times the original student/teacher disagreement. A four-input exhaustive enumeration checks the formula at every possible subset size.

This is a retrospective analysis of the same official CIFAR-100 predictions. It supplies an ablation, not an independent replication. No model, threshold or prediction was changed. Equal call counts match the number of additional suffix evaluations; they are not a wall-time match.

The contribution is the separation between preserving an old interface and recovering the benefit of an update under a measured cost. Keeping the healthy path explains old-function preservation by construction. Matched auxiliary controls establish learning gains, and this ablation shows that useful current-output recovery depends on which inputs receive extra computation. Distillation, side tuning and confidence cascades remain established ingredients.

Remaining research limits include the fixed checkpoint in the complete visual deployment result, broader deployment workloads, and the strength of conceptual novelty relative to existing backward-compatible learning methods.

Replay: `python repro/scripts/analyze_vit_routing_specificity.py --records repro/results/CPUCONT/contrastive_contrastivevitnativeaudit_export --arrays contrastivevitnativeaudit` from the numerical archive root.
