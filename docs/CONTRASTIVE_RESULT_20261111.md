# Cross-view auxiliary benefit on ViT-B/32

Cross-view training on CIFAR-100 improves fine-label accuracy by 0.84
percentage points over matched ten-epoch supervised continuation. The
95% interval across four fitted references is [0.66, 1.03]. Held-out
contrastive loss falls by 19.5%, with a loss ratio of 0.805
[0.794, 0.816]. Each reference contributes three matched continuation
seeds. The references share one ImageNet-pretrained checkpoint.
[Confirmation record](../repro/results/CPUCONT/contrastive_confirmation_20261111.json)

Only the last two attention blocks update. The primary task uses one
sampled view per example, with labels on 20% of the 10,000 fitting images.
The standard NT-Xent loss uses both views with temperature 0.2 and
adds no projection parameters. The weight of 0.1 was the smallest
qualifying weight across three development seeds. Independent replay
matched the pilot supervised weights and logits to the earlier baseline,
allowing reuse of the corresponding confirmation baselines.
[Frozen study plan](plans/2026-11-11-contrastive-benefit.md)

| Pilot control | Validation accuracy |
|---|---:|
| Ten-epoch supervised continuation | 70.94% |
| Two-view cross-entropy | 70.62% |
| Twenty-epoch supervised continuation | 70.92% |
| Cross-view auxiliary, weight 0.1 | 71.44% |
| Frozen-feature decoder, width 1,024 | 68.46% |
| Frozen-feature decoder, width 2,048 | 67.90% |
| Frozen-feature decoder, width 4,096 | 66.96% |

These controls use the same pilot seed and labeled examples. The
confirmation matches update steps and label access. Computing the
contrastive loss also requires a second view, and the pilot compute
controls address that difference on one split. The fixed reconstruction
screen and other earlier controls remain in the evidence index.
[Compute controls](plans/2026-11-11-contrastive-compute-controls.md)

Applying the original coarse head to the updated features changes
7.35% of class decisions on average. Replaying the retained FP32
reference yields zero observed changes in 72 class and threshold
checks. The selected lower-precision reference changes at most 0.18%
of decisions. Both variants preserve the audited current FP32
classifications, and all 144 consumer checks meet the five-point
remainder bound. A separate local replay reproduces the decision counts,
certificates and accuracies from the exported tensors. Paired-view
instance retrieval improves by 0.69 points in a secondary analysis.
[Output replay record](../repro/results/CPUCONT/contrastive_native_array_replay_20261111.json)

| Reference program | Batch 1 CPU ratio and 95% CI | Batch 128 CPU ratio and 95% CI |
|---|---:|---:|
| Selected lower precision | 1.139 [1.124, 1.154] | 1.098 [1.020, 1.176] |
| Full precision | 1.139 [1.128, 1.150] | 1.118 [1.018, 1.219] |

Isolated timing uses one endpoint per reference, twelve warmups and
thirty-one alternating repetitions on a Xeon Platinum 8570 with
twelve threads. The ratios compare forward latency with the fixed
FP32 current programs. Both reference variants meet the batch-one
limit of 1.20. Both intervals extend above the batch-128 limit of
1.10, so the combined serving gate remains unmet.
[Timing and fidelity summary](../repro/results/CPUCONT/contrastive_native_summary_20261111.json)

The validation results establish auxiliary benefit for this recipe and
pretrained checkpoint. Cross-backbone replication, a full matched
compute-control confirmation, and the batch-128 latency limit remain
open.

The initial compute-control dispatch invoked an earlier reconstruction
driver. That queue was stopped and excluded, and the intended controls
were rerun with the fixed study settings.
[Dispatch correction](../repro/results/CPUCONT/contrastive_compute_dispatch_exclusion_20261111.json)
