# Paired supervised training-work controls

The full matrix contains four fitted TorchVision ViT-B/32 references
and three continuation seeds per reference. Each of the twelve joint
endpoints is paired with both fixed supervised controls. Joint and D2
use ten epochs and two suffix evaluations per minibatch. D20 uses twenty
epochs and one evaluation. Each processes 200,000 training-image views through
the suffix. The frozen image prefix and final reader-fitting recipe are
shared. [Fixed protocol](plans/2026-11-11-compute-control-confirmation.md)

| Reference | Joint minus two-view CE (points) | Joint minus twenty-epoch CE (points) |
| --- | ---: | ---: |
| 6201 | +0.993 | +0.807 |
| 6202 | +0.340 | +0.320 |
| 6203 | +0.753 | +0.547 |
| 6204 | +0.813 | +0.433 |
| Mean and 95% interval | +0.725 [0.286, 1.164] | +0.527 [0.195, 0.858] |

Both accuracy intervals have positive lower endpoints and pass the fixed
joint success rule. The contrastive loss ratios are 0.791 [0.781, 0.800]
against two-view CE and 0.785 [0.777, 0.793] against twenty-epoch CE.

All 24 control accuracies and their paired differences are independently
recomputed from saved logits and matched joint outputs. The output labels,
reference and continuation identifiers, plan hashes and output hashes are
checked before aggregation. Intervals use four reference means, each
averaging three continuation differences on CIFAR-100 validation splits.
The official test split stays closed.

The budget matches training-image suffix evaluations. Joint additionally
computes feature similarities. D20 takes twice as many optimizer steps.
Elapsed times come from separate numerical executions under different host
loads. The result establishes positive gains against both specified
supervised controls within this pretrained checkpoint and training recipe.
