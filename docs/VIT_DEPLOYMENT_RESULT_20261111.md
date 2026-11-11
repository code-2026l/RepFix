# Exact visual compatibility with selective current computation

The fixed visual program reuses the healthy ViT-B/32 representation for
the old coarse and fine consumers. A 128-unit student predicts the
current fine output. Low-margin inputs additionally use the updated
two-block teacher suffix, sharing the same frozen prefix.

## Official CIFAR-100 test

Four fitted reference groups each have three continuations. All 10,000
official test images were held out from fitting and model selection.

| Contrast | Accuracy change in percentage points, paired 95% interval |
| --- | --- |
| Routed joint vs routed two-view supervised | 0.86 [0.68, 1.05] |
| Routed joint vs routed doubled-epoch supervised | 0.60 [0.41, 0.79] |
| Routed joint vs direct supervised MLP | 2.23 [1.93, 2.54] |
| Routed joint vs unrestricted joint teacher | -0.023 [-0.044, -0.003] |
| Pure student vs unrestricted teacher | -0.97 [-1.14, -0.79] |

The teacher receives 32.2% of real inputs. All audited old
class and confidence decisions agree. Current predictions disagree
with the unrestricted teacher on at most 0.06% of each
program's test corpus. This bounds the accuracy difference on that
corpus without knowing its labels.

## Physical CPU execution

| Batch | Complete FP32 forward ratio to original current-only teacher |
| --- | --- |
| 1 | 1.046 [1.019, 1.074] |
| 128 | 1.032 [1.005, 1.058] |

The final implementation uses a scripted conditional branch. Its
outputs match the prior implementation on all twelve programs and
the same 10,000 inputs. Batch-one timing averages 32 fixed images.
Each timing run pins twelve distinct physical cores on one NUMA node.
A first padded implementation missed the confidence-interval limits.
The source also retains failed affine and nonlinear teacher-retention
screens and the excluded overlapping timing run.

The routing percentile was calibrated on training-only validation
references before official predictions. The official evaluation uses
one pretrained checkpoint and one benchmark. The pure student and
the routed program are separate operating points. The routed program
can evaluate a second learned suffix on selected inputs.

The reproduction commands are in the numerical archive. Inputs,
frozen base features and historical teachers can be reconstructed
from the source and public datasets. The numerical archive contains
the compact predictions, fitted readouts and source hashes.

[Equal-call routing control](ROUTING_SPECIFICITY_20261111.md): fixed routing gains 0.63 points [0.51, 0.75] over the exact random expectation and captures 99.33% of student/teacher decision differences. This is a retrospective fixed-prediction analysis.
