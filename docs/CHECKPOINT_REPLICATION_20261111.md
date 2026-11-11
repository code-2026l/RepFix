# Visual checkpoint replication

We repeated the fixed CIFAR-100 cross-view continuation with the published
AugReg ViT-B/32 checkpoint. The fitted splits, images, normalization, label
masks, view pairing and optimizer settings match the TorchVision study.
Each AugReg reference uses the same supervised continuation and a second
arm with auxiliary weight 0.1. The four reference pairs use seed 6310.
[AugReg weights](https://huggingface.co/timm/vit_base_patch32_224.augreg_in21k_ft_in1k)
· [Fixed protocol](plans/2026-11-11-augreg-replication.md)

| Fitted reference | AugReg gain (points) | Contrastive loss ratio | Old class changes with the unchanged head |
| --- | ---: | ---: | ---: |
| 6201 | +0.72 | 0.914 | 7.76% |
| 6202 | +0.74 | 0.912 | 6.32% |
| 6203 | +0.76 | 0.920 | 7.02% |
| 6204 | +0.18 | 0.916 | 6.56% |
| Paired mean and 95% interval | +0.60 [0.15, 1.05] | 0.915 [0.910, 0.921] | 6.92% |

The retained FP32 reference preserves all 60,000 audited decisions across
classification and two fixed confidence rules. The largest difference from
the original unsplit encoder is 1.15e-5 in FP32 logits. Local replay
checks decisions and current accuracy from the saved outputs. The
intervals vary the fitted reference within CIFAR-100's training corpus.
The visual claim covers two checkpoints of the same ViT-B/32 architecture.

## Batch-128 CPU timing

The current-only and protected programs share the compiled FP32 prefix
and current suffix. The protected program runs the old suffix on the same
prefix activations. Four TorchVision endpoints use twelve threads and
thirty-one interleaved forward passes per program. Timing begins with
preprocessed inputs. [Fixed protocol](plans/2026-11-11-matched-batch-latency.md)

| Reference suffix | Latency ratio [95% interval] | Largest old disagreement |
| --- | ---: | ---: |
| BF16 | 1.078 [1.069, 1.088] | 0.18% |
| FP32 | 1.105 [1.092, 1.119] | 0% |

The BF16 program meets the 1.10× latency budget and preserves all current
classifications. The FP32 program also preserves every old decision. Its
latency interval extends above 1.10×. The timing result applies to the
original TorchVision checkpoint and the tested CPU.
