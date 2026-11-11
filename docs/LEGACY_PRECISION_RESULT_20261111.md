# Legacy precision under strong encoder updates

The three narrowed FP32 blocks miss the development gate. Widths 64,
128 and 256 require 80%, 70% and 50% exact calls on both monitors. The
width-256 student's convolution cost plus its referral cost equals one
full old block. [Width protocol](plans/2026-11-11-stable-prefix-compression.md)

Static INT8 convolutions with FP32 residual addition are selected by the
fixed development priority. The old block shares the frozen image prefix
and uses the original FP32 affine consumer. Calibration uses 1,024
unlabeled prefix tensors. [Precision protocol](plans/2026-11-11-legacy-precision.md)

Six disjoint partitions from the CIFAR-10 test batch supply 9,000
examples. Both actions are recorded before loading these images. The
prior policy uses 90.6% exact calls after rounding batches. INT8 uses
zero exact calls, with a worst class or confidence-rule disagreement
of 1.73% against the exact old block. The current FP32 outputs are shared.
The full INT8 old block still runs on every example.

| Batch-128 CPU program | Ratio to current-only | 95% interval |
| --- | ---: | ---: |
| INT8 old block | 1.043× | [1.036, 1.050] |
| Prior exact referral | 1.187× | [1.175, 1.200] |
| Exact FP32 two-output | 1.185× | [1.171, 1.199] |

INT8 takes 0.878× [0.874, 0.883] the latency of the prior referral program.
The paired ratio to exact two-output serving is 0.880× [0.875, 0.886].
Timings include the image prefix, current suffix, old branch and any
referral operations. The fidelity result uses six legacy readers on
the same shared prefix and quantized block.
