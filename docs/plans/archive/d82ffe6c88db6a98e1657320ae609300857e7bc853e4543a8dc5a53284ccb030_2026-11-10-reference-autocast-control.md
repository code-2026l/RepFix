# Reference-suffix Autocast Control (final precision implementation screen)

Manual reference-product BF16 passes both fidelity screens, but ResNet-18
latency is 1.38/1.106 at batch 1/128. ViT passes the cost screen at about
1.14/1.086. Whole-encoder AMP is a passing, stronger ViT development control;
its own matched current-only baseline and absolute times remain visible.

Test one final standard implementation: CPU autocast confined to the old
suffix, with the old affine consumer outside autocast at FP32. Current and
prefix stay FP32. Unlike manual product conversion this permits autocast's
activation dtypes and fused bias handling. First check every class/80/50
decision on the same 5000 validation inputs at the unchanged <=1% gate.
No thresholds, output corrections or labels tune precision. No additional
precision variants follow this screen.

Then isolated development timing compares the frozen fastest FP32 current
ABI, both original FP32 protected ABIs, manual-product reference BF16 and
reference-suffix autocast (monolithic/modular), plus independent functions.
Retain the matched whole-AMP ViT control. Use the same 12 warmups and 31
alternating repetitions. Select by absolute time among fidelity-passing
implementations; current remains FP32 for this choice. A separate AMP control
does not receive an FP32 normalization denominator.

Before confirmation, run the selected actual native code on all validation
inputs at batch 1/128 and check the original utility and interface gates.
Freeze the final choice publicly; only then audit ten selected endpoints
per backbone on all test inputs at both sizes with the original precision
plan's fresh monitor, alpha and budgets. Any failure is retained. Independent
AMP may be confirmed as an established control if it clears its own matched
utility, consumer and cost gates. Numerical queues stay within the original
CPU, RAM and disk limits, and timing remains isolated. All authoring is local.
