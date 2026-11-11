# Legacy precision at the same frozen prefix

The three narrowed FP32 blocks missed the development gate. Test
precision as a separate way to reduce the cost of the old computation.
Keep the earlier width-compression results and keep the official
CIFAR-10 test partition unopened until this selection is frozen.

Copy the healthy ResNet-18 final block, with its fused convolution and
normalization weights. Test three candidates in the fixed order
INT8 residual convolutions with FP32 skip addition and output gate,
INT8 convolutions with quantized skip addition and output gate, and
BF16 convolutions with FP32 skip addition and output gate. The primary
encoder, prefix and affine consumers stay FP32. INT8 uses the x86
quantized backend with its default static observers and per-channel
weight configuration. Calibrate the observers on 1,024 clean prefix
tensors from the same predeclared fitting pool and seed 7550. Candidate
weights and calibration are shared across legacy consumers.

For each candidate, reconstruct the two development consumers and
score the full approximate old output on their 512 monitor inputs.
Select the first candidate with at most 3% disagreement in every
class and threshold rule on both references, using zero exact
reference calls at inference. Retain all three candidate outcomes and
freeze the result before opening the official test inputs.

Use the earlier compression protocol's strong continuation, six
reference fits, epoch-two receipt, and six disjoint official-test
partitions of 1,500 examples. Reproduce the earlier terminal-feature
policy at 90% exact referral. Evaluate the approximate old block on
all inputs, with zero exact reference calls and the same FP32 current
output. Confirmation requires at most 5% disagreement in every
decision rule on every partition. Report all candidate- and reference-
level outcomes. Exact calls count FP32 old-block executions. The
selected low-precision block still executes on every example.

After the access gate, time the four physical programs at batch 128
on twelve CPU threads under the prior matched protocol. Use a shared
FP32 current graph and the same image batch for each method. The
primary latency comparison is against the 90%-referral program, with
a paired upper 95% interval endpoint below one. Also report the ratio
against current-only and the exact two-output program. The numerical
driver runs on CPU. Analysis and paper preparation run locally.
