# Reuse a compressed legacy block across encoder updates

The strong CIFAR-10 update required 90% exact referral from a reader
attached to updated features. The prior affine and residual prefix
readers failed the 1% disagreement screen. Preserve the legacy block's
spatial residual and output gate while narrowing its intermediate
channels. Fit one compressed block to the healthy block's activations
and pooled features before fitting any of the six legacy consumers.
Both blocks take the same frozen-prefix tensor.

Train widths 64, 128 and 256 in FP32. Initialize channels by the product
of their teacher activation second moment and their second-convolution
kernel energy. Slice the teacher's two convolutions at those channels.
Then distill the old preactivations and pooled features for 12 epochs,
batch 128, AdamW 3e-4. Use both fixed views of 6,000 inputs sampled from
the original confirmation fitting pool with seed 7550. Scale pooled
feature errors by teacher channel standard deviations with a floor at
10% of their positive median. Add 0.25 times normalized preactivation
MSE. The target is the healthy block. Semantic labels do not enter
distillation. The one-time budget is 12,000 teacher-image forwards.

On the two earlier development references, calibrate a fixed fraction
of examples for exact referral using the lowest joint decision margins.
Use the first fraction in 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1 with
at most 3% maximum disagreement on each 512-example monitor. A full
reference action bypasses the student. Rank development candidates by
the mean of width/512 plus referral fraction, counting full reference
as one. Break ties toward the narrower student. Advance if both fitted
references require at most 50% exact calls and their mean proxy is below
one. Freeze the width and the full development table before opening the
new evaluation partition.

For each earlier confirmation reference, reconstruct the original
healthy consumer and strong joint continuation. Reproduce its sealed
epoch-two old-output action using the same repair anchors and monitor.
Calibrate the selected compressed block once on the 512 monitor inputs
using the fixed 3% rule. Save both actions before loading evaluation
images. Use disjoint slices of 1,500 examples from the official CIFAR-10
test batch, after fixed permutation seed 7590. The official batch
belongs only to this new experiment. Test inputs and labels stay out of
distillation, width selection, repair fitting, and action calibration.

Pair the previous early-selected terminal-feature policy and the new
prefix policy on the same strong updated encoder, consumer, and test
images, in batches of 128 with a final partial batch. Evaluate all three old decision rules against the exact healthy
suffix. Record exact calls, disagreement, current predictions, the
compression cost proxy, and the earlier policy's calibration access.
The access gate requires at most 5% disagreement, at most 50% exact
calls, and unchanged current decisions on every reference. The compressed
policy takes only frozen-prefix features, so its old-output function is
invariant to replacing the trainable suffix.

For an access-gate pass, time the physical image programs serially on CPU
at batch 128. Share the same FP32 prefix and current suffix for both
programs and include all routing operations. Compare with the
current-only and exact two-output controls. Use 12 warmups and 31
interleaved samples per program. Report the ratio of medians and 95%
paired t interval across the six reference ratios. A latency improvement
requires the upper interval endpoint below one versus the earlier
policy. Report access, proxy, and latency separately. All numerical jobs
use CPU. Writing, plotting, analysis and compilation stay local.
