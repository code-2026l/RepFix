# Match offline work for the exact RadioML program

The anchored study confirmed the exact/16 candidate and its latency
at batches 1 and 128. To isolate the auxiliary contribution from
offline distillation, compare two exact/16 students on fresh seeds
9400–9415. Generate a joint-SNR teacher and a detached-SNR teacher
from the same healthy state, with identical labels, batches, Adam
state, rate 0.0001 and 24-epoch continuation. Give both students the
same fixed healthy layers, identical initialization and minibatch
order, 12,000 unlabeled targets and 36 distillation epochs. Their
architectures and serving work are identical. The only intervention
is whether SNR gradients reached the corresponding teacher encoder.

Evaluate 4,400 signals per seed using the previous fixed sampling
rule. Do not access those signals before both students are frozen.
Compare student accuracy with a paired 95% t interval over references.
Require a positive lower bound, noninferiority to unrestricted joint
at -0.25 points, zero class or threshold changes in both students,
and an independently verified copy of the healthy subgraph.
Report descriptive paired accuracy gains for signals below 0 dB and
at or above 0 dB, using the same test samples and no further selection.
Report the earlier isolated timing as a separate matched experiment
on the same architecture.
