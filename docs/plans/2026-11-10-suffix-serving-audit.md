# Matched Suffix Serving Audit

After development utility selection, compare a monolithic frozen TorchScript
network and a modular frozen prefix plus frozen suffix TorchScript network.
Both current-only and protected implementations use identical float32 input,
pruning, prefix, folding, threads and preallocated/preprocessed inputs. Include
two independently called whole functions as a cost control. No duplicated
prefix work is added to the current-only baseline. Measure the stronger H
frozen-feature reader too: it may offer a better utility/cost frontier than
changing encoder weights, and its gain is not representation learning.

Development uses selected seed-3800 validation states, 12 CPU threads,
batches 1/128, 12 warmups and 31 alternating repetitions. Separately choose
the fastest absolute current and protected ABI for each architecture, U/H
and batch. Freeze this JSON before confirmation serving; do not select on
test timing or change the utility choice. Report all measured paths and
absolute times, not only ratios. The limits remain 1.20/1.10; report seed
paired uncertainty rather than treating timing repeats as independent
training replicates. Benchmarking is isolated from all owned training jobs.

For every fresh endpoint, reconstruct saved weights with state hashes. Audit
actual chosen protected code on all test inputs, relative to the original
healthy coarse function, using the same fresh monitor and alpha allocation.
Replay F/N/H/L (or M)/U on a common physically evaluated frozen prefix;
record differences from the original cached-prefix evaluation, including
near-tied classifications. Preserve online-head results, final primary
refits and raw timing samples. Input decoding/preprocessing, compilation
and warmup are outside forward latency and reported separately.
