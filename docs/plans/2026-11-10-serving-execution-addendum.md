# Serving Execution Addendum (before confirmation serving)

Keep every utility, consumer, latency and 31-repetition criterion unchanged.
All test inputs are still audited at both registered input batch sizes. Split
the numerical execution into an isolated one-worker timing queue and a
two-worker untimed full-output replay queue, each using 12 threads/job. They
run sequentially. This avoids serializing expensive correctness checks under
the isolated timing restriction. No outcome-dependent scheduling or stopping.

Confirmation times only the development-frozen current, protected and
independent ABIs, rather than re-running the backend search on test inputs.
All development timings, including slower alternatives, remain reported.
The frozen-feature H reader is exactly the same saved function in every
continuation seed. Memoize its timings and full output arrays after checking
reference, state, source, ABI and corpus fingerprints; compute each seed's
monitor certificate separately from those actual full outputs. Count H as
one unique model/timing case per backbone, not ten independent replicates.

Untimed replay may memoize the physically evaluated frozen-prefix tensor
for identical reference/source/corpus fingerprints. Every distinct U function
still receives full chosen-code evaluation on every test input at batch 1
and 128. No approximate feature cache, monitor reduction or imputed output
is used. Retain cache fingerprints, raw logits privately and public numerical
replay records. The 24-thread/24-GiB aggregate and 8-GiB/job limits remain;
all source writing, statistical aggregation and manuscript work stay local.
