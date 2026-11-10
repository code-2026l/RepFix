# CPU scheduling addendum before full joint development

This changes scheduling limits only. Scientific objectives, data, optimization,
seeds, gates and all previous outcomes are unchanged. All tensor computations
remain on CPU with four PyTorch threads per small-model job.

A read-only inventory before launching Stage B found 112 allowed logical CPUs,
load averages about 22--26, and more than 2 TiB available RAM. Completed Stage A
peaked at 2.26 GB aggregate with two workers. A synthetic step benchmark used
no scientific inputs or accuracy selection; it does not justify changing the
planned four-thread numerical recipe.

Allow at most six concurrent small-model workers (24 numerical threads) for
joint development and later small-model confirmation/replay. Keep the existing
8 GiB per-process, 24 GiB aggregate, 4 GiB campaign disk, CPU-time and wall-time
limits. Require at least 48 GiB host-available RAM, at least 48 allowed logical
CPUs and host one-minute load below half that count at launch. Stop on the
existing host/aggregate limits. Earlier queues remain unchanged. Visual work
still has at most two eight-thread workers; serving timings remain isolated
with one worker after all training queues finish. No GPU calls are introduced.
