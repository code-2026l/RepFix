# Resource-only preparation/training pipeline

Once only one twelve-thread preparation worker remains with no pending
preparation jobs, one twelve-thread training queue may overlap it. A shared
guard covers both opaque roots and enforces at most two workers and 24 GiB.
The combined thread cap remains 24. The training queue has one worker and
waits for each reference's complete record before loading its cache. Once
preparation ends, the same one-worker queue continues. Isolated timing still
requires every other numerical queue to have ended.

This changes scheduling only. Backbones, fixed data/reference, candidates,
controls, epochs, optimizer, selection, test access and cost criteria are
unchanged. The preparation-only healthy/H probe values are not used to tune
the fixed pilot or remove a backbone. Every declared pilot remains scheduled.
# Two numerical queues

After freezing the ResNet-18 choice, permit its one-worker confirmation queue
to overlap one one-worker final-attention preparation/training queue. Across
both roots the existing guard enforces at most two workers and 24 GiB RSS.
No timing overlaps numerical training. This changes scheduling only; all
gates, seeds, labels, epochs and outcome selection remain as registered.

