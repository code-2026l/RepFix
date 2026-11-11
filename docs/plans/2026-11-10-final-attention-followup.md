# Final Attention Follow-up (before this scope's outcomes)

The three-epoch final-map screen, reader-refit audit and ten-epoch complete
MLP screen failed on ViT-B/32. Preserve each failure. A final CLS MLP cannot
change which patch features attention supplies. This follow-up tests that
specific scope restriction, using one complete final transformer block. It
does not retune the preceding studies or claim that their gates passed.

Use the identical ImageNet-V1 weights, CIFAR-100 fitting/validation/anchors,
8705 labeled fitting inputs, healthy fine/coarse heads and fixed reference.
Cache the 50 float32 tokens after the first 11 blocks, before the final LN1.
No earlier weights or normalization parameters change. Last-block candidate
U updates QKV, attention output, and both MLP affine maps, including biases.
M updates only its two MLP maps, with attention frozen. F,N,H are the same
head-only, residual-head and richer frozen-feature controls. H reads healthy
features and the final healthy MLP's pre-projection GELU features.

Use exactly the ten-epoch, two-view, explicit-head-L2, AdamW and final-refit
recipe in 2026-11-10-nonlinear-suffix-followup.md. No learning-rate, rank or
epoch grid. Keep the actual original healthy CLS output for comparison;
pruned attention may reorder float32 operations. Only the final CLS query
and CLS MLP are needed, since no later attention consumes other outputs.
Apply this pruning equally to the current-only and protected paths and
check its equivalence to full attention on an independent tiny ViT.

Pilot seed 3800; only U is eligible. Require >=2 points over healthy/F,
within 0.5 points of N,H, and each old consumer changes <=1%. If it passes,
run the same complete grid at 3801--3802, require all three seeds, publicly
freeze U before loading official test data, then use 3810--3819. Require
paired 95% lower gain >=2 points over healthy/F and >=-0.5 over N,H;
retain M and every failed earlier scope. The old-consumer monitor, finite
corpus certificate and 1.20/1.10 batch-1/128 isolated CPU latency gates
remain unchanged. Intervals concern continuation variability on one
fixed reference, not independently pretrained models or data splits.

The shared-prefix implementation is established, not an algorithm novelty
claim. A positive result tests a consumer/utility/cost construction beyond
small models. It is supervised transfer, not auxiliary-task benefit.

CPU only, maximum two 12-thread workers, 8 GiB/job and 24 GiB aggregate,
3600 seconds/job and 7200 seconds/campaign. Float32 token caches require
raising this round's total disk ceiling from 6 to 12 GiB, after checking
host free space. No duplicate frozen weights per continuation seed. All
source, writing, aggregation, figures, compilation and packaging stay local;
remote roots remain opaque, and records/states are pulled with SHA-256.
