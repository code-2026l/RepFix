# Two-block Scope Check

The final-attention pilot gained exactly two validation points over the
healthy head, but only 0.94 points over F, and failed. Keep this failure and
do not score its official test split. This final scope escalation opens the
last two transformer blocks, freezing the first ten, on the same fixed
ViT-B/32/CIFAR-100 reference and 20% labels. Earlier attention may change
the patch features supplied to the last block; unlike the single-block
screen it cannot be pruned to CLS until its final block.

Use the same views, conditioning, head penalty, ten epochs, learning rates,
masked fitting orders, final primary refits, F/N/H and M controls, seeds,
two-point/half-point utility gates and consumer contracts from the final
attention plan. U updates all affine maps of both blocks; every LayerNorm
parameter stays frozen. M only updates the last block's two MLP maps.
The first of the two blocks processes all tokens; the final block uses
the exact CLS-only query and MLP reduction, identically for all serving
comparators. Independent full-transformer output/gradient checks precede
the pilot. No learning-rate, epoch, or further cut search follows a failure.

Pilot 3800, then 3801--3802 only if it passes. Freeze the three-seed choice
publicly before official test loading; if selected, confirm 3810--3819.
Require paired 95% lower gains >=2 points over healthy/F and >=-0.5 over
N/H; retain M. Use the unchanged 1024 monitor, alpha 0.05/2000, five-point
remainder bound, <=1% observed disagreements and isolated 1.20/1.10
batch-1/128 timing gates. This is supervised transfer with a fixed split
and healthy reference; it is not auxiliary benefit or a new prefix-sharing
algorithm. Report the stronger frozen-feature alternative's utility and
cost alongside U so that a cheap head is not mistaken for encoder learning.

At most two 12-thread numerical workers globally, one per queue when queues
overlap; the 24 GiB guard remains active. Per job 8 GiB RSS, 3600 seconds
wall and 43200 CPU seconds; campaign 7200 seconds. The additional float32
tokens require this round's disk ceiling to become 16 GiB after checking
free host space. Pull the completed one-block states and caches, with SHA-256,
before any optional removal of specific remote cache files. Preserve the
source and evidence for all preceding failures. All authoring, aggregation,
figures, manuscript compilation and packaging stay local; remote CPU only.
