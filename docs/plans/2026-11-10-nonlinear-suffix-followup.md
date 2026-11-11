# Nonlinear Suffix Follow-up: Fixed Transfer Recipe

The preceding single-map pilots and primary-refit audit failed the two-point
gate on all three backbones. Primary refits recover much of the online-head
decline but do not reveal the required feature gain. Preserve those failures.
This follow-up opens a complete nonlinear suffix and uses a matched transfer
recipe; it does not retune or rescue the earlier registered gates.

Use the same fixed reference, split, 20% labeled inputs and coarse consumer.
ResNet-18/50 update the entire final identity residual block; ViT-B/32 updates
both affine maps of the final CLS MLP. Attention and earlier layers stay frozen.
All CNN suffix BN transforms are fixed and folded before defining the two
suffix implementations; independently measure their float32 deviation from
the original reference. ViT keeps its LN2 and final LayerNorm fixed. Both old
and current outputs share the long frozen prefix. CLS pruning applies equally
to the current-only and protected comparators. Known prefix sharing is a
baseline and may be the winning construction; it is not a new algorithm claim.

Candidates U (all suffix affine weights/biases) and R64 (rank-64 updates to
every suffix affine map, zero left factors and gradient-SVD right factors).
Controls F (head only), N (width-128 residual GELU head), H (linear probe on
healthy features concatenated with pooled pre-final-map features), and L
(only the final suffix map updates). L isolates update scope under the same
training recipe. It is not a new rank search.

Fixed recipe: ten epochs, AdamW weight decay 1e-4, suffix/factor lr 3e-5,
current-head lr 1e-4; CE plus 1e-4 times squared primary-head weights and bias,
matching the healthy head's explicit L2 coefficient rather than equating it
with AdamW decay. For N include its trainable linear layers in that penalty.
Current normalization uses the two-view healthy fitting mean and per-coordinate
SD floored at 0.1 times median positive SD. Rebase the initial affine head so
its healthy function is unchanged in real arithmetic. Old coarse weights and
normalization are untouched. This conditioning rule is fixed before outcomes.

Views: clean input and one fixed CIFAR training augmentation per labeled image
(reflection padding 4, uniform 32x32 crop, horizontal flip with probability 0.5,
RNG 3901 in fitting order), then the declared 64/224 preprocessing. Choose one
of these two views per fitting input occurrence using RNG seed+71000; future
fitting permutations use seed+20000. All learned arms share both generators.
No new annotations. Fixed prefixes may be cached once; inference uses clean
inputs. H fits both views with equal weight and the original standardized
LBFGS/max-200/L2-1e-4 recipe.

After training, primary refits use the same two views and labels for F,L,U,R64.
N refits the same residual-head function class on concatenated healthy and
learned GELU features. This fixed refit removes a head-conditioning confound;
retain online-head scores too. No encoder changes during refitting. All
published utility uses the actual refitted serving arithmetic.

Pilot seed 3800 for every backbone. Gate unchanged: candidate gains >=2 points
over healthy and F, is >= U-0.5, N-0.5, H-0.5 points, and every old-contract
disagreement <=1%. If any candidate passes for a backbone, run its complete
fixed follow-up grid at 3801--3802. Require all three seeds to pass, select
lowest serving extra MACs then lower update rank (R64 before U on a tie), and
freeze publicly before fresh continuation seeds 3810--3819. Confidence intervals
then concern continuation seeds on this one fixed reference and data split.

Confirmation: same paired 95% lower gain >=2 points against healthy/F and
>=-0.5 points against U,N,H; retain the L scope control. Test is opened only
after that backbone's development gate. Old class/80/50 decisions use a fresh
1024-input monitor, RNG 13700001+seed, alpha 0.05/2000, five-point bound, <=1%
empirical changes. Native timing remains 1.20 batch-1 / 1.10 batch-128, with
the same fair frozen-prefix, two-suffix and independent-network controls.
Any failure remains a partial frontier. No post-test recipe or epoch choice.

At most two 12-thread workers, 8 GiB/job and 24 GiB aggregate; 3600-second
wall/43200 CPU seconds per job, campaign 7200 seconds. All source, writing,
analysis and manuscript compilation stay local. Opaque remote roots, CPU only.
The additional exact augmented caches raise the round's disk ceiling to 6 GiB
after checking >500 GB free; all other limits stay fixed. Pull records promptly
and saved states/checkpoint fingerprints with SHA-256. Preserve prior releases.
