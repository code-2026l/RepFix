# Replay Frontier Extension Implementation Plan

**Goal:** Resolve the remaining utility/cost and stronger-backbone gaps with
controlled mechanisms, matched CPU implementations and fixed late-update screens.

**Architecture:** Preserve all incoming dense/sparse results. Separate row support
from matrix rank, then test a small versioned suffix after a shared frozen prefix.
Late tuning, low-rank adaptation and side heads are established controls, not new
algorithmic claims. The new studies test explicit consumer and resource contracts.

**Tech stack:** CPU PyTorch/torchvision, NumPy/SciPy, a standalone C++ CPU kernel,
local statistical analysis/Matplotlib/LaTeX, bounded opaque remote directories.

## 1. Controlled mechanism (development only)

Use original saved SNR development references, seeds 800--802. Fix first-layer
rank ceiling 16. Cross first-map family {16 supported rows, dense rank 16} with
second-map rank {8,12}, paired detached/joint arms. Keep the original 25 epochs,
Adam 0.001, batch 256, dose 1, inherited old moments and future orders. Both
calibrations use the same 1024 fitting inputs; second-map right factors must
match. First sparse rows use the original gradient-norm ordering, dense right
factors use its SVD. Record actual update ranks, nonlinear gates and all outcomes.
No test scoring. This is a controlled development attribution, not confirmation
that rank alone caused the prior 0.79-point gap. A row-supported rank-16 update
and a dense rank-16 update differ in orientation and degrees of freedom.

## 2. Native CPU cost audit

Implement a fused float32 CPU kernel for selected SNR dense (16,12) and HAR
sparse (4,4), plus an independently merged current-only kernel with identical
layout/compiler flags. Inputs/weights are contiguous, work buffers allocated
before timing, one API call per operation. SIMD and fused activation work apply
to both. Report empty-call overhead and C-internal repeat timing separately;
do not add work to the current baseline. Include the earlier TorchScript path
and independent two-function evaluation, and absolute latencies. Use fixed
compiler flags, four threads maximum; batch-one kernels use one thread.

Primary limits remain 1.20 / 1.10 at batch 1 / 1024. Compare protected serving
to the fastest registered current-only implementation on development, then
freeze the backend before any confirmation-serving audit. Report matched
backend ratios too. Native kernels may change floating-point associations;
audit actual outputs on all ten saved SNR seeds 2520--2529 and HAR 2500--2509,
fresh monitor RNG 12700001+seed (+100000 HAR), alpha 0.05/2000, 1024 inputs.
This is a new implementation audit on existing states, not new training.
Do not call measured failures a proof of unreachability for every two-output
program or architecture. Correct MAC accounting against the actually folded
current-only graph, separately from the historical encoder-only denominator.

## 3. Fixed late-update screens on three backbones

Architectures: ImageNet-V1 ResNet-18, ImageNet-V2 ResNet-50 and ImageNet-V1
ViT-B/32. CNN images use the established 64x64 normalization; ViT uses 224x224
bilinear resizing and ImageNet normalization. All numerical work is float32,
with fixed evaluation-mode normalization. These are frozen-prefix studies,
not full-backbone or auxiliary-benefit studies. Pretrained parameter count is
reported separately from updated parameters. Official weight fingerprints are
retained; only official cached/downloaded torchvision weights are used.

Fixed training-only split/mask/reference seed 3800: 5000 validation, 2048
isolated anchors, 42952 fitting, 20% labeled fitting (the existing seeded
Bernoulli mask). Fit healthy fine-100 and coarse-20 heads on the same labeled
subset using the established standardized LBFGS/l2 recipe. Coarse labels are
not used in continuation; their derivability from fine labels is explicit.
Cache frozen activations once per architecture. Reference and split stay fixed
across continuation seeds, so intervals describe continuation variability only.
Do not open the official test pickle before a backbone's development gate.

Fork point: ResNet-18's last block conv2; ResNet-50's last block conv3; ViT's
last MLP projection, reading only CLS since no attention follows it. Reuse
the preceding frozen computation. Fixed CNN BN is folded into the reference
last map before healthy heads are fitted. ViT keeps its final LayerNorm and
residual. All trailing nonlinearities are evaluated exactly. Current-only
serving uses the same CLS pruning and fusion. Legacy reads the healthy path.

Candidates: rank 16,64,128 updates to the last map, initialized with zero left
factor and gradient-SVD right factors on 1024 labeled fitting inputs. Learn
both factors and an effective bias. Strong controls: F continued healthy linear
head; U unrestricted last-map update; N residual width-128 GELU head on frozen
healthy features (zero final residual output, trainable base head); H a regularized
linear probe on concatenated healthy features and pooled pre-projection features.
H is fitted once with the same labels and normalization/l2 recipe. It is a
stronger frozen-feature control. For convolutions, use a kernel reduction and
1x1 expansion; for ViT a dense reduction/expansion. U is a legitimate known
shared-prefix deployment candidate; do not hide it if it wins.

All learned arms use three epochs, original fitting-index permutations from
seed+20000, masked 128-example batches, AdamW weight decay 1e-4, tail/factor
lr 1e-4 and current-head lr 1e-3; N head lr 1e-3. No epoch or lr search.
Use common fitting order across arms and no consumer penalty. Record finite
losses/gradients, parameters, gate/feature changes, and independently merged
agreement. Cache features for numerical training, not paper authoring.

Development: pilot seed 3800 on each architecture. A candidate must gain >=2
points over healthy and F, lie within 0.5 points of U, N and H, and change
each old class/selective consumer on <=1% of validation inputs. If a backbone
has any eligible candidate, run its full fixed grid on seeds 3801--3802.
Require the candidate to pass all three seeds, then select smallest analytic
extra MAC count, then smaller rank. U is included in this selection.
If a screen fails, retain every outcome and stop that backbone before test.

After choice is publicly frozen, ten fresh continuation seeds 3810--3819
confirm each passing backbone with the selected candidate plus F,U,N,H.
Require paired 95% lower gain >=2 points against healthy and F, and >=-0.5
points against each U,N,H comparator. Test monitor: 1024 uniform inputs,
RNG 13700001+seed, alpha 0.05/2000, five-point finite-corpus remainder budget,
also <=1% empirical changes for class and the fixed 80/50 threshold readers.
No test selection or monitor enlargement. Report all task results even if a
criterion fails. Numerical equality and any near-tied decisions are distinct.

Cost: isolated 12-thread float32 TorchScript serving, batches 1/128, current-only,
shared-prefix two-dense-suffix, low-rank replay and two independent whole networks.
Apply pruning/fusion/normalization equally. Warm all paths, alternate operation
order, retain 31 repetitions. Development chooses a serving implementation only
after utility selection; freeze it before confirmation. Require latency ratios
<=1.20/1.10 with paired uncertainty reported locally. Absolute time, parameters,
extra MACs and compilation/warmup are recorded. A pass for a late suffix does
not rescue the old full-MLP cost failure.

## 4. Resources and delivery

Numerical CPU only; empty CUDA visibility. At most three four-thread mechanism
workers, two 12-thread vision workers, or one isolated timing worker. Campaigns
are sequential so the combined cap is 24 threads and 24 GiB; per job 8 GiB,
wall 3600 seconds and CPU 43200 seconds (timing 2400 / 28800). Campaign wall
7200 seconds; new data/state/cache disk <=4 GiB. Check host load/free space
before launch; stop only owned processes on limits. Opaque random root names.
Pull useful records promptly, all states with SHA-256, and official weights
locally for independent replay. Do not duplicate frozen backbone weights per seed.
Runtime C++ compilation is numerical-kernel evaluation; manuscript compilation,
analysis, figures and all source authoring remain local. Verify exact identities
before training, replay endpoints, preserve old releases, and build the current
PDF/source/supplement/submission folder from an extracted source archive.
