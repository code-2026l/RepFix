# Independent Reference and Real Vision Auxiliary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test real reconstruction-auxiliary benefit on the fixed stronger
backbone/suffix families, then replicate useful learning across independent
fitted references, label masks and train/validation partitions.

**Architecture:** Reuse the ImageNet-pretrained ResNet-18 final residual block
and ViT-B/32 final two attention blocks. A 256-unit ReLU decoder reconstructs
8x8 RGB input from terminal features. Detached and joint auxiliary branches
share the initial decoder, primary reader, optimizer initialization and orders.
The healthy suffix remains available for the old coarse consumer.

**Tech Stack:** Existing CPU PyTorch suffix/cache engine; immutable raw CIFAR-100
training file; local analysis, plotting and LaTeX. No GPU or test-set selection.

## Fixed data and reference protocol

Collect clean and one fixed augmented view of every official training image.
Use the existing augmentation rule (reflect-pad 4, random crop, horizontal flip),
with RNG 6200 in original image-index order. Targets are bilinear, antialiased
8x8 RGB values of the corresponding view, in [0,1]. Their sample variance sets
one scalar reconstruction-loss normalization using fitting inputs only.
The prefix is fixed ImageNet computation and may be cached exactly in FP32.
The large cache is an execution optimization, not additional supervision.

For fitted reference seeds 6200--6204, independently permute 50000 training
indices. Take the first 10000 as fitting, next 5000 as validation, next 2048
as isolated anchors. The remaining images are unused for fitting. A Bernoulli
0.2 mask with RNG reference_seed+10000 selects fine/coarse fitting labels.
The same existing standardized LBFGS/L2 recipe fits healthy fine/coarse heads
on clean labeled images. Thresholds retain the original 80/50 fitting-coverage
definition. Current normalization uses both fitting views and the 0.1-median-SD
floor, as in the selected complete-suffix study.

These fitted references and splits differ from the original fixed reference;
the ImageNet pretrained backbone weights remain shared. Report that distinction.
No official test inputs or labels are loaded during preparation/development.

## Development pilot and stopping rule

Use reference 6200 and continuation seed 6300 on both backbones. Ten AdamW
epochs, batch 128, suffix rate 3e-5, primary-head rate 1e-4, decay 1e-4 and
explicit primary-head L2 1e-4. Freeze LayerNorm parameters. Primary CE uses
only masked fitting labels; auxiliary reconstruction uses every fitting image.
Each batch chooses one of its two fixed views with a shared seeded order.

Initialize the decoder Linear(m,256)-ReLU-Linear(256,192) with RNG 6300. Warm
it for five epochs on detached healthy features, AdamW rate 1e-3, decay 1e-4.
Clone the resulting decoder and its optimizer state into all D/J arms. The
primary/suffix optimizer starts identically in each arm. During continuation,
decoder rate stays 1e-3. Reconstruction loss is mean pixel MSE divided by the
fixed fitting-target variance. D trains this decoder on detached features.
J uses auxiliary coefficients {0.01,0.1,1}, fixed before outcomes. D also updates
the decoder with coefficient 1; decoder normalization/weight effects are reported.
Retain a dose-matched detached decoder per joint coefficient before interpreting
an auxiliary-MSE ratio. Do not confuse a different decoder step size with an
encoder benefit; primary CE-only D is shared across coefficients.

Strong frozen controls: F affine readout, N width-128 residual GELU reader and
H affine readout on healthy features concatenated with the existing hidden
features. The same two-view labeled pool fits final readers. Report online
and final-refit scores separately. A primary-gain claim uses the final reader
and cannot substitute an online-head collapse as its comparator.

Extend an auxiliary coefficient to the two additional continuation seeds
6301/6302 only if the first pilot has J-D >=0.2 percentage points, reconstruction
MSE ratio <=0.95 against its dose-matched D decoder, and unchanged old decisions.
For each backbone select the smallest coefficient with all three positive
J-D gains and mean >=0.2 points, mean MSE ratio <=0.95 and unchanged consumers.
No eligible coefficient means no auxiliary confirmation for that backbone.
All grid outcomes, including failures, remain reported.

## Independent confirmation

Only after development choices are frozen, use fitted references 6201--6204,
each with three continuation seeds 6310--6312. Confirm F/N/H, CE-only D and
the selected J family. If no auxiliary is eligible but CE-only D exceeds H
by >=0.2 points in the first pilot, independently confirm F/N/H/D to address
the original single-reference gap, without an auxiliary-benefit claim.

The primary replication units are the four independent fitted references.
Report each reference mean, every continuation seed, and paired 95% t intervals
over the four reference means (df=3). A pooled 12-seed interval is descriptive
only and cannot replace the reference-level decision. Auxiliary success needs
a positive reference-level lower limit for J-D and MSE ratio upper limit <=0.95;
the old class/selective outputs must meet the original zero-change FP32 check.
For CE-only transfer, require a positive reference-level lower limit over H
and report the strongest F/N/H comparator as well. The dataset test split stays
closed: this independently split validation confirmation concerns this training
population, with no benchmark-test accuracy claim.

## Execution and review

Author locally and commit before execution. Prepare at most two prefix caches
concurrently; train at most four 12-thread CPU jobs. Empty CUDA visibility,
no fixed RSS cap, 256 GiB host-available reserve, at least 100 GiB disk free.
Time limits are 2h per job and 6h per campaign, with an explicit technical-stop
audit. Each completed job's metrics and logs are pulled promptly; useful model
states are recovered with SHA-256. Canonical prefix caches are reproducible
inputs and need not be duplicated in compact source or supplement archives.

New evidence enters the paper only after its stated gate and independent
replay. Preserve the polished release and all conditional-routing failures.
