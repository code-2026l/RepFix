# Final Evidence Completion Plan

**Goal:** Close material baseline, deployment-contract and stronger-backbone
gaps, then deliver one coherent manuscript and verified submission artifact.

**Architecture:** Keep every earlier protocol and verdict. Add a fixed
retrospective matched HAR baseline/consumer audit and a separate gated vision
study. Author, analyze, draw and compile locally; remote CPU numerical work only.

## 1. Matched HAR controls and consumer contracts

Use all saved fresh confirmation seeds 1300--1309. No new original-study
choices are made. This supplemental comparison is specified after those
outcomes were known and is labeled retrospective.

On each saved healthy model and its original Adam moments, train a feature map
with lambda 1, the existing development grid's lowest reconstruction-MSE ratio
among candidates whose mean primary difference is within 0.005 of detachment,
mean reconstruction ratio is <=0.95 and every validation disagreement <=0.05.
This utility-aware choice is made on development data and does not replace the
original compatibility-only policy's failed gate. Use the original 30 epochs,
dose 100 and identical future minibatches. Save models and folded fp32 readers.

Fit the existing fixed standardized OLS to healthy training logits on detached,
joint, protected and new feature-aligned features, with no validation/test fit.
Report both float64 and fp32, using fp32 as the serving implementation. Include
the saved raw readers and the new feature map; do not tune a ridge coefficient.

Evaluate old-to-new correct/wrong flows for ordinary classification. Separately
audit two versioned selective consumers: return the class above a healthy
training confidence threshold, otherwise abstain. Threshold quantiles 0.2 and
0.5 target 80% and 50% training coverage, respectively, and remain fixed.
Report coverage, accepted-set accuracy, NLL/ECE and decision disagreement.
Classification harm and total decision disagreement remain different metrics.

The new contract audit uses a fresh uniform 1024-input monitor with RNG seed
8490001+seed, total tail 0.05/1000 (a conservative fixed bank covering all new
HAR/vision readers and contracts). It does not replace original certificates.
Independently replay fitted heads/metrics, and benchmark actual fp32 current
plus legacy outputs in one encoder pass after training queues finish.

## 2. Stronger-backbone, different-consumer vision screen

Use the existing CIFAR-100 cache and ImageNet-V1 ResNet-18, inputs 64x64,
float32, frozen BN running statistics. Fine labels define the current 100-class
task; official coarse labels define a distinct 20-class legacy consumer.
Both healthy heads fit the same fixed 20% labeled subset of the fitting pool.
Coarse labels are never used in continuation or unsupervised repair fitting;
the hierarchy makes them derivable from fine labels, so label inaccessibility
is not claimed. A supervised refit using the existing coarse fitting labels is
an explicitly stronger-access comparison.

Split official train into 5000 stratified validation inputs, 2048 isolated
anchors and 42952 fitting inputs, using the established split routine. The
official test split stays unopened during the screen. Cache healthy fitting,
anchor and validation features once. Use the established LBFGS healthy-head
fit, then fold its fixed normalization into fp32 affine weights. The primary
fine-task labeled mask is fixed with generator seed+501.

One-epoch screen seed 1800, batch 128, SGD momentum 0.9, weight decay 5e-4,
encoder LR 1e-3, current and extra-head/map LR 1e-2. All branches clone the
same healthy state and use identical minibatch permutations (seed+20000).

- F: frozen encoder, current head learns.
- U: unconstrained encoder and current head learn.
- P: jointly learned coarse legacy head, initialized to healthy mapping,
  distilling healthy coarse training logits, lambdas 0.1, 1, 10.
- B: bias-free learned 512x512 identity-initialized feature map, matching all
  healthy training features by per-coordinate MSE, lambdas 0.1, 1, 10.

Fit training-only OLS legacy readers on U and P endpoints, and a supervised
coarse refit on U using the same old labeled subset. Record fine/coarse task
metrics, decision/harm flows, nonlinear-layer/BN/order hashes and both old
confidence-threshold contracts. No augmentation or auxiliary objective is
added. This tests primary-update deployment generality, not a new auxiliary
benefit or a foundation-model claim.

Resource gate: U must improve fine validation accuracy over F by >=0.02 and
change >=0.10 of legacy decisions. A protected candidate must have empirical
legacy disagreement <=0.05 and fine accuracy >= U-0.005 and >=F+0.02. Among
feasible candidates select highest fine accuracy, tie lower lambda. If no
candidate qualifies, stop this vision route without retuning or opening test.

If the screen qualifies, repeat the full fixed grid at development seeds
1801--1802. Freeze one candidate per passing family using the same joint
utility/compatibility rule on three-seed means, requiring disagreement <=0.05
in every seed. Only then train five confirmation seeds 1810--1814 with F, U
and selected candidate families. Test acceptance requires fine gain over F
with a positive paired 95% lower endpoint, difference to U lower endpoint
>=-0.005, every 1024-monitor legacy bound <=0.05 and <=0.05 empirical changes.
Use fresh monitor RNG 8490001+seed, tail 0.05/1000, native fp32 serving ratio
<=1.10 at batch 128 and <=1.20 at batch one. Confidence-threshold contracts
are reported independently and cannot substitute for the primary criterion.

## Resources, verification and final deliverables

At most two workers globally, four threads per HAR job and eight per vision
job; 8 GiB RSS/job, 24 GiB aggregate; 1200 seconds wall and 9600 CPU seconds
per vision job, campaign wall 3600 seconds. One isolated serving benchmark.
Opaque remote root names, empty CUDA visibility, no GPU calls. Check free disk
before staging; keep this round's remote files below 4 GiB. Pull JSON/logs
promptly and checkpoints on completion, verifying SHA-256. Stop on nonfinite
loss, resource limit, or failed scientific gate. No search extension follows.

After the fixed work: review mathematical and statistical claims, eliminate
unsupported necessity/universality and unfair baseline comparisons, polish
natural prose and figures, preserve nine main pages, run required checks and
clean archive rebuilds, update the existing GitHub draft and the single latest
delivery folder. Preserve all prior releases and negative outcomes.
