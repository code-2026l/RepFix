# Shared-Reference Replay: Prospective Construction and Validation

**Goal:** Retain an old consumer by sharing its healthy computation, while a
mergeable structured parameter update learns nonlinear features without a
distillation penalty or a second full encoder evaluation.

**Architecture:** In a two-ReLU encoder, update k rows of the first affine
map and a rank-r factorization of the second map. Evaluate the healthy dense
path once. The first activation difference is supported on k coordinates;
the second preactivation difference is an old k-column product plus the
low-rank update. This produces the same current function as merging the
weight updates, in real arithmetic. Its training objective contains only
the original primary/auxiliary losses. The healthy output supplies old
readers independently. Exact replay and the affine-compressed old-head
implementation have separate floating-point audits.

## Prior work and intended increment

Low-rank adaptation, sparse fine-tuning, side networks, model caching and
incremental activation computation are established ideas. DeltaCNN (CVPR 2022)
reuses computation for sparse input-frame changes; activated LoRA reuses causal
prefix KV state. Neither general reuse nor LoRA is claimed as new. The tested
construction concerns two parameter versions on the same input, the old
consumer's reachable computation, and an explicit exact-arithmetic/cost
relation for structured nonlinear updates. Literature comparison and complete
empirical baselines are required before any novelty claim.

Let a1=ReLU(W1*x+b1), t2=W2*a1+b2. First-map updates have row support S:
delta_a1 has support S. Let the second weight update be U*V and bias update db2.
Then the current second preactivation is

  t2 + W2[:,S]*delta_a1[S] + U*(V*a1_current) + db2.

Evaluate both nonlinearities exactly; no linearization or activation truncation
is used. The last encoder affine map and task heads remain trainable and can
be folded for serving. A fixed affine legacy head can likewise fold its healthy
last map. Additional MACs depend on k*d + k*m + 2*r*m + C*m, not on a second
full first-layer product. Real-arithmetic identities do not imply bitwise
floating-point equality after folding; score errors and actual predictions
are measured separately.

## Local checks before remote training

Implement `shared_reference_replay.py` and compare with independently merged
dense networks in float64 and float32, including nonzero updates that change
ReLU masks. Check current features/logits, parameter gradients, healthy output
invariance, and exact versus affine-compressed legacy arithmetic. Check the
analogous convolution/identity-residual formula before a visual experiment.
No new loss coefficient is selected and no outcome is claimed from algebra alone.

## Small-model development

Use saved healthy checkpoints and original fitting/validation splits of seeds
800--802 for HAR reconstruction and SNR. Retain all preceding cell-loss studies,
their failed gates, and the separate classification-only ablation. These are
independent new structural development comparisons, not rescues of their gates.

Fixed configurations (k,r): (0,4), (4,4), (8,8), (16,8), hidden width 64.
Select nested first-layer supports by row norm of the original joint-loss
gradient on 1024 fitting inputs (permutation seed+51000). Initialize V from
the leading right singular vectors of that same healthy second-weight gradient,
and U, sparse first-weight/bias changes, and second-bias changes to zero.
Use the same support/basis in paired detached and joint arms. This is a
training-only initialization choice, not a novelty claim.

Keep the same original 30 HAR / 25 SNR epochs, batch 256, dose 100 / 1, learning
rate 0.001, future minibatch generator seed+20000, and primary label fractions
0.1 / 0.2. Adam moments for the final affine/task heads are inherited; frozen
base moments are retained but unused; new sparse/low-rank parameter states
start at zero. Use the original unclipped Adam recipe in both paired arms.
Numerical nonfinite losses/gradients stop the affected job. The sparse and
low-rank restrictions are identical for D and J. Healthy fitting activations
may be cached because their parameters do not update.

All current utility is evaluated on the actual serving arithmetic. Also report
the difference from the saved ordinary full-update J control; do not compare
only with a weaker structured baseline. Healthy model/source/optimizer/split,
support/basis and future-order hashes are recorded. No test scoring during
development. All three class/selective consumers are scored relative to the
healthy reference. Full replay is checked separately from compressed output.

Development utility gate: HAR mean joint/detached reconstruction ratio <=0.95
and mean primary difference >=-0.005; SNR paired primary-gain 95% lower >0 and
mean current accuracy difference from ordinary full-update J >=-0.005. Require
every empirical old-consumer disagreement <=0.01, and changed nonlinear
activations/parameters whenever that configuration permits them. Rank feasible
configurations by analytic extra MAC count, then smaller k/r. Retain all outcomes.

Before confirmation, measure serving on saved development endpoints with one
isolated worker. Compare the protected two-output implementation with the same
merged current-only function under the same dtype/backend and affine folding.
Primary limits remain <=1.20 at batch 1 and <=1.10 at batch 1024. Show ordinary
eager and optimized implementations separately when used; apply any runtime
optimization to both sides. No hardware-independent latency guarantee follows
from MAC counts. Cost failures remain failures. A route may retain utility and
compatibility while missing the complete target.

## Confirmation

Freeze development choices and their hashes before new seeds 2500--2509 for
passing domains. These are new training seeds on the existing fixed official
split, not new populations. Train paired structured D/J plus ordinary full D/J
controls on each new warm checkpoint. There is no consumer penalty and no
test-based selection. Fix all support/basis rules before fitting.

Use a fresh 1024-input uniform monitor, RNG 10700001+seed (+100000 HAR), tail
0.05/600. Certify all three consumers under the unchanged five-point budget.
Require HAR reconstruction-ratio 95% upper <=0.95 and primary-difference lower
>=-0.005; SNR primary-gain lower >0 and difference-to-ordinary-J lower >=-0.005.
Require the stated native serving limits for the complete target. Independently
reload raw states and recompute dense-merge equivalence, actual consumer
outputs/bounds and measured costs. Compressed and full-precision contracts
remain distinct. If any criterion fails, report the partial frontier explicitly.

## Fixed stronger-backbone screen after a small-model utility gate passes

Use ImageNet-V1 ResNet-18 / CIFAR-100, image size 64, seed 2600, the established
5000 validation / 2048 isolated anchors / 42952 fitting split, and a fixed 20%
fine-labeled fitting subset. Fit healthy fine-100 and coarse-20 heads on that
same subset. Coarse labels are not used in continuation; their derivability
from fine labels and the stronger-access baseline are explicit. No official
test input is decoded during screening.

Place the structured update in the last basic block, whose shortcut is the
identity. Freeze the earlier prefix and BN statistics/affines. Fold the last
block's fixed BN into the healthy convolutions before fitting healthy heads;
this defines the initial numerical consumer. Conv1 changes k output channels;
conv2 changes a rank-r factorization (3x3 reduction, 1x1 expansion). The same
delta identity uses old conv2 input-channel slices, and retains the skip/ReLU.
The healthy coarse output is read directly; current and legacy functions
share one healthy backbone evaluation. Cache prefix inputs/healthy intermediate
activations for numerical training, not paper writing or remote analysis.

Fixed configurations: (0,8), (8,8), (16,16). Include a frozen-prefix/head-only F
and a fully trainable-last-block U control, with the same frozen earlier prefix.
All use three epochs and identical original fitting-index permutations
(seed+20000). With frozen BN and no auxiliary loss, only the labeled inputs
in each original 128-input batch need a training forward. This preserves its
masked primary objective in real arithmetic. Use AdamW, weight decay 0.0001,
lr 0.001 for sparse/low-rank updates and normalized current head; U's dense
block lr is 0.0001 and head lr 0.001; F trains its head at 0.001. Current-head
normalization is fixed to healthy labeled-fitting features. The support/basis
uses a fixed labeled fitting gradient. No early stopping or epoch retuning.

Pilot utility requires current accuracy >=max(healthy,F)+0.02 and >=U-0.005,
and each old-consumer disagreement <=0.01. A frozen baseline degraded by
continuation cannot manufacture the required net gain over the healthy head.
Rank feasible candidates by analytic extra MACs. If any passes, repeat the
entire fixed screen at 2601--2602 with no tuning; only then freeze one choice
for five confirmation seeds 2610--2614 and official-test scoring. Confirmation
uses the same utility intervals, all three fixed-consumer bounds and isolated
cost checks (batch 1 <=1.20, batch 128 <=1.10). Compare to the matched trainable
last-block control, and state that this is a primary-update/PEFT study, not
full-backbone adaptation, an auxiliary-benefit result, or a foundation model.

## Resources and delivery

At most six four-thread small-model workers or two eight-thread vision workers,
8 GiB/job, 24 GiB aggregate, 4 GiB new remote disk, 1800/2400-second job wall
and 7200/19200 CPU-second limits. No GPU allocation. Shared guard covers any
overlapping owned queues. All roots are opaque; authoring, analysis, figures,
LaTeX and release assembly remain local. Pull useful results promptly and all
saved states with SHA-256. Runtime tracing/JIT, if used, is part of model
numerical evaluation, never manuscript compilation. Record its backend and
compile/warmup costs separately. Preserve every earlier release and verdict;
update the article and final source/supplement/submission folder from evidence.
