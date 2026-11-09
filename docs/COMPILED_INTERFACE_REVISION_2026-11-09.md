# Compiled compatibility: a measured plasticity boundary

This round follows the user's explicit request to keep advancing the paper.
It tests the main practical gap left by the preceding positive result:
whether exact old-consumer compatibility can be retained in one encoder while
preserving a real auxiliary's primary-task benefit. The full target fails on
both fixed tasks. The measured failure is retained and explained, not tuned away.

## What was fixed before calculation

The design was committed as **fa4469b** before new numerical execution.
The tested implementation and independent replay driver were committed as
**a5e93f8**, after small smoke checks and before confirmation. No new validation
grid or test-selected endpoint is introduced.

RadioML uses real SNR regression, dose 1, 20% of fitting primary labels and
25 healthy + 25 continuation epochs. HAR uses real 561-D input reconstruction,
dose 10, 10% labels and 60 + 30 epochs. The doses come from the prior archived
validation grid: RadioML's selected choice and HAR's positive max-min candidate.
Each task uses ten new seeds 400--409, the unchanged prepared test split and
a stratified 20% training reservation before model/head fitting. Width is 64,
Adam 1e-3, batch 256. All seeds and fixed endpoints are retained.

Six paired branches clone model and complete Adam states and future orders:

- D/J: full-encoder detach/joint continuation.
- LD/LJ: frozen nonlinear trunk, unconstrained last-affine detach/joint.
- CD/CJ: frozen trunk, constrained last-affine detach/joint.

Every arm uses a native float64 final encoder affine map and classification
computation, with float32 trunk and auxiliary parameters. The healthy consumer
is fitted to that same interface. This precision difference and the new seeds
prevent pooling this follow-up with the earlier encoder-fp32 confirmation.
Validation/test inputs and the unlabeled test monitor never enter fitting.

## Algebra and deployment result

For `h0 = B0 phi0(x) + c0`, keep `phi0` fixed and let `P` project onto the
nullspace of the normalized consumer `A = W0 diag(1/s0)`. Parameterize
`Bbar = B0 + P(B-B0)` and `cbar = c0 + P(c-c0)`. Both `A Bbar = A B0` and
`A cbar = A c0`. Fold the effective weight and bias once after training.

The deployed encoder has one trunk, one final affine map, the original shapes
and three affine evaluations. It contains no reference-map copy or projector.
This construction is an elementary control, not a new projection/adaptation
algorithm. It cannot learn new nonlinear trunk features. Changing the trunk
invalidates its global functional guarantee.

All 40 CD/CJ endpoint consumers preserve old predictions, NLL and ECE. The
maximum logit error is **3.5527e-14**. Train-time and folded test features are
bitwise identical. The separate 120-consumer family, delta 0.05 and a uniform
1,024-input monitor per seed, certifies all 40 compiled consumers under the
five-point relative classification-error budget. Other arms may abstain;
abstention is not observed harm. The bank is separate from the earlier 40 and
3,015 candidate families, and provides no new-population/future-update bound.

At batch 1,024, the compiled/ordinary ratio of per-seed median CPU latency is
**1.006 [1.001, 1.011]** for SNR and **1.002 [1.001, 1.004]** for HAR.
Batch-1 means are both approximately 1.000. The corresponding new two-encoder
paths average 2.223x and 2.161x at batch 1,024. Measurements use one Xeon Platinum
8570 thread, five warm-ups and 21 alternating timed repeats of the encoder plus
current-head path. The comparator has
the same precision and native parameter shapes. These are host/batch-specific
cost measurements, not performance evidence or a universal zero-latency claim.

## The task-benefit result is negative

All differences below are percentage points, mean with paired descriptive 95%
t intervals over ten seeds conditional on each fixed dataset/split/recipe.

| Predeclared contrast | RadioML SNR | HAR reconstruction |
|---|---:|---:|
| J-D: unrestricted auxiliary benefit | +1.110 [0.904, 1.315] | +0.037 [-0.300, 0.375] |
| CJ-CD: compiled auxiliary benefit | +0.007 [-0.020, 0.035] | +0.007 [-0.079, 0.093] |
| CJ-D: total gain | -2.423 [-3.201, -1.645] | -0.034 [-0.431, 0.363] |
| CJ-J: noninferiority contrast | -3.532 [-4.233, -2.832] | -0.071 [-0.473, 0.330] |

The first three required positive lower limits; the fourth required a lower
limit above -0.5 point. Together with exact consumer preservation, these form
the five declared conjunctive criteria. Neither task passes. HAR's
noninferiority alone does not establish useful auxiliary benefit.

SNR's explanatory controls isolate the dominant observed cost:
LJ-J is **-3.529 [-4.213, -2.846]**, while CJ-LJ is only
**-0.003 [-0.039, 0.033]**. Freezing, rather than an additional large constraint
penalty, explains most of this recipe's performance loss. LJ-LD is a small
positive **0.028 [0.0006, 0.0554]**, but it does not preserve the full-network
benefit or make the compiled control pass. All nine planned contrasts per task
are present in the generated table and JSON, regardless of signs.

The earlier positive two-encoder result remains valid within its own declared
scope. This follow-up rules out a simple cheap shortcut in the tested SNR
recipe. Compatibility with continuing nonlinear learning at one-encoder cost
remains an open design target. It is not established by a nonempty feature
nullspace, a good scalar scatter value or an exact logit identity alone.

## Integrity and resource use

Local verification checks all **180** saved state files: 20 healthy model/Adam
states, 120 endpoint model/Adam states, and 40 compiled models. It checks source,
plan, split/mask, initial-model/Adam and future-order hashes, normalized
projectors, unchanged frozen parameters and Adam states, folding and consumer
bounds. An independently launched process reconstructs **all 120 consumers**;
accuracy, NLL, ECE and bounds match exactly, maximum metric difference zero.

The 40 principal confirmation/replay jobs use **1,374.2 aggregate job-wall
seconds**, at most four single-thread workers, peak aggregate RSS
**5,062,254,592 bytes (4.71 GiB)**. Two small smoke jobs are separate. All
calculation is CPU-only with empty CUDA visibility. Across all 23 opaque
campaign directories, final inspection verifies **1,287 local/remote file
SHA-256 matches**, no owned active jobs, a total footprint **800,515,833 bytes**,
and zero article files uploaded. No unrelated job is touched. Machine/connection
details remain private under ignored `tmp/`.

All code editing, statistics, illustration, article writing, LaTeX compilation
and packaging remain local. No GitHub push or OpenReview action occurs.

## Article and visual changes

The abstract, introduction, main intervention section and conclusion now state
the measured frozen-trunk limitation alongside the earlier priced positive
control. A new appendix gives the folding corollary, exact precision and
controls, all failed criteria, full tables and independent replay. Two new
vector figures show the parameter fold/single inference path and every
predeclared contrast with all seeds. The prior evidence is preserved.

The article remains nine actual main-text pages, with references starting on
page 10. The latest complete PDF is 62 pages. There are 15 included scientific
figures with PDF/SVG/PNG exports, and 39 meaningful structural/control tests.
The final source package excludes raw datasets, trained states and private
operations; all figure inputs and numerical records are included.

Relevant primary literature is acknowledged without claiming algorithmic
novelty: [OGD](https://proceedings.mlr.press/v108/farajtabar20a.html),
[backward-compatible learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html),
[LoRA](https://arxiv.org/abs/2106.09685), and
[NESS](https://proceedings.mlr.press/v328/pham26a.html).
