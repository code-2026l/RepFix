# Nonlinear compatible interface: verified construction, unclosed benefit gate

This round follows the user's renewed request to inspect the gaps and continue
research. It reviews the outstanding utility/cost, external-scope, prospective-
policy and novelty problems, then tests a narrow nonlinear channel that avoids
the previous frozen-trunk function restriction. The development gate is not
passed. No independent scientific test confirmation is launched.

## Plan and exact decision boundary

The design was committed as **b5b99eb**, with the manuscript-gap audit and real-
auxiliary corroboration criterion committed as **1241a8f** before development.
The tested numerical implementation was committed as **e7bbd9f** before the
six principal development runs. The corrected exact-count decision, archived
first serialization and independent replay implementation were committed as
**e3344c1** before any scientific confirmation.

Development seeds 500--502 use width-64 base MLPs and side widths 16/32.
RadioML uses 20% primary fitting labels, SNR dose 1 and 25+25 epochs. HAR uses a
new **1%** fitting-label reconstruction regime, doses 1/10 and 60+30 epochs.
HAR fitting/validation counts are 5,882/1,470 with 59 labeled fitting examples;
RadioML counts are 88,000/22,000 with 17,600 primary labels. The stratified 20%
reservation precedes all model/head fitting. All encoders are native float64;
head parameters are fp32. Adam is 1e-3, batch 256. These are not pooled with
prior fp32/mixed-precision studies or other label fractions.

Each candidate compares full D/J, unconstrained-side UD/UJ and constrained-side
TD/TJ, reusing healthy references and shared Adam states, side initializations,
masks and future orders. The score is the minimum validation mean among J-D,
TJ-TD, TJ-D and (TJ-J + 0.5 percentage point). Ties favor smaller width then
dose. A task qualifies only if its best score is strictly positive. The fixed
confirmation criteria would additionally assess independent test intervals,
real auxiliary MSE progress and the CPU-budget target. They are not assessed
after the development gate fails.

## Structural advance and its price

Emit `h = f0(x) + P g(x)` with the frozen base `f0`, trainable nonlinear `g`,
and `P` onto the kernel of `A = W0 diag(1/s0)`. Then `A P g = 0` for every input
in exact arithmetic while the side's nonlinear prefix can learn. Equal-depth
paths fuse through stacked first-layer weights, a block-diagonal middle map
and a concatenated final map `[B0, P Bside]`. Final biases are combined as well.

The result is one dense width-80/96 MLP with three affine evaluations and a
64-dimensional output. It retains the frozen base weights inside that wider
encoder. Dense zeros and increased capacity are explicitly counted. Encoder
parameter ratios are 1.301/1.623 on RadioML and 1.279/1.569 on HAR for side
widths 16/32. There is no same-size or measured-runtime-saving claim. Full-data
timing was contingent on confirmation and was not performed.

All **36** constrained validation endpoints preserve old predictions, NLL and
ECE with maximum logit error **2.4869e-14**. The side's nonlinear prefix changes
in every constrained branch. Frozen base parameters and Adam states stay
unchanged. Fused/unfused feature error is at most **1.0658e-14**. These verify a
compatible nonlinear construction; they do not establish useful adaptation.

## Complete development outcomes

All entries below are mean validation accuracy differences in percentage points
over the three development seeds. They are not independent confirmation.

| Task | Side width | Dose | J-D | TJ-TD | TJ-D | TJ-J | Gate score |
|---|---:|---:|---:|---:|---:|---:|---:|
| SNR | 16 | 1 | +0.34 | +0.53 | -1.73 | -2.07 | -1.73 |
| SNR | 32 | 1 | +0.34 | +0.32 | -0.69 | -1.02 | -0.69 |
| HAR | 16 | 1 | 0.00 | +0.70 | +0.34 | +0.34 | 0.00 |
| HAR | 32 | 1 | 0.00 | +0.39 | +0.25 | +0.25 | 0.00 |
| HAR | 16 | 10 | +1.50 | +0.32 | -0.05 | -1.54 | -1.04 |
| HAR | 32 | 10 | +1.50 | -0.29 | -0.43 | -1.93 | -1.43 |

The ranked SNR candidate's TJ-TD is **+0.315 [-0.167, 0.798]** points, while
TJ-D is **-0.688 [-1.930, 0.554]**. The ranked HAR candidate's TJ-TD is
**+0.703 [-1.760, 3.165]**, while ordinary J-D is exactly zero in aggregate
correct counts. Both fail the complete positive-mean selection rule. A positive
selected validation effect alone cannot be promoted to a held-out result.
The intervals are descriptive paired t intervals conditional on the three
development seeds and these fixed splits/recipes; selection further limits
their inferential interpretation.

Validation auxiliary MSE does improve in the side: the SNR width-32 ratio to
matched TD is 0.710; HAR width-16/dose-1 is 0.847. That progress does not close
the primary-benefit/noninferiority conjunction. All six candidates, all
unconstrained controls and all null/adverse results remain available.

The initial floating average represented the tied HAR J-D count as about
3.7e-17, which would wrongly open a strict-positive gate. This was corrected
using integer validation correct counts and exact rational means, before any
scientific confirmation. The initial JSON is archived as
`nonlinear_development_choice_float_bug.json`; the corrected immutable choice
records its hash and explanation. No dose, seed, margin or score criterion was
changed. Regression tests cover exact ties and different sample denominators.

## Integrity, resources and scope

Local verification checks all **165** state files: six healthy states, 87
endpoint model/Adam states and 72 deployed side models. It checks all sources,
plan/split/mask/order hashes, common optimizer state, paired side initialization,
frozen base values/Adam state, projected logits and dense zero blocks. An
independent tensor-only process replays **all 87 validation consumers**;
accuracy/NLL/ECE have maximum metric difference zero. Auxiliary losses are also
checked. The source package excludes large states and raw datasets, while
shipping full records, choice/correction, replay results and figure inputs.

The 12 principal development/replay jobs use **795.1 aggregate job-wall seconds**
at most four single-thread CPU workers, peak aggregate RSS **3,974,725,632 bytes
(3.70 GiB)**. Two one-epoch smoke jobs are separate. Final audit across all
**26** opaque campaign directories verifies **1,528** local/remote SHA-256 matches,
no active owned run process, total footprint **1,143,232,491 bytes**, and zero
article files uploaded. No GPU or unrelated job is used or affected. All
source authoring, statistics, figures, writing, compilation and packaging stay
local. No GitHub or OpenReview action occurs.

The planned 200-consumer confirmation family is not instantiated; no sampling
guarantee or full-data CPU latency is claimed for this development round.
Changing nonlinear weights is possible under the exact interface constraint,
but a matched frozen-side-prefix control was not tested, so prefix learning
alone is not isolated as a causal source of performance. Cross-task independent
confirmation, strong-backbone gains, nonlinear downstream consumers and online
warning/intervention policies remain open. The main contribution stays on
diagnostic scope and measured compatibility/utility/cost, not a new optimizer.

## Article and visual changes

The main text adds a short statement of the failed development screen. The new
appendix gives the compatible nonlinear construction, fusion and parameter
cost, complete gate, arithmetic correction, all candidates and the explicit
decision not to use test confirmation. Two editable vector figures show the
nonlinear/fused path and every development seed. Earlier evidence and artifacts
are preserved.

The manuscript has **nine actual main pages and 64 total pages**, references
starting on page 10, with 17 included PDF/SVG/PNG figures. The complete check
runs **47** structural/control tests and a clean source/figure/LaTeX rebuild.
The broader gap review is `docs/MANUSCRIPT_GAPS_2026-11-09.md`.

Established primary literature is acknowledged, including
[Side-Tuning](https://arxiv.org/abs/1912.13503),
[backward-compatible representation learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html),
[OGD](https://proceedings.mlr.press/v108/farajtabar20a.html) and
[LoRA](https://arxiv.org/abs/2106.09685). No claim is made to invent side
adaptation, projection or branch fusion.
