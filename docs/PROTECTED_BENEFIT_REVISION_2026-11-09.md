# Useful real auxiliary learning with a retained shared interface

This revision follows the user's request for both stronger evidence and layered,
more expressive figures. The previous source/PDF artifacts are preserved. All
source authoring, statistical analysis, figure rendering, writing, compilation
and packaging stayed local. Remote execution was bounded numerical CPU work
only; no GPU, GitHub push or OpenReview action was used.

## What the new result establishes

In the predeclared low-label RadioML SNR protocol, a reference-anchored
readout-nullspace control retains an auxiliary accuracy gain while preserving
the old consumer's complete logits. Ten new confirmation seeds pass all five
predeclared criteria. This is evidence for coexistence in one shared
64-dimensional representation, with an explicit retained-reference cost. It is
not a new projection algorithm, a cheap warning predictor, a single-encoder
remedy or a broad-architecture performance guarantee.

The ordinary joint model gains **0.97 percentage points** over ordinary
detachment (paired descriptive 95% interval **[0.63, 1.31]**). Within the guarded
architecture, joint learning gains **0.89 points** over guarded detachment
(**[0.60, 1.18]**). Guarded joint learning exceeds ordinary detachment by
**1.25 points** (**[0.92, 1.58]**); the guarded architecture itself contributes
part of this difference, so 1.25 is not labeled the auxiliary's isolated gain.
Guarded joint versus ordinary joint is **+0.28 points**, interval **[0.08, 0.47]**,
passing the predeclared half-point noninferiority margin.

| Arm | Current-head accuracy | Old-consumer accuracy | Old NLL | Old ECE | Certified |
|---|---:|---:|---:|---:|---:|
| Ordinary detach (D) | 0.275 | 0.259 | 2.176 | 0.153 | 0/10 |
| Ordinary joint (J) | 0.285 | 0.258 | 2.135 | 0.155 | 0/10 |
| Guarded detach (GD) | 0.279 | 0.259 | 2.031 | 0.048 | 10/10 |
| Guarded joint (GJ) | 0.288 | 0.259 | 2.031 | 0.048 | 10/10 |

Means are conditional on the fixed prepared split and recipe. The full tables
include seed SD and the planned paired intervals. Ordinary frozen accuracy is
almost unchanged despite about 59% changed predictions for J and worse ECE.
Thus this is not presented as rescuing a large accuracy collapse. The guarded
consumer preserves all scored predictions and NLL/ECE, with maximum absolute
logit error **4.62e-14**. Its SNR MSE is **0.766** times matched guarded detachment.

## The construction, its limits, and a necessary baseline

For the healthy affine consumer, define the normalized head
`A = W0 diag(1/s0)` and project onto `null(A)`. The emitted features are
`H0 + (Htheta - H0) P_perp`. Because `A P_perp = 0`, the healthy logits are
unchanged in real arithmetic. This is also the Euclidean-nearest feature vector
to the updated features subject to that logit constraint. All tested heads have
rank 11, leaving 53 free dimensions. Full-column-rank heads leave no feature
freedom; general nonlinear consumers do not inherit this global guarantee.

Both heads consume the same guarded 64-vector. The frozen reference encoder
must be retained at inference. Mean per-seed median single-thread CPU batch
latency is **2.37x** the ordinary encoder/head path (batch 1,024; nine timed
repetitions after three warm-ups, Xeon Platinum 8570). Fitting reference
features can be cached during training. The ordinary and guarded arms use the
same float64 classification evaluation and fp32 encoder/auxiliary parameters;
the projector is float64.

A simpler dual-routing alternative retains an old model for legacy consumers
and uses J for the current task. By construction it has the reference legacy
metrics and J's primary metrics, with two separate feature vectors. The present
control does not establish an efficiency advantage over that alternative; its
contract is one fixed-width shared interface. This distinction is explicit in
the article, together with the reference cost. Relevant prior art is acknowledged:
[backward-compatible representation learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html)
and [orthogonal gradient descent](https://proceedings.mlr.press/v108/farajtabar20a.html).

## Selection was frozen before confirmation

The complete design was locally committed as **1291d19**, before calculations.
The validation choice was committed as **532eb31**, before confirmation.

- Development seeds 200--202 tested six fixed domain/dose candidates: HAR input
  reconstruction at 10% fitting primary labels and doses 1, 10, 100; RadioML SNR
  at 20% and doses 0.1, 1, 10. No development test metrics were computed.
- The declared score is the minimum mean validation gain among J-D, GJ-GD and
  GJ-D. The fixed rule selects RadioML dose 1, without a later grid expansion.
  All six candidates, including weak and negative HAR/SNR cases, are retained.
- Confirmation uses seeds 300--309 and that one frozen recipe: width 64, Adam
  1e-3, batch 256, 25 healthy and 25 continuation epochs. Four branches share
  initial model/Adam states, label masks and future minibatches. No checkpoint
  or seed is selected from test results.
- The five declared criteria are three positive paired lower limits, the
  half-point noninferiority requirement, and preserved old-consumer logits/
  predictions. All five pass. The intervals quantify seed variation on this
  fixed dataset/split/recipe, not performance on new populations.
- Training-anchor drift is logged at fixed epochs and a fixed 0.05 threshold.
  Protection is fixed before training; this is not an adaptive warning policy
  or a measured warning-lead-time result.

The separate confirmatory consumer bank has 40 fixed consumers, total tail
probability 0.05, and 1,024 uniformly sampled monitor inputs per seed. It is
separate from the earlier 3,015-candidate bank. Guarded branches have zero
disagreement and a finite-corpus upper bound about **0.0065**, below the
five-point relative classification-error budget. This is not absolute accuracy
identification. New populations and unconstrained future models receive no
guarantee from that sampling bound; the structural identity has its own stated
affine/reference assumptions.

## Numerical verification and CPU limits

All **104** development/confirmation model--Adam states were pulled and checked
locally, including reference states, projector identities, source hashes,
initial-state/order pairing, primary-label isolation and all consumer bounds.
An independently launched saved-state replay reconstructs **all 40 consumers**;
accuracy, NLL, ECE and bounds match exactly (maximum metric difference zero).

The 26 principal development/confirmation/replay jobs used **1,357.9 aggregate
job-wall seconds**. Peak aggregate RSS was **5,300,883,456 bytes** (4.94 GiB),
with at most four single-thread workers. Two small real-data smoke jobs were
kept separate. Final inspection across all 20 opaque campaign directories,
including previous rounds, found no active owned run process. All **977**
remote output/state/log files match local SHA-256 copies; the complete remote
footprint is **549,106,429 bytes**. No paper documents or plotting programs
were uploaded. Connection details and mappings remain private under `tmp/`.

## Article and visual changes

The two main figures now use layered perspective feature planes, restrained
transparency, thin connectors and serif mathematical typography, based on the
user-supplied visual reference. Layers are explicitly schematic; no numerical
outcome or network depth is encoded in their apparent thickness. Figure 1 keeps
the original seed-level evidence. Figure 2 shows the protected shared interface
and both encoder branches, making the deployment cost visible. The detailed
paired trajectory/repair hierarchy remains in the repair appendix.

The new result table is in the main text. The appendix gives the exact
construction and rank/minimum-distance statement, frozen selection rule,
all planned contrasts, the complete seed plot, sampling scope and dual-routing
baseline. All thirteen included figures have vector PDF, editable SVG and PNG
exports at final manuscript width. Quantitative charts remain two-dimensional
with original seeds and uncertainty. The older numerical evidence is unchanged.

Local build checks preserve **nine actual main-text pages and 58 total pages**, references starting
on page 10, and no unresolved references or overfull boxes. The full document
is rendered for visual inspection. The source package excludes raw datasets,
large trained states and private machine information. Rebuilding figures uses
the shipped numerical records; full trained-state verification additionally
requires the local checkpoints and prepared data described by the drivers.

Main sources: `benefit_compatibility_cpu.py`, `readout_nullspace.py`,
`verify_benefit_consumers_cpu.py`, `summarize_benefit_compatibility.py`,
`plot_benefit_confirmation.py`, `spatial_diagrams.py`, and
`docs/plans/2026-11-09-prospective-benefit-compatibility.md`. The recorded choice
digest accepts the LF/CRLF serialization of the same frozen Windows-authored
JSON; the selected content, source, plan and branch configuration are checked.
