# Scientific assessment of the interface frontier

The new evidence establishes useful nonlinear-suffix transfer on ResNet-18 and
ViT-B/32 beyond strong frozen-feature readouts. It complements the confirmed
HAR reconstruction and SNR auxiliary-regression results. The actual serving
frontier below states which numerical implementations meet the original budgets.
These are measured conditional results, with complete failed controls retained.

## Scientific contribution and attribution

The central distinction is between retained task information, a specified
consumer's decisions, and the computation needed to retain its old function.
Matched continuation and unlabeled input repair expose the scale component;
exact affine crossings and signed update audits identify what the old scatter
and conflict diagnostics do not determine. Class agreement alone need not retain
an original confidence-threshold decision.

The constructive evidence then separates three mechanisms: coordinate repair,
a learned legacy readout, and retained nonlinear reference computation. The
same model can need more learned nonlinear scope for current-task benefit while
reusing a large frozen prefix for its old output. The complete-suffix controls,
strong frozen-feature probe, post-hoc repairs, standard distillation control and
actual serving comparison test that explanation directly.

Low-rank updates, model stitching, side-tuning, distillation and CPU AMP are
established components. The paper cites their sources and uses them as controls;
the shared-product identity specifies the retained and updated computations.
The scientific contribution is the separation and measurement of consumer,
capacity and cost conditions, including the complete-suffix gains and failures
of cheaper repairs. The rank comparison, output contracts and measured CPU
frontier provide the evidence for that interpretation.

## Fresh strong-backbone confirmation

| Backbone | Actual selected accuracy %, 95% interval | Gain over head-only F, pp | Gain over richer frozen probe H, pp | Utility gate |
|---|---:|---:|---:|---|
| ResNet-18 | 47.45 [47.33, 47.56] | 3.58 [3.47, 3.69] | 0.49 [0.37, 0.60] | pass |
| ViT-B/32 | 77.45 [77.35, 77.54] | 2.38 [2.28, 2.47] | 1.59 [1.49, 1.68] | pass |

Each backbone has ten fresh continuation seeds, 3810--3819, on one fixed
pretrained reference, fitting/validation split and label mask. The task is
supervised fine-100 transfer; the old interface is coarse-20 classification plus
two fixed confidence thresholds. Coarse labels are derivable from fine labels.
No inaccessible-new-label assumption is needed. The intervals describe
continuation variability, not independently pretrained references or populations.
Only the stated final residual block / two attention blocks learn; full-backbone
fine-tuning and foundation-scale auxiliary benefit are not established.

The final-map, final-MLP, one-attention-block, ResNet-50 and original restricted
ResNet screens remain negative under their fixed recipes. The full U scope has
matched L/M and F/N/H controls. Every selected serving program runs from image
tensors at both registered batch sizes; final task metrics come from its actual
output arrays, not only a cached-prefix surrogate.

## Consumer fidelity and cost

Each backbone/precision has 60 checks: three output rules, two actual batch
implementations and ten current endpoints. Those are repeated checks of fixed
legacy interfaces, not 60 independently trained old tasks. The full-precision
and approximate reference outputs, negative flips, NLL/ECE and maximum logit
errors are separate records. A real-arithmetic identity is distinct from bitwise
floating-point equality or a bounded decision contract.

| Backbone / batch | Reference implementation | Current / protected ms | Protected/current, paired 95% | Mean / upper interval budget |
|---|---|---:|---:|---|
| ResNet-18/1 | FP32 reference | 1.467 / 1.652 | 1.126 [1.121, 1.131] | pass / pass |
| ResNet-18/1 | Selected reference | 1.467 / 1.652 | 1.126 [1.121, 1.131] | pass / pass |
| ResNet-18/128 | FP32 reference | 34.227 / 41.994 | 1.228 [1.201, 1.256] | fail / fail |
| ResNet-18/128 | Selected reference | 34.227 / 38.231 | 1.118 [1.095, 1.140] | fail / fail |
| ViT-B/32/1 | FP32 reference | 15.824 / 17.745 | 1.122 [1.114, 1.129] | pass / pass |
| ViT-B/32/1 | Selected reference | 15.824 / 18.080 | 1.143 [1.136, 1.150] | pass / pass |
| ViT-B/32/128 | FP32 reference | 950.892 / 1035.193 | 1.089 [1.056, 1.122] | pass / fail |
| ViT-B/32/128 | Selected reference | 950.892 / 1022.693 | 1.076 [1.057, 1.095] | pass / pass |

The budgets remain 1.20 at batch 1 and 1.10 at batch 128. Every ratio holds the
current function at FP32 and uses the development-selected current-only program.
The Xeon 8570 is BF16-capable; PyTorch 2.7.1 uses CPU tensors and empty CUDA
visibility. Per-kernel AMX dispatch is not profiled. Measurements are isolated,
with 12 warmups/31 alternating repetitions, excluding preprocessing/compilation.
Serialized program sizes are reported separately from runtime activation memory.
Timing repeats are not treated as independent training replicates.

Standard whole-encoder AMP remains an important faster absolute-time control:
its final ViT batch-128 development times are approximately 580 ms current /
647 ms protected, versus 941/1003 ms for old-only AMP with FP32 current. Whole
AMP misses its own matched 1.10 budget in that final run, despite passing an
earlier timing run. Both remain; no test selection chooses the favorable run.
The selected policy is a fixed-current-fidelity frontier on this CPU.

For the small MLPs, a fused native implementation improves absolute batch-one
time substantially, but the matched 1.20/1.10 limits remain failed. All old
decisions survive both audited implementations; two current near-ties differ
from independently merged arithmetic. The original SNR dense confirmation still
matches unrestricted joint accuracy and retains useful auxiliary regression.

## Strong cheap controls and corrected rank interpretation

All nine fixed unlabeled affine repair arms fail the visual one-percent gate
on the three development seeds. They use 2048 isolated anchors; paired maps
require old/new features, whereas marginal-statistic repairs do not. No repair
uses test outcomes or a tuned ridge. They do not support the proposed generic
CIFAR recovery by marginal rescaling alone.

The later one-forward distilled legacy-head study is explicitly registered
after unconstrained test loading. Its fixed lambda grid 1/10/100 fails all six
pilot configurations; no development expansion or test confirmation follows.
A read-only teacher/initialization audit checks every fitting view and validation
input, with zero initial class/selective differences and negligible initial KL.
The failure is conditional on the fixed loss, rates, data budget and schedule,
not an impossibility statement about all distillation methods.

Retrospective SVD truncation concerns one saved update. ReLU rescaling symmetry
can change parameter differences without changing a function. The matched
support/rank factorial holds the first-map rank ceiling at 16 while changing
orientation, bias support and degrees of freedom; its three-seed contrasts have
intervals covering zero. The dense SNR result therefore does not identify rank
alone as the cause of the original accuracy gap. The matched folded SNR graph
has 21,184 current MACs and 11,456 extra dense replay MACs (54.1%); the old 12,288
count mixed heads/bias operations and an encoder-only denominator.

## Remaining scientific scope

A new four-reference ViT confirmation supports transfer beyond the original
fitted reference, partition and label mask: D-H is +1.39 [0.80, 1.97] points;
against each reference's strongest mean F/N/H control it is +1.35 [0.88, 1.81].
The interval uses four reference means (df=3), with three continuations each.
Shared ImageNet weights and the training population remain fixed.

Real visual auxiliary benefit remains open: neither fixed reconstruction pilot
passes its joint gain/MSE gate. Full-backbone or foundation-scale auxiliary
learning and a validated adaptive online intervention also remain open. There is no broad dominance over existing backward-
compatibility methods under every data, training or hardware budget. The evidence
supports the specific interface and computation distinctions above. The main
text keeps that scope while presenting positive results directly.

## Recovered evidence and release verification

The manuscript has 8 main pages and 95 total pages.
The release contains 28 independently regenerated vector figures and 134 passing
focused structural checks. The extracted source is rebuilt and compared page
by page in both text and rendered pixels; the final result is in REVIEW_CHECKS.json.
The known SciPy SLSQP intermediate clipping warning is retained separately from
LaTeX diagnostics and does not invalidate its checked solution.

This round recovered and SHA-256 checked 380 tensor files;
all 10618 floating tensors are finite. Accuracy, decisions
and certificates are recalculated locally from recovered actual output arrays.
This output replay is not a second execution
of the pretrained backbone; the numerical driver reconstructs and hashes its
weights separately. One stopped screen's immutable cache was removed from the
CPU host only after verified local recovery; the local copy remains complete.
The ledger distinguishes cumulative recovered bytes from live remote storage.
Two full-corpus ViT jobs hit an author-set 8 GiB guard, rather than exhausting
the host. The user clarified the memory policy; missing endpoints resume under
a 256 GiB host-available reserve, up to four 12-thread workers and unchanged
numerical code/choices. All completed original records remain. This operational
change does not modify the already finished isolated timings or decision gates.
All this round's numerical queues are stopped, including explicitly retained
technical failures/reissues. All writing, statistics, figures, LaTeX compilation
and packaging take place locally. Raw datasets, official weights and trained
tensors remain outside the compact archives, with locations in the local guide.
Previous release folders and their evidence are preserved.


## New research follow-up

The complete fixed-protocol follow-up is recorded in FOLLOWUP_REPORT_20261110.md.
It adds 480 replayed routing policies, two valid real-auxiliary pilots and twelve
transfer confirmations across four fitted references. Separate CPU saved-model
replays cover fourteen study records and sixty-two branch outputs. All legacy
class and threshold decisions agree; maximum old-logit rounding error is
1.05e-5. Local checks verify returned decisions, legacy logits and reconstruction
errors. Twenty-nine returned tensor files are SHA-256 checked; all 365 floating
tensors are finite. Large canonical caches and other regenerable model states
remain outside the compact archives.

Duan (2026), arXiv:2606.15980, directly studies activation-monitor staleness and
unlabeled realignment. The paper now cites it in the introduction and related
work. The contribution cannot be the generic observation of staleness or the
invention of label-free repair. Scale interventions, specified decision rules
and the measured preservation/utility/cost comparison carry the narrower case.
The new screen failures are conditional evidence, not impossibility results.


## Argument and English finish

The current eight-page main text follows the consumer-maintenance argument.
Detailed unsigned-proxy calibration moves to the appendix; the existing consumer
disagreement inequality moves into the serving check. Mathematical statements,
proofs, tables, numerical inputs and figure data are unchanged. The language
and structural audit is in STORY_FINAL_20261110.md and STORY_INTEGRITY_20261110.json.


## Confirmed low-query diagnostic addition

The saved-model confirmation uses four fitted references and twelve endpoints.
At32/64 extra reference calls, disagreement-estimation MAE is0.509/0.502 points
versus4.773/2.964 for direct counts. The paired improvement intervals across
reference means exclude zero at both primary budgets. Augmented-view controls
and independent per-consumer rate replay retain the low-budget benefit.
The isotropic comparator performs similarly; no covariance-direction advantage
is established. This is a useful diagnostic result, not a repaired-utility result,
a new variance-reduction theorem or a distribution-free deployment certificate.
The1024 initial map-fitting reference outputs remain part of the access account.
Full results and assumptions are in SMALL_BUDGET_RISK_RESULT_20261110.md.
