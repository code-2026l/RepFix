# Shared-replay evidence and scientific assessment

The new result is a partial constructive frontier, not a completed universal
deployment method. A fixed sparse/low-rank architecture learns nonlinear
features on two tasks while retaining the old function through shared healthy
computation. Twenty fresh seeds confirm useful learning and all 120 specified
class/selective contracts. The SNR full-update accuracy and CPU cost criteria
remain failed; both stronger-backbone development screens stop before test.

## Scientific increment

The construction makes the two-version computation explicit. Sparse first-map
row support survives a coordinatewise nonlinearity; a second low-rank update
then admits an exact correction using the shared healthy trace. Rank alone
does not supply that invariant: the paper gives a rank-one weight update whose
activation change has arbitrary full rank. Independent merged-network and
gradient checks include nonzero changes that cross ReLU gates.

Low-rank adaptation, sparsity and incremental computation are established.
The article cites LoRA, DeltaCNN, Punica and S-LoRA and does not claim general
reuse, adapter serving or nullspace projection as new. The exact identity is
elementary; its scientific role is to identify a structural condition and test
its utility/cost consequences. A new method's broad novelty is not established
by renaming these components or by this literature search alone.

The healthy function uses a separate output path. This relaxes the requirement
that old weights read the new terminal features. Reference parameters remain;
the method shares their dense computation rather than discarding the reference.
It does not dominate the earlier common-feature nullspace control under identical
access assumptions. Native old heads are affine-compressed and float32; full
reference arithmetic and actual decision contracts are audited separately.

## Independent confirmation and adverse criteria

Development freezes HAR (k,r)=(4,4) and SNR (16,8) using seeds 800--802.
Fresh seeds 2500--2509 train structured detached/joint pairs and ordinary
full-update controls from matched healthy checkpoints and optimizer moments.
The original task losses alone drive continuation; no consumer loss is added.

| Outcome, paired 95% interval | HAR | SNR |
|---|---:|---:|
| Primary J-D, percentage points | -0.027 [-0.468, 0.413] | +0.678 [0.383, 0.973] |
| Auxiliary MSE J/D | 0.929 [0.917, 0.941] | 0.817 [0.782, 0.853] |
| Primary J-full J, percentage points | -0.024 [-0.689, 0.641] | -0.792 [-1.196, -0.388] |
| Certified class / selective-80 / selective-50 readers | 20/20 each | 20/20 each |

HAR passes its primary noninferiority and reconstruction criteria. SNR confirms
positive auxiliary transfer relative to the structured detached arm but fails
noninferiority to ordinary joint training. No old decision changes on either
task. The new 1024-input monitors and alpha=0.05/600 allocation are distinct
from earlier studies; all original allocations and verdicts remain unchanged.

Independent raw-state replay verifies 40 structured endpoints and 120 contracts.
The declared full reference path matches healthy logits exactly. Compressed old
float32 maximum errors are 1.27e-5 (HAR) and 8.84e-6 (SNR), with zero observed
class/selective changes. Packed versus independently merged current serving
has maximum error 1.53e-5 and two near-tied current-label differences across
the replayed endpoints. Real-arithmetic equality is not bitwise equality.

## Cost and stronger backbone

Native four-thread TorchScript packed/current ratios are 2.20/1.81 on HAR
and 2.54/2.07 on SNR at batch 1/1024. Independent-function and shared-first-map
controls show that fewer MACs do not ensure faster execution. Inductor's smaller
matched ratios still fail, and its batch-one current baseline is slower in
absolute time than TorchScript. Compilation setup and every repetition are
retained. The original 1.20/1.10 cost target remains failed.

ResNet-18/CIFAR-100 updates the last residual block with a distinct coarse
consumer and fine current task. The initial three configurations fail. A fixed
larger-capacity grid improves the best current validation score to 45.30%,
against healthy 43.74%, head-only 43.48% and dense 45.74%. It misses the required
two-point net gain, so the protocol stops at one development seed without
decoding the official test pickle. Ten completed endpoint states are independently
replayed with zero metric error. The initial folded-bias autograd stop occurs
before any continuation outcome and is kept separately from scientific failures.

## Loss route and scope

The probability-cell study has 114 frozen-feature, 216 joint-continuation and
96 class-only development endpoints. Neither task passes the three-consumer
gate. The eligible class-only HAR cell candidate is dominated by standard KL;
an explicitly additional development futility judgment stops it before test.
The article retains its analytic projection and high-confidence KL saturation
example without claiming a training advantage.

The remaining hard gap is a competitive, low-cost positive result on a stronger
backbone or a broader task/population set. There is now fresh positive transfer
on SNR and reconstruction progress on HAR with threshold contracts, but no
complete cross-task cost/utility target and no foundation-model result.

## Artifact and computation

The article has nine main pages and 81 total pages, with 23 included vector
figures. The focused suite has 91 passing checks. Standard SLSQP may emit its
documented intermediate bound-clipping warning during the independent optimizer
comparison; the solution and gradient assertions pass. No bold run-in prose
headings are introduced. Longer probe and historical filter details move to
the supplement so the main argument stays focused.

Authoring, statistical analysis, plots and LaTeX compilation are local. All
numerical campaigns use CPU with empty CUDA visibility, at most six four-thread
small-model workers or the specified eight-thread vision workers. All 630 new
scientific state files have been pulled and SHA-256 checked. The transport
ledger records approximately 1.20 GB and all owned queues finished, including
the two retained technical failures. Runtime model JIT is recorded as numerical
evaluation, separately from local manuscript compilation.

The compact source and supplement include all records, plans, historical source
bytes, scrubbed failure logs and dataset-free verification. Raw datasets and
trained tensors remain outside the submission archives. REVIEW_CHECKS.json
records the final extracted-archive tests, figure reconstruction and every
page's text/pixel comparison. The previous release is preserved.
