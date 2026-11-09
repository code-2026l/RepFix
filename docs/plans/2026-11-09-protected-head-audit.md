# Protected-head verification and frozen-feature control plan

**Goal:** Verify the newly supplied protected-head studies before extending them,
and separate a learned reader from encoder regularization using saved models.

**Architecture:** Preserve commits `80320da` and `da57cb4` and every original
source/result byte. A separate numerical process reconstructs all confirmation
consumers from tensors. A fixed retrospective baseline fits the healthy logits
on frozen D/J/protected features, without updating any encoder or current head.
One isolated benchmark measures current plus legacy heads in one encoder pass.

**Tech stack:** Locally authored Python/PyTorch numerical drivers, bounded remote
CPU execution, local statistics/writing/figures/LaTeX. No GPUs.

## Scope fixed before the new control

The original outcomes have already been read. This is a retrospective control
and claim audit, not a new preregistered confirmation. Keep the original HAR
9/10 and SNR 0/10 certificate counts and the original all-seed rules unchanged.
Confirmation seeds 300--309 were also used in earlier studies; they are held out
from protected-head development 800--802, not fresh across the research program.

Recover the original SNR-only plan byte string from the current file prefix
only if its SHA-256 matches the recorded `866bec...` value. The HAR addendum
has its own recorded hash. Preserve both. A filename dated November 11 is a
label; git dates these commits November 9. The commits combine plans and results
and do not by themselves establish an independently registered timestamp.

## Independent replay

Use all 20 confirmation records (10 HAR, 10 SNR). Verify each original source,
plan, healthy/endpoint model and protected-head hash, split and future-order
pairing. Recompute current, protected, fixed-weight and healthy metrics and
the original monitor bounds from saved tensors. Evaluate the declared conjunction
explicitly; a near pass cannot be serialized as a pass.

Do not edit `protected_head_cpu.py`: its timing code assigns identical operations
to ordinary/protected and separately times a legacy head. These records cannot
measure serving both outputs. Keep them and add the corrected benchmark.

## Fixed frozen-feature control

For each saved D, J and protected encoder, fit two affine legacy heads on ALL
fitting inputs and their healthy teacher logits, with no primary labels:

1. Standardized ordinary least squares, via float64 SVD with fixed tolerance
   `max(n,m) * eps * largest_singular_value`.
2. Standardized ridge regression, fixed coefficient `1e-3`, unpenalized intercept.

No hyperparameter or reader-family search, no validation selection, and no
encoder updates. Center teacher logits on fitting inputs, standardize current
features on fitting inputs (SD floor `1e-6`), then solve for all class outputs.
The old fitting logits are already an input allowed by the original online arm.
Score validation/test, current-task invariance, auxiliary MSE and teacher
disagreement. Save the new affine heads and fitting diagnostics.

This compares old-function accessibility at the fixed J encoder with the
encoder jointly trained under distillation. It can expose a sufficient post-hoc
control or a remaining approximation gap; failure does NOT establish information
loss, nor do the two fitting methods exhaust nonlinear readers.

The new finite-corpus audit uses a separately fixed family of 140 consumers:
20 endpoints × (three donor encoders × two fitted readers + one online head).
Use total delta `.05`, budget `.05`, and fresh uniform monitor membership of
1024 inputs, RNG seed `6290001 + seed + (100000 if HAR else 0)`. These candidates
are fixed from fitting inputs independently of this new monitor. Old published
bounds retain their original family and are replayed separately.

## Corrected timing

On the saved protected endpoints, alternate ordinary current inference and
inference returning BOTH current and protected logits from ONE encoder call.
Also report legacy-only inference separately. Same four CPU threads and data;
batch 1 and 1024; five warmups and 51 alternating repetitions. Time this as a
single isolated job after the replay queue, not alongside other owned workloads.
Count actual encoder calls and parameters. Report measured overhead; do not
describe a small extra head as free or infer total cost from parameter count.

## Execution and analysis

At most two numerical workers, four CPU threads each, 8 GiB RSS per worker,
24 GiB aggregate RSS, 300-second per-job wall budget and 1800-second campaign
budget. Empty CUDA visibility. Opaque remote directories and no manuscript or
rendering uploads. Retrieve numerical outputs/logs/state hashes promptly.

Analyze all outcomes locally and correct only unsupported claims, retaining the
positive HAR result. No new large encoder/third-corpus training follows before
these controls are assessed. Rebuild the integrated manuscript and repair its
source/supplementary packages after incorporating the newly supplied work.
