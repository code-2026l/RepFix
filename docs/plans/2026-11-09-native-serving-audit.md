# Native Float32 Serving Audit

**Goal:** Check the complete serving cost and primary utility in ordinary
float32 arithmetic, in addition to the inherited float64-logit diagnostic.

**Architecture:** Use exactly the frozen fresh-confirmation models/readers.
Return current and legacy outputs from one encoder call, with both heads in
float32. No weights, precision choice, training, reader fit or repair is selected
from these results. Keep the original benchmark and all original criteria.

**Scope fixed before reading cost outcomes:** The original serving helper calls
`bc.current_logits`, which evaluates the current head in float64. This protocol
adds an isolated float32 measurement rather than treating that diagnostic as a
production kernel. The training/compatibility results are already known. This
is a retrospective implementation audit, not a new training confirmation.

Use the same four CPU threads, batches one and 1024, five warmups and 51
alternating repetitions per path. Count encoder calls. Record every seed and
protected reader. Also evaluate float32 current-head accuracy/NLL/ECE on all
saved branches and compare its predictions/logits with the recorded diagnostic.
Legacy-reader weights and their float32 evaluation are unchanged.

For a complete practical pass require BOTH the original diagnostic cost test
and the new native cost test: mean per-seed median ratio <=1.20 at batch one
and <=1.10 at batch 1024. Recheck the declared half-point primary noninferiority
using native metrics. A missed bound is retained; no kernel fusion or alternate
arithmetic is tried to rescue it. No additional scientific test outcomes are
used to change the classifier. Numerical CPU only, one isolated worker,
8 GiB RSS and 900 seconds wall. Authoring and aggregation remain local.
