# Certified Reference Precision (before implementation outcomes)

The selected full-precision suffixes have positive development utility and
unchanged old decisions. Development serving passes the 1.20 batch-1 limit,
but misses 1.10 at batch 128 (approximately 1.20 ResNet-18 and 1.112 ViT).
Preserve those failures. Packing two full-precision versions did not improve
this frontier. This implementation extension tests the CPU's BF16 arithmetic
on the protected branch, holding the entire current function at FP32.

Only reference matrix products use BF16 weights/inputs/outputs. Return their
results to FP32 before adding original FP32 biases; keep activations, norms,
attention, residuals and the old affine head at FP32. No global autocast can
affect the current path. Prefix computation is shared at FP32. Compare an
identical reference layout with FP32 products to separate layout and precision.
This approximate reference is not the exact real-arithmetic replay theorem.
Its claim concerns explicitly audited class/80/50 decisions and a finite
corpus remainder bound; retain logit error, NLL/ECE and every disagreement.

First probe only the two saved development references and cached validation
inputs. No labels select precision: every old consumer must have <=1%
disagreement against the original FP32 reference. All three development U
states are already fixed; their primary FP32 function must remain unchanged.
Reject a failed precision without changing the consumer tolerance or fitting
a correction. No temperatures, anchors, distillation or output-label fitting.

Only if the probe passes, isolated 12-thread development timing compares the
frozen fastest FP32 current ABI, modular FP32-reference control, monolithic
and modular BF16-reference protection, and two independent FP32 functions.
Use 12 warmups/31 alternating repetitions at batches 1/128. Choose the fastest
passing protected ABI per batch by absolute time, and freeze publicly before
its confirmation audit. The same 1.20/1.10 limits apply. These ratios hold
current FP32 fidelity fixed, not an optimally quantized whole-network baseline.
Hardware dependence (Xeon 8570 AMX/BF16) is explicit.

Audit ten fixed selected U endpoints per backbone, 3810--3819, and all test
inputs at both batch sizes. Primary FP32 utility and the three old consumers
are checked on actual serving outputs. Monitor 1024 inputs, RNG 14700001+seed,
alpha 0.05/2000, <=1% observed full-corpus disagreement and <=5% certified
remainder. Require the existing paired utility criteria, including >=2-point
lower gain against healthy/F and >=-0.5 against N/H. No test selection, monitor
enlargement or retraining. R18 is a new implementation audit of known states;
ViT test outcomes remain unopened at registration. Compare all precision and
FP32 outcomes, not only the successful variant.

At most two 12-thread numerical workers, 8 GiB/job and 24 GiB aggregate; timing
uses one isolated worker. Per-job wall 3600/CPU 43200 seconds; campaign 7200.
Numerical-state disk <=16 GiB after free-space checks. All authoring, statistics,
figures and manuscript work stay local; remote numerical CPU only, opaque
root folders, prompt state/log recovery with SHA-256.
