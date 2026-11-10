# Whole-encoder AMP Control

Protected-only BF16 products pass the development decision screen on both
references. Before claiming their practical value, test standard whole-encoder
CPU BF16 autocast with the original FP32 affine consumer. This is a stronger
known numerical baseline; it is not a new method. Run all 5000 validation
inputs, with no label-dependent precision selection, and the same <=1% class,
80/50 decision gate. Keep norm parameters at their original FP32 values as
standard autocast does. Follow PyTorch 2.7's documented JIT autocast handling;
the attention fast-path compatibility setting is recorded and restored.

If this whole-encoder control passes, additionally evaluate the three already
selected development U states under the same AMP encoder policy. Require the
original two-point/half-point utility gates against FP32 F/N/H. It becomes a
competing deployment policy only if all three pass. Measure both its own
matched current-only baseline and two-output shared-prefix implementation;
never normalize it by a slower FP32 baseline. Preserve its absolute latency
even if its relative budget fails. If it fails a consumer or utility gate,
retain the failure and do not confirm that policy on test.

No new training, labels, fitting, threshold changes or output calibration.
All original fixed-reference, split, CPU resource, local-authoring and evidence
recovery constraints apply. This control is declared after the protected-only
probe, before its own outcomes and before any ViT official-test outcome.
