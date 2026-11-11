# Compute-control confirmation

Continue the robustness addition fixed in
`2026-11-11-contrastive-compute-controls.md`. Run both supervised controls
on references 6201–6204 and continuation seeds 6310–6312. Each comparison
uses its existing matched ten-epoch joint endpoint at auxiliary weight 0.1.
All 24 control endpoints are retained.

The two controls are ten epochs of cross-entropy on both views and twenty
epochs of the original one-view cross-entropy. Keep the fitting and
validation indices, 20% label mask, two fixed views, initialization,
optimizer, regularization, learning rates and final affine reader fitting.
The model and reader definitions are unchanged. Both controls roughly
match the number of trainable-suffix evaluations. This is a matched
training-work comparison. Record elapsed times with their execution
context separately.

For each control, average the three paired accuracy differences within
each reference, then calculate the mean and two-sided 95% t interval over
the four reference means. The joint objective passes this robustness check
only if both lower endpoints exceed zero. These are marginal intervals,
and the success rule requires both comparisons. Also report the paired
held-out contrastive loss ratios. No comparator, epoch, learning rate or
seed selection follows these outcomes.

Save compact validation logits and labels alongside each full endpoint.
Recover the compact outputs and logs locally as jobs finish. Recompute
all accuracy differences from those outputs, check hashes against the
execution receipt, and match the existing joint outputs by reference and
continuation seed. The official CIFAR-100 test split stays closed.

Use CPU-only workers on the existing opaque numerical workspace
infrastructure. Check host load before launch and retain the host memory
reserve. All analysis, figures, manuscript editing and compilation run
locally. Update the paper and consolidated release with every outcome.
