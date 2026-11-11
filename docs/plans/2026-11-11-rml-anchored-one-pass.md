# Protect the legacy function from the start of one-pass distillation

The first one-pass development screen found that fitting the old head
after the useful SNR update left large residuals. The best jointly
distilled student was near the budget but missed the development gate.
Test two alternatives on fresh development seeds 9100–9107 and fresh
confirmation seeds 9200–9215. Use the previous matched RadioML setup,
release gates, splits and physical timing protocol. Keep all earlier
outcomes.

1. Keep the healthy first and second hidden activations and old affine
   map as an exact subnetwork. Add 0, 4, 8 or 16 second-layer ReLU units
   driven by the healthy first activation. Fit the current output on
   the concatenated activations to the unrestricted joint teacher.
   Initialize the original columns with ridge 0.001 and extra columns
   at zero. Fit on 12,000 paired fitting signals for 36 epochs with
   Adam 0.003 and the previous centered MSE plus KL. Deploy one folded
   FP32 graph. The old rows ignore the added units and reuse the
   healthy activation values. Check equivalence on all audit inputs
   because finite precision can change after folding.
2. Initialize a width-64 dual-head student at the healthy checkpoint
   with the exact legacy affine map. Distill the current and old
   logits from their respective teachers. Add a one-sided loss for
   shrinking the healthy margin at each of the three decision rules,
   capping the target preservation margin at 0.25 logit units. Use
   old loss weights 4, 12 and 36. Add twice the mean margin shortfall
   to the old MSE plus KL. Use the same fitting windows and 36 epochs
   at Adam 0.0003. Initialize the current head at the healthy head.

Use the exact same candidate selection rule as the first one-pass study.
When candidate costs tie, rank their worst development risk and then
the listed order. Also run the original 512-anchor, fixed-feature
affine and exact reference controls. Confirm the frozen candidate even
if the development gate failed, with the same stopping and reporting
rules as the first experiment. The exact architecture establishes the
old function algebraically in real arithmetic. Report the observed
FP32 agreement and distinguish that guarantee from the matched current
task gain and measured latency.
