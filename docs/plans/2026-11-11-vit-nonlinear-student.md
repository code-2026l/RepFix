# Nonlinear readout on the exact healthy ViT path

The affine pilot retains the auxiliary contrastive advantage but loses
0.76–1.04 accuracy points from the joint teacher. Keep that experiment.
Test a nonlinear residual directly on the fine logits, alongside an
affine residual fit at ridge 0.001. Use the same healthy features,
teacher triplets, fitting inputs, pilot and confirmation references.

Normalize the healthy final features and last MLP activations using
both fitting views. Add a 3840-to-128-or-512 GELU map and a fine-logit
output initialized at zero. Fit the output residual to each corresponding
teacher using variance-normalized centered MSE plus unit-temperature KL.
Train for 30 epochs, batch 256, Adam at 0.001 with weight decay 0.0001.
The affine map stays fixed. All three teacher students share the new
hidden initialization, input order, training epochs and objective.
Fit on the cached two views with no new semantic labels.

Select width on pilot validation. Require no more than 0.25 points
lost from the joint teacher and at least 0.2 points over each matched
supervised student. Choose the smaller qualifying width. If neither
qualifies, choose the width maximizing the smaller gain over those
controls, and label the choice as a development failure. The old coarse
and fine functions read the retained healthy features exactly. Existing
affine-student contrastive measurements remain separate from the
nonlinear student's fine-logit predictions.

Confirm the locked width on the same twelve historical teacher triplets
with four reference-level intervals. If the pilot qualifies, run the
already fixed official-test audit with the selected width. Use the
original joint teacher as the primary latency baseline and a
current-only nonlinear student as an additional comparator.
The nonlinearity adds serving computation and trainable parameters;
time the complete shared-prefix graph. Keep the previous negative
linear and RadioML distillation results in the manuscript.
