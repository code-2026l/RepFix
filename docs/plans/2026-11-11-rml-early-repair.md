# Prospective choice of a lightweight RadioML repair

Keep the 3% label budget and paired SNR continuation protocol. Use new
calibration references 8700–8711 and confirmation references 8800–8815.
Cross detached and joint training with rates 0.000003, 0.00001 and
0.0001. Hold out 512 fitting-pool signals as unlabeled alignment
anchors, separate from fitting, monitoring and evaluation signals.

At epochs two, six and 24, compute two old-output candidates. The
direct candidate uses the unchanged reader. The aligned candidate adds
an affine residual estimated from current anchor features and healthy
anchor logits. Normalize the feature columns with anchor statistics
and use ridge 0.01 with an unpenalized intercept. No task labels enter
alignment. Add one affine map to the old readout, with no reference
encoder call on serving inputs where the map is selected.

Fit the same early forecast and comparators for both candidates,
treating repair as an additional cell key. Freeze the fitted forecasts
before confirmation. After epoch six, choose the first of direct,
aligned and exact reference whose maximum predicted risk plus
0.5 percentage points stays within 5%. The reference has zero risk
by construction. Record all decisions before epoch seven. Compare
with the same policy driven by the cell-aware schedule baseline, and
report raw-only policies and persistence as additional controls.
After epoch 24 independently replay every candidate on the 4,400
held-out signals. The practical gate is zero violations and fewer
exact reference paths than the schedule-based three-choice policy.
Also report actual terminal utility of joint vs detached training and
changes in current outputs under the selected old-only repair.
