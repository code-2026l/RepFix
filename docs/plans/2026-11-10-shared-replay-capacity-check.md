# Fixed capacity check after the first residual-block screen

The original three-config seed-2600 screen failed its utility criterion.
Healthy fine accuracy was 0.4374, head-only 0.4348, and the dense last-block
control 0.4574. Structured accuracies were 0.4358, 0.4358 and 0.4386.
Every observed old-consumer decision was unchanged. These results remain
failed; no official test input was decoded.

Run one additional development-only capacity screen on seed 2600 with
(k,r)=(64,32),(128,64),(256,128). Match the structured update learning rate
to the dense block's 0.0001; retain head lr 0.001. Keep the original three
epochs, AdamW, data, masks, gradients, support/basis, order and healthy
heads. Repeat F and U to check matching. No coefficient search or early
stopping. The new screen tests whether the restricted update capacity
explains the original failure; it does not replace that failure.

Keep the original utility gate: current >=max(healthy,F)+0.02 and >=U-0.005,
every old-contract disagreement <=0.01. If seed 2600 passes, repeat this
entire larger grid at 2601--2602. A configuration must pass all three seeds;
rank by analytic extra MACs, then k/r. Freeze selection before five fresh
confirmation seeds 2710--2714. No reuse of any previous confirmation seed.
Official test scoring is allowed only after all development gates pass.

Confirmation requires paired 95% lower bounds of +0.02 against healthy
and F, and -0.005 against U. Require all three independently monitored
old contracts under alpha=0.05/600, 1024 monitor inputs from the 10000-input
test corpus, RNG seed 11700001+seed, five-point budget. Timing must use
the same dtype/backend/normalization for current-only and two-output
serving, include a prefix-sharing/two-dense-last-block baseline, and retain
the original <=1.20 batch-1 / <=1.10 batch-128 limits. Inspect speed only
after utility selection; a failure remains a partial frontier.

This is a last-block primary-task adaptation study. It cannot confirm
auxiliary benefit or full-backbone/foundation-model generalization.
Resources and local-writing constraints are unchanged. If the added
development grid fails, stop it before official test scoring.
