# Early diagnosis of a planned encoder update

## Setting

Use the official CIFAR-10 binary training batches (MD5 of archive
`c32a1d4ab5d03f1284b67883e8d87530`) and ImageNet-pretrained ResNet-18.
Keep the prefix through the first block of layer4 frozen and update the
final residual block. Resize to 64 pixels using the existing ImageNet
normalization. Reuse the two fixed CIFAR views, affine reader fitting,
AdamW, head learning rate 1e-4, label mask probability 0.2, batch 128,
and NT-Xent weight .1 from the visual continuation study.

A fixed permutation (seed 7400) divides the 50,000 training images into
development fitting (8,000), development policy (2,000), development
evaluation (4,000), confirmation fitting (14,000), confirmation policy
(6,000), and confirmation evaluation (16,000). Two development references
(7410–7411) and six confirmation references (7420–7425) each use 4,096
fitting images, 1,024 repair anchors, and a disjoint 512-image monitor.
Each reference has a disjoint 2,000-image evaluation partition inside
its phase. Reserve remaining confirmation evaluation images. The official
test batch is never read. Within a phase, reference fitting pools can overlap.

Use encoder learning rates 3e-6, 3e-5, and 3e-4 for mild, nominal, and
strong planned updates. The three schedules provide different drift
levels under one auxiliary objective. On the nominal rate also run the
matched supervised-only update for each confirmation reference.
The consumer is the initial ten-class affine reader plus two fixed
confidence gates with fitting coverages of 80% and 50%. Risk is the
maximum disagreement across these three rules.

## Decision made after epoch two

Collect healthy and epoch-two features on the repair anchors and the
separate monitor. Forecast epoch-eight features with
`max(0, h0 + k * (h2 - h0))`. Freeze multipliers 1, 1.5, 2, 3, and 4
as the development grid. For each forecast feature matrix, estimate all
three decision risks of the original head, per-channel moment repair,
and paired ridge repair. Fit repairs on forecast anchors and evaluate
the three risks on monitor examples. Ridge regresses the centered old
logit residual on updated features, with coefficient .001. Use the old
anchor features for moments, and old anchor logits for ridge.

Choose the multiplier with lowest mean absolute error for maximum
decision risk, averaging the three candidate families, three schedules
and two development references. Break ties toward the smaller multiplier.
Keep epoch-two persistence (k=1) as a comparator. For a second comparator,
freeze each schedule's mean terminal risk per candidate from the two
development references, and use it without monitor features. Transfer
these choices unchanged to all confirmation references.

Flag a future failure if predicted direct disagreement exceeds 5%.
Before epoch three, choose the first family in the order direct, moments,
ridge, retained FP32 suffix whose forecast maximum disagreement is at
most 4%. The 1-point gap is a fixed decision margin. Save a separate
hashed forecast receipt before any epoch-three optimizer step and before
accessing evaluation examples. Record both comparator actions too. The
fallback has zero predicted disagreement by construction. The ordering
reflects calibration access and the extra suffix evaluation.

## Paired outcome

Continue the planned trajectory to epoch eight. The chosen intervention
acts on the old output branch. Refit its already-selected repair family
on the same unlabeled anchors at epoch eight. Evaluate the direct and
selected old outputs on the same sealed 2,000 examples and pair them
with the same current output. Also score the epoch-two model as a
stop-early intervention, both other repair families, and exact
reference fallback. A post-update cheapest-safe oracle serves as a
diagnostic comparator.

The confirmation has 18 planned updates and six nominal supervised
controls. Report terminal direct-risk MAE and its paired difference from
persistence, and the schedule-only comparison. Average schedules within
each reference for 95% t intervals across six reference means. Report
true/false alarms and missed failures at the 5% limit, actual disagreement
by consumer rule, false-safe actions, fraction using the exact suffix,
primary accuracy, auxiliary loss, and accuracy relative to stopping at
epoch two. Keep every reference and schedule.

The forecasting gate requires at least three safe and three unsafe
updates, sensitivity and specificity each at least 80%, and a negative
upper 95% interval endpoint for the difference in absolute direct-risk
error from persistence. The intervention gate requires all selected
actions within 5% disagreement, unchanged current decisions, and at least
half of the unsafe updates repaired without the exact suffix. Report
the gates separately. The prospective comparison evaluates empirical
prediction error and policy decisions on the held-out examples.
The nominal auxiliary benefit is secondary and uses six paired reference
differences with intervals for accuracy and loss ratio.

Use CPU execution only. Recover compact logits, pre-epoch-three
receipts and logs locally. Analysis, writing, graphics, and compilation
run locally. Confirmatory evaluation stays separate from development.
