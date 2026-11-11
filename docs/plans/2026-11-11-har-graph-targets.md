# HAR continuation with graph-derived activity targets

Use the same 3% labeled subset, encoder, healthy warm start, two views,
minibatches and 24 continuation epochs as the HAR utility study. Keep
nominal Adam at 0.001. The first screen found that modeling generic
feature neighborhoods did not improve activity classification. Test a
semi-supervised auxiliary that propagates the available activity labels
along those neighborhoods.

Build a symmetric 30-nearest-neighbor graph on fitting windows only.
Normalize the mean-centered feature vectors and weight edges by
exp((cosine - 1) / .08). Initialize labels with the empirical labeled
class prior and clamp the labeled nodes to their one-hot targets.
Iterate row-normalized diffusion 100 times. Fit the current head to
the cached soft targets on masked unlabeled windows, alongside the
same two-view supervised loss. The soft-target controls use the
healthy model's cached clean-view probabilities on the same windows.
Control and auxiliary arms have the same trainable parameters and
encoder passes. Graph construction is a separate one-time CPU cost.

Screen graph and warm-teacher doses .1, .3 and 1 on new training-subject
splits, seeds 8300–8307. Include both-view supervised continuation and
retain every arm. Select the graph dose with the largest mean gain
over the stronger of the matched-dose warm-teacher and supervised
controls, requiring at least 0.5 points over both. Ties favor the
smaller dose. Do not run confirmation if no dose qualifies.

With the dose frozen, use seeds 8400–8415 and the same 16 pairs of
fitting and monitoring subject definitions induced by their seeded
partitions. Evaluate the trained head on the nine official held-out
subjects. The gate requires positive lower paired 95% t endpoints
against both controls. Also report subjectwise mean gains across
seeds to show the variation across the fixed test cohort. Independently
replay the healthy activity and movement outputs on all test windows.
Compare against bitwise equality. Report graph setup time and the
separate healthy path when describing the cost of this intervention.
This is semi-supervised continuation with the same labeled budget
as its controls. No task labels from monitor or held-out subjects enter
training or model selection.
