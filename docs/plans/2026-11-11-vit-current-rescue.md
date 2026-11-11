# Conditional current-tail rescue for the fixed nonlinear ViT student

The width-128 pilot exceeds both matched supervised students but misses
the fixed teacher-retention margin by 0.07 points. Keep the failed pilot.
Before reading any official-test outcomes, add a conditional current
suffix. Execute the healthy ViT once for its old output and student
features. Evaluate the saved updated two-block teacher suffix only on
inputs with the smallest student top-two logit margins. The healthy
prefix tokens are shared. The old output always uses the same healthy
path.

Use the already frozen width 128. Compute a cutoff separately for each
of the three students from clean fitting images only, at the 10th,
25th or 50th margin percentile. Apply that fixed cutoff on validation
and test images, including batch 1. There is no per-batch ranking.
On the reference-6200/seed-6300 pilot choose the smallest percentile
whose joint program loses at most 0.25 accuracy points from its teacher
and exceeds both matched supervised rescue programs by at least
0.2 points. The pilot comparisons use the same percentile for all three
programs. If none qualifies, choose the percentile with the smallest
teacher gap and mark failure. Always report actual referral fractions.

Freeze the cutoff percentile before confirmation or official test.
Run the official test even if the earlier pure-student pilot failed,
using this separately fixed rescue experiment. No cutoff or student
fit uses official test data. Evaluate both the pure student and rescue
outputs. Keep both sets of primary contrasts. Repeat the same four
reference intervals and original -0.25 retention criterion. Confirm
both coarse and fine old consumers, their two confidence rules and the
actual served conditional graph. Compare full forward CPU latency
against the original current-only teacher, with exact old outputs and
all student-head and routed-tail work included. Batch-128 confidence
interval upper endpoint must be at most 1.10 and batch-1 at most 1.20.

The resulting program may execute an extra learned suffix on some
inputs. Report this explicitly alongside the fraction. The separate
healthy encoder is executed once. All earlier gates retain their
original outcomes.
