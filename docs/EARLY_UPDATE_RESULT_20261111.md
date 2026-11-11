# Early update decisions on CIFAR-10

The study uses the final block of ImageNet-pretrained ResNet-18, a fixed
ten-class readout and two confidence gates. At epoch two, the driver
records each forecast and repair choice using 1,024 repair anchors and
512 unlabeled monitor inputs. Only then does it resume training. The
terminal evaluation uses a separate 2,000-image partition for each of six
confirmation references. [Protocol](plans/2026-11-11-early-update-diagnosis.md)

## Original confirmation

The two development references select the epoch-two persistence forecast.
Across 18 planned updates, the unchanged consumer exceeds the 5%
disagreement budget in 17 cases. The early forecast has 1.99 points of
mean absolute error, compared with 1.08 for a schedule-only development
mean. One safe case leaves the original forecasting gate unmet.

All 18 early-selected repairs meet the held-out budget. Twelve require
the full reference suffix. Five of the 17 unsafe updates use a repair with
no exact suffix, below the original half-coverage target. Across the
same planned updates, completing training gains 1.45 points [1.06, 1.84]
over stopping at epoch two, with identical current predictions under
the two old-output programs.

## Stability and limited reference access

The separate protocol was frozen before inspecting the confirmation
outcomes. Six lower-rate runs add stable controls. The extended warning
flags 15 of 17 exceedances and clears seven of seven safe cases. Both
misses are mild updates near the 5% threshold.
[Protocol](plans/2026-11-11-early-stability-and-referral.md)

The outcome-blind replay applies a fixed policy to the hashed epoch-two
monitor logits. The policy selects a repair and, when needed, sends
examples with small decision margins to the old suffix. No selected
program exceeds the 5% budget on the 24 held-out update cases. The
largest observed disagreement is 4.10%. Median reference use on the 17
unsafe updates is 25%, compared with 75% for uniform referral.
At matched query fractions, margin ranking lowers disagreement by
0.93 points [0.72, 1.13]. Over the original 18 updates, mean reference
use falls from 66.7% to 38.3%.

The nominal auxiliary comparison gives a primary difference of 0.00
points [-0.12, 0.12] versus matched supervised continuation and a loss
ratio of 0.980 [0.977, 0.982]. The accuracy gain over stopping early
comes from continuing the planned update. The reported access metric is
the number of exact reference calls in the batched replay.
