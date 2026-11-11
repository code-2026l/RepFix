# Early forecasts and auxiliary benefit on held-out populations

## Activity recognition

- Frozen 3%-label protocol, eight development seeds and 16 confirmation
  seeds on nine official test subjects. All eight reconstruction,
  contrastive and neighborhood-consistency candidates missed the dual
  selection gate. The fixed reconstruction fallback lowered held-out
  loss to 0.897 of the matched detached control [0.886, 0.908].
  Its primary gain was +0.06 percentage points [−0.17, 0.28].
- Forecasts at epoch three estimated maximum class or confidence-rule
  disagreement at epoch 24. MAE was 4.37 points versus 5.04 for the
  recipe/rate/reader/rule baseline and 12.23 for persistence. Paired MAE
  differences were −0.67 [−1.72, 0.38] and −7.85 [−9.61, −6.10].
  All 128 episodes exceeded 5%, so this cohort cannot establish
  specificity at that budget. Both trajectory and schedule flagged all
  episodes. Persistence missed 14.
- A graph teacher propagated the same sparse labels through fitting
  windows. Its selected weight improved development accuracy by 1.00
  points over the matched supervised arm, then lost 0.82 points
  [−1.52, −0.13] on the official cohort. The difference against a
  matched warm-model teacher was −0.22 [−0.93, 0.49]. Figure
  `fig_subject_transfer` shows gains by held-out subject. Graph setup
  averaged 0.51 CPU seconds in these jobs.
- Each HAR confirmation reproduced the healthy activity and movement
  logits bitwise on every test window. The old output uses a separate
  healthy encoder execution.

Intervals are paired by reference on a fixed held-out cohort. All
selection outcomes, including failed doses, are kept in
`repro/results/CPUCONT/`. Per-input logits and chronological forecast
receipts are in the numerical archive.

## RadioML

The epoch-six trajectory predicted epoch-24 disagreement on 16 fresh
references and six rate/objective pairs per reference. MAE was 2.25
points, versus 2.87 for schedule and 8.27 for persistence. The paired
improvements were 0.62 [0.28, 0.96] and 6.02 [4.79, 7.26] points.
Forecasts and policies were sealed before epoch seven. Each reference
used 4,400 test signals held out from fitting and monitoring.

At 5%, trajectory and schedule both retained the healthy function on
all 96 episodes. Five direct paths would have met the budget.
Persistence retained 69 healthy paths and incurred 22 violations.

With the same labels, initialization, training steps and encoder
forwards, joint SNR improved modulation accuracy by 0.35 points
[0.20, 0.50] at rate 0.0001. The simultaneous interval for that rate
is [0.16, 0.54] across the three specified rate contrasts. The lower
and higher rates have intervals that include zero. The old reader
is bitwise identical when served through the retained healthy path.

## Prospective choice of readout repair

A separate RadioML study sealed all 96 path decisions at epoch six.
The old-only ridge correction used 512 disjoint unlabeled anchors.
The current logits are shared across the three paths.

| Policy | Direct | Aligned | Reference | Violations | Max disagreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Schedule, direct only | 0 | 0 | 96 | 0 | 0.00% |
| Trajectory, direct only | 6 | 0 | 90 | 0 | 3.84% |
| Persistence with alignment | 59 | 16 | 21 | 39 | 9.77% |
| Schedule with alignment | 0 | 64 | 32 | 3 | 7.57% |
| Trajectory with alignment | 6 | 57 | 33 | 2 | 6.70% |

The prespecified practical gate did not pass. On these fresh references the SNR gain at rate 0.0001 was +0.24 points [0.05, 0.42].

Both trajectory-with-alignment violations came from class decisions.
At the middle rate, the three-path trajectory policy kept the healthy
branch for all 32 updates, including the joint runs with an accuracy gain.
