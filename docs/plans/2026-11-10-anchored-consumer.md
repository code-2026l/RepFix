# Anchored-consumer continuation (pre-registered)

**Question.** The margin hinge constrains decisions, not the consumer's function.
It is satisfied on fitting examples while drift between anchors moves test
decisions, so it cannot reach the five-point budget. Does constraining the
frozen consumer's logits on the fitting set, which pins the A-visible part of
the encoder and leaves the nullspace free, keep one encoder, retain the
auxiliary gain, and hold test disagreement inside the budget?

**Arm.** Continuation minimises `L_primary + alpha*L_aux + gamma*R_a`, where

    R_a = mean_i || z(x_i) - z_0(x_i) ||^2 / d,

`z` is the frozen consumer's logit vector on the current features and `z_0` its
logit vector on the healthy features of the same fitting example. The healthy
encoder enters fitting only; inference is the single shared encoder, so the
deployed cost is the ordinary cost and no reference pass is needed. `R_a` has a
zero gradient at initialisation when the features coincide, and it pins the
consumer's function rather than its top-1 decisions.

**Fixed protocol.** Identical to the ten-seed benefit confirmation and to the
decision-preserving study (`2026-11-09-decision-preserving.md`): RadioML, 20%
primary labels, width 64, Adam 1e-3, batch 256, 25 warm-up and 25 continuation
epochs, dose 1, the same splits, minibatch order, healthy readout from 20%
labels, the same seed-1850001+seed monitor partition and family tail. Margin
fixed at 0.

**Selection, fixed before outcomes.** `gamma` in {0.1, 1, 10, 100} on
development seeds 800-802. Choose the smallest `gamma` whose validation
disagreement at the end of continuation is at most 0.05; if none reaches 0.05,
choose the largest. Readout accuracy and the auxiliary gain are not used to
select `gamma`. No retuning after confirmation outcomes.

**Criteria, fixed before outcomes.** The arm is credited with all three
properties only if, over ten confirmation seeds 300-309:
(i) mean auxiliary gain against the detached control is positive and its paired
descriptive 95% interval excludes zero;
(ii) the gain is within 0.5 points of the joint arm's gain;
(iii) every seed's test disagreement is at most 0.05 and its exact
finite-population upper bound at delta 0.05/40 is at most 0.05;
(iv) inference uses one encoder (by construction), with the old weights applied
unchanged to the shared features and no reference pass.

**Provenance and honesty.** This arm was designed after the margin-hinge and HAR
outcomes and is therefore exploratory in origin. It is reported as a new arm,
never as a retuning of the earlier one: the earlier `gamma` selection and the
SNR decision are not revisited. Development evaluates validation only;
confirmation never tunes anything. CPU only, CUDA visibility empty, four threads
per job.

**What a negative outcome means.** If no `gamma` reaches the budget with a
positive gain, then function-space anchoring at one encoder is insufficient as
well, and the paper's frontier gains a stronger point: the cost of compatibility
is not a matter of choosing a better penalty.
