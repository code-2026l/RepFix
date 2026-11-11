# Decision-preserving continuation (pre-registered)

**Question.** The three delivered arms each fail one axis: the reference-anchored
nullspace keeps the old logits but needs a second encoder at inference
(2.37x CPU); folding the constraint into the trunk keeps one encoder but loses
the auxiliary gain; the post-hoc chart keeps one encoder and the gain but changes
44.07% of legacy decisions. Can a *training-time* constraint on the frozen
consumer's decisions keep one encoder at inference, retain the gain, and hold
disagreement inside the five-point budget?

**Arm.** Continuation minimises `L_primary + alpha*L_aux + gamma*R`, where

    R = mean_i relu( max_{j != p0_i} z_i[j] - z_i[p0_i] ),

`z_i` are the frozen consumer's logits on the *current* features, and
`p0_i` is the frozen consumer's prediction on the healthy features. `R` is zero
at initialisation and penalises only decisions that would flip, not drift. The
frozen encoder and readout are needed during fitting only; inference is the
single shared encoder, so the cost is the ordinary cost.

**Fixed protocol.** Identical to the ten-seed benefit confirmation
(`benefit_compatibility_cpu.py`, plan `d7d9447...`): RadioML, 20% primary
labels, width 64, Adam 1e-3, batch 256, 25 warm-up epochs with the auxiliary
detached, 25 continuation epochs, dose 1.0, the same splits, the same
minibatch order, the same healthy readout from 20% labels, the same
seed-1850001+seed monitor partition and family tail. Margin fixed at 0.

**Dose selection, fixed before outcomes.** `gamma` in {0.1, 0.3, 1.0, 3.0} on
development seeds 800-802. Choose the smallest `gamma` whose validation
disagreement at the end of continuation is at most 0.05; if none reaches 0.05,
choose the largest. Readout accuracy and the auxiliary gain are not used to
select `gamma`. No retuning after confirmation outcomes.

**Decision criteria, fixed before outcomes.** The arm is credited with all three
properties only if, over ten confirmation seeds 300-309:
(i) mean auxiliary gain against the detached control is positive and its paired
descriptive 95% interval excludes zero;
(ii) the gain is within 0.5 points of the joint arm's gain;
(iii) every seed's test disagreement is at most 0.05 and its exact
finite-population upper bound at delta 0.05/4 is at most 0.05;
(iv) inference uses one encoder (by construction) and current-head logits are
the deployed logits, with no reference pass.
Partial outcomes are reported as a frontier point.

**Provenance.** CPU only, CUDA visibility empty, four threads per job, at most
twelve concurrent jobs. Source hashes, split hashes, minibatch-order hashes,
initial model/optimizer hashes and the selected `gamma` are recorded.
Development evaluates validation only; confirmation never tunes anything.

## Addendum: second corpus (HAR), written before any HAR outcome

**Fixed before running.** Domain HAR, reconstruction auxiliary, dose 10: the
highest-scoring HAR candidate in the earlier benefit development
(`benefit_development_choice.json`), chosen there by the original rule and not
re-chosen here. Everything else is unchanged: 10% primary labels, width 64,
Adam 1e-3, batch 256, 60 warm-up and 30 continuation epochs, seeds 800-802 for
the same `gamma` grid and the same selection rule, then seeds 300-309 for the
selected `gamma`.

**Reporting.** HAR's auxiliary gain at the tested doses is near zero, so the
gain axis is expected to be flat. The HAR rows therefore report the primary
gain, the auxiliary reconstruction MSE ratio against the detached control, the
disagreement, and certification. No criterion is added after the fact and the
SNR decision is not revisited.
