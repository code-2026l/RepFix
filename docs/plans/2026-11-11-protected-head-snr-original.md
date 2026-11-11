# Protected-head continuation (pre-registered)

**Question.** The margin hinge pins decisions, and the logit anchor pins the
fixed consumer's function on the drifting feature; both pay in primary accuracy
because the pinned directions are also useful for learning, and the reference
control pays a second encoder. A learned readout asks a weaker question: must the
old consumer's *weights* be applied to the new feature, or only its *function* be
recoverable from it? If a trained 11-vector head can reproduce the healthy
consumer's logits from the drifting shared feature, the deployed model keeps the
old function at one encoder and one forward pass.

**Arm.** Continuation minimises

    L_primary + alpha*L_aux + lam*mean_i || p(h(x_i)) - z_0(x_i) ||^2,

where `p` is a linear head (n_classes x width) initialised to the healthy
consumer map `A, B`, trained jointly with the encoder, and `z_0` is the frozen
healthy consumer's logit vector, precomputed for the fitting set before
continuation. At inference the old prediction is `argmax p(h(x))`. One encoder,
one pass, no reference forward. The head adds `n_classes*width + n_classes`
parameters.

**Fixed protocol.** Identical to the ten-seed benefit confirmation and to the
decision-preserving study (`2026-11-09-decision-preserving.md`): RadioML, 20%
primary labels, width 64, Adam 1e-3, batch 256, 25 warm-up and 25 continuation
epochs, dose 1, the same splits, minibatch order, healthy readout from 20%
labels, the same seed-1850001+seed monitor partition and family tail. The
protected head starts from the healthy consumer map; its optimizer state starts
fresh while the encoder inherits the warm-up Adam moments.

**Selection, fixed before outcomes.** `lam` in {0.1, 1, 10} on development seeds
800-802. Choose the smallest `lam` whose validation disagreement of the
protected head against the healthy logits at the end of continuation is at most
0.05; if none reaches 0.05, choose the largest. Readout accuracy and the
auxiliary gain are not used to select `lam`. No retuning after confirmation.

**Criteria, fixed before outcomes.** The arm is credited with all four
properties only if, over ten confirmation seeds 300-309:
(i) mean auxiliary gain against the detached control is positive and its paired
descriptive 95% interval excludes zero;
(ii) the gain is within 0.5 points of the joint arm's gain;
(iii) every seed's test disagreement of the protected head, against the healthy
consumer's predictions, is at most 0.05 and its exact finite-population upper
bound at delta 0.05/40 is at most 0.05;
(iv) inference uses one encoder and one forward pass, with no reference pass;
the extra parameter count and measured latency are reported.
The fixed-weight consumer applied to the same feature is reported alongside; it
is expected to keep changing decisions, which separates function compatibility
from weight compatibility.

**Provenance and honesty.** The arm was designed after the margin-hinge, HAR and
logit-anchor outcomes and is exploratory in origin; it is pre-registered before
any protected-head result and is not a retuning of an earlier arm. Development
evaluates validation only; confirmation never tunes anything. CPU only, CUDA
visibility empty, four threads per job.

**What each outcome means.** A positive result gives a single-encoder,
one-pass construction that keeps the auxiliary gain and the old function, with
the price paid in a small learned readout rather than in accuracy or a second
pass, and the paper's dichotomy becomes three prices rather than two. A negative
result says that function compatibility is no cheaper than weight compatibility
in this regime, and the frontier is a property of the information the auxiliary
destroys, not of the constraint's parameterisation.
