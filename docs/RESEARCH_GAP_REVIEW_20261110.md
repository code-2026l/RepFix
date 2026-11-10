# Research gaps after the refined manuscript

## Directly related work

Evan Duan's [activation-monitor staleness study](https://arxiv.org/html/2606.15980)
already evaluates frozen probes under language-model updates, predicts which
monitors degrade, and uses label-free activation realignment. RepFix should not
present those broad ideas as newly discovered. Its controlled scale intervention,
signed-gradient audit, class versus fixed-threshold distinction and actual
preservation/utility/cost comparison need to carry the specific contribution.

[Functional Alignment Can Mislead](https://proceedings.mlr.press/v267/smith25a.html)
also cautions that a successful stitch need not imply similar information.
A repair is an intervention on an interface; it does not by itself identify an
information-theoretic mechanism. RepFix's scope needs that distinction.

[Backward Compatible Weight Interpolation](https://arxiv.org/html/2301.10546)
is an inexpensive preservation baseline. Its changed current function differs
from a serving construction holding the selected FP32 current outputs fixed,
but this difference must be stated rather than treated as general superiority.

Confidence-based cascades and learned disagreement gates are established.
The proposed study compares them directly; a novel name for a cascade would
not establish a new scientific contribution.

## Testable next hypothesis

For a fixed updated model and cheap legacy reader, preserving a *set of old
decisions* requires resolving particular boundary crossings. The overlap of
class and selective errors determines the oracle amount of reference repair.
A consumer-boundary gate may exploit that structure better than a gate that
simply refers low-confidence inputs. The current task can keep its entire
learned function because the router changes only its separate legacy output.

This hypothesis has three independent checks: the minimum-boundary-margin
condition, a fixed-reader oracle repair-mass bound, and measured serving cost
at matched actual consumer fidelity. Neither the oracle nor referral fraction
alone establishes a latency advantage. A successful gate may still be dominated
by a conventional learned-disagreement gate.

## Generalization criterion

The existing visual confirmations condition on one fitted reference, split and
label mask per backbone. New continuation seeds do not resolve this dependence.
A subsequent confirmation must vary those fitted-reference/data conditions,
with its recipe and seed hierarchy fixed before results. CIFAR supervised
transfer should remain distinct from real auxiliary learning and from
foundation-scale adaptation.

All new numerical choices and failures will remain separate from the previous
release. Development outcomes cannot be promoted to independent confirmation.
