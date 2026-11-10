# Strong one-forward distillation control on the selected visual suffixes

This follow-up is registered after the original unconstrained visual test has
opened; its purpose is to test whether an established cheaper training control
can replace retained-reference inference. It is not a new distillation method.
Use development only for selection and fresh confirmation seeds if selected.
The unconstrained nonlinear-suffix and serving choices remain unchanged.

Use the fixed ResNet-18 final block and ViT last-two-block U recipes, references,
8705 labeled fitting inputs, two fixed views, rates, AdamW, ten epochs, final
fine-head refit and validation split. Add a learned coarse affine readout p,
initialized to the healthy coarse affine function. Its rate is 1e-4, with no
weight decay or explicit L2 on p; the current head and suffix keep the original
penalties. Minimize the existing fine CE/L2 plus lambda KL(healthy||p(current)),
temperature one, mean over examples. Teacher targets are the cached healthy
features' coarse logits, using no coarse labels or additional annotations.
Only one current suffix is required in deployment. This explicitly relaxes
fixed legacy weights into a trained replacement head.

Fixed lambda grid {1,10,100}; pilot seed 3800 on both backbones, training only
this protected U arm. Compare its final fine accuracy to the existing matched
F/N/H/U development controls. Gate: >=2 points over healthy and F, within
0.5 points of U/N/H, and <=1% disagreement for every class/80/50 old consumer
on all 5000 validation inputs. No coarse accuracy selects a candidate. If no
lambda passes for a backbone, stop before any test evaluation of this control.
For a passing backbone run all three lambdas on development seeds 3801--3802,
then choose the smallest lambda passing in all three seeds. Publish the choice.

If selected, use new confirmation seeds 3820--3829 (not the already opened
3810--3819), paired with the same original unconstrained U and residual-head
controls, with F/H deterministic final-refit controls retained. Use the original
two-point paired lower-bound and half-point noninferiority gates, original
one-percent empirical / five-percent remainder consumer limits, monitor
16700001+seed, 1024 inputs, alpha .05/2000. No lambda, threshold, precision,
normalization or recipe changes follow test. Native one-encoder/two-head cost
and actual outputs must also pass before claiming a complete deployment point.
Preserve the full pilot/grid, including every scientific failure.

At most two 12-thread CPU workers, original 8 GiB/job, 24 GiB aggregate and
3600-second job/7200-second campaign caps. Code is authored locally; numerical
CPU only in opaque roots. No duplicate representation cache is created.
Recover states and logs promptly; keep the 16 GiB numerical-result cap.
