# Vision boundary-crossing screen (pre-registered)

**Question.** The affine first-flip boundary predicts when a frozen readout
survives an auxiliary continuation. On the bf16 rotation probe, the covariance
trace fell to 0.321 of healthy, i.e. a scale ratio of 0.567, which is above the
median first-flip scale 0.432; the frozen readout survived, as predicted. This
screen tests the other side of the same boundary on the same backbone: push the
scale below 0.432 and check that the frozen readout breaks while the
scale-invariant readout does not.

**Fixed protocol.** ImageNet-V1 ResNet-18, CIFAR-100 train split only, 64x64
input, 100-class linear head fitted on healthy features, 2,048 anchors isolated
from fitting, 5,000-example validation split, official test split never loaded.
Continuation clones the healthy model and shares minibatch order across
branches; SGD 1e-3 encoder, 1e-2 heads, momentum 0.9, weight decay 5e-4, batch
128, three epochs, batch-normalization running statistics frozen, float32
training (the bf16 rotation branch at dose 10 produced a non-finite loss and is
not part of this screen).

**Fixed branches.** Wave 1: `D` (primary cross-entropy only); `J3` and `J5`
(natural rotation auxiliary at doses 3 and 5); `S0.2`, `S0.5`, `S1.0` (controlled
scale instrument `mean(h^2)` at lambda 0.2, 0.5, 1.0, one epoch only). Wave 2:
the selected `S<lambda>` for three epochs.

**Selection rule, fixed before outcomes.** Choose the smallest lambda whose
epoch-1 trace ratio is at most 0.15; if none reaches 0.15, choose the largest
lambda. Readout accuracy is not used to select the dose. No dose or epoch is
retuned after seeing frozen-readout results.

**Decision criteria, fixed before outcomes.** A branch counts as a boundary
crossing when (i) its endpoint trace ratio is below 0.187 (scale below 0.432),
(ii) the frozen readout loses at least 10 accuracy points against healthy, (iii)
the standardized linear refit on new features stays within 2 points of healthy,
and (iv) diagonal-statistic recalibration recovers at least 80% of the frozen
loss. All branches and controls are retained whether or not they cross.

**Resources and provenance.** CPU only, CUDA visibility empty, 8 threads per
job, at most six concurrent jobs, 8 GiB RSS per job, 3,600 s wall per job.
Source files, plan hash, seed, minibatch-order hash, BN-buffer hash and endpoint
states are recorded. Results are pulled locally; no manuscript or figure is
generated on the remote host.
