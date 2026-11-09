# Post-hoc affine chart and vision transfer implementation plan

> Implementation uses the local execution workflow. Source, analysis, writing,
> figures and compilation remain local. Only numerical programs and inputs are
> uploaded to an opaque remote directory; CUDA visibility stays empty.

**Goal:** Test whether post-hoc coordinate repair can retain nonlinear learning
and current-head predictions at single-encoder cost, and test a real vision
auxiliary without assuming a compatibility failure or successful repair.

**Architecture:** Train matched unconstrained D/J branches, fit label-free affine
charts from old/new anchors, invertibly rebase the current head, and fold the chart
into the final encoder affine layer after training. The old readout is approximate;
its finite-corpus disagreement is evaluated separately. This differs from the
earlier exact constraint applied throughout frozen-trunk training.

**Tech stack:** PyTorch CPU, NumPy/SciPy; torchvision ResNet-18 for vision. Plans
and scientific source hashes are recorded before development outcomes.

## Fixed chart study

- Development seeds 600--602, both HAR and RadioML; confirmation seeds 700--709
  are used only if a task passes its complete development gate.
- Use the previous compiled-control loader and fully trainable mixed-precision
  width-64 model: final encoder affine float64, trunk/head parameters fp32.
  HAR reconstruction: 10% primary fitting labels, dose 10, 60+30 epochs.
  RadioML SNR: 20%, dose 1, 25+25 epochs. Adam 1e-3, batch 256.
- Every seed clones healthy weights and complete Adam state. D/J use identical
  future orders. No trunk freezing, constraint, or choice from test outcomes.
- Fit maps on 1,024 uniformly sampled fitting inputs, without their labels. Old
  anchor features are cached offline. Diagonal-statistic and paired-ridge charts
  are two fixed candidates, in that order. Ridge is 1e-3 in standardized
  coordinates. Singular values of the resulting raw chart are floored at 1e-3
  times the largest singular value, enabling stable current-head rebasing.
- All candidates and their frozen/primary metrics are retained. No test features
  or labels are evaluated during development. A supervised refit using available
  old fitting labels is a legitimate access-rich control, never called cheating.
- Per task, choose the first candidate satisfying ALL development means: J-D
  current accuracy >0; repaired-old accuracy minus healthy >=-0.005; old NLL
  increase <=0.05; ECE increase <=0.02; all current-head validation predictions
  unchanged with maximum logit error <1e-8; identical deployed encoder shapes;
  CPU median latency ratio <=1.10 at batch 1024 and <=1.20 at batch 1.
- Confirmation, if opened, additionally requires positive lower paired 95%
  intervals for J-D and repaired-current J-D, old-accuracy difference lower
  interval >=-0.005, and the declared NLL/ECE, head-preservation and cost targets.
  A separate 40-consumer family, delta .05, 1,024 monitor inputs and .05 error
  budget evaluates reference disagreement, with abstention. The joint target
  requires all ten repaired joint consumers to certify. No automatic extension
  of the grid follows a failed gate.
- Save all models, optimizer states, fitted maps, deployment states, orders,
  hashes, metrics and timing records. Independently replay any claimed positive
  confirmation. Primary preservation is an algebraic identity; old-output
  recovery is an empirical question. Existing alignment and drift-compensation
  methods receive attribution; this is not a claimed new adaptation family.

## Fixed vision development probe

- CIFAR-100 official train/test partition, torchvision ImageNet-V1 ResNet-18,
  full network trainable during continuation. Resize inputs to 64x64; retain
  the original ResNet stem. This is a pretrained transfer protocol, not a
  from-scratch CIFAR accuracy comparison or a foundation-model experiment.
- Development seed 900; reserve 5,000 training examples for validation and
  2,048 independent random training inputs for unlabeled anchors before fitting.
  Fit the healthy linear readout on the remaining pretrained features.
- Three matched branches: primary-only D and rotation auxiliary J at doses
  1 and 10, each three continuation epochs. SGD lr 1e-3 for encoder and 1e-2
  for heads, momentum .9, weight decay 5e-4, batch 128. CPU bfloat16 training,
  fp32 evaluation. BN running statistics remain frozen in every branch;
  affine parameters can learn. Rotation labels come from transformations.
- Record every branch and evaluate frozen/current/refitted readouts, diagonal
  statistics and paired affine repairs at 128/512/1024/2048 anchors. No fitting
  on the official test images. Test outcomes stay unopened during development.
- A useful transfer signal requires healthy validation accuracy >=.25, current
  accuracy loss <=.005, frozen loss >=.05, and a fixed unlabeled repair recovering
  >=80% of that loss and ending within .01 of the healthy accuracy. A single
  development seed can only justify independent confirmation, not generality.
- Imposed scale, covariance-preserving rotations and prior-shift anchor stress
  tests are diagnostic controls, kept distinct from learned auxiliary updates.
  No desired 75-to-20-to-74 trajectory is assumed or targeted by tuning.

## Tasks and validation

1. Implement `repro/scripts/affine_chart.py` and meaningful tests for affine
   folding, current-head rebasing, bad conditioning and the marginal-ambiguity
   counterexample. Check on CPU before upload.
2. Implement `posthoc_chart_cpu.py`; smoke-check small runs, then the six fixed
   development runs. Freeze a choice only if the complete gate passes.
3. Implement `vision_chart_cpu.py`; verify rotation labels, split exclusion,
   frozen BN buffers and branch pairing before the one-seed development probe.
4. Analyze returned records locally. Retain all null/adverse outcomes and the
   exact resource limits. No positive claim is added before its specified check.

## Resource envelope

Use at most four owned workers globally: chart jobs use one CPU thread, vision
at most four; no GPU APIs or devices. Per-job RSS <=8 GiB, aggregate RSS <=24 GiB,
per-job wall <=3600 s, campaign <=7200 s, per-job CPU <=14400 s. Stop after two
failed jobs or host free memory <8 GiB. Do not touch unrelated processes.
Download only the CIFAR data and official pretrained weight inputs. Pull logs
and numerical outputs incrementally, verify hashes, and audit owned jobs at end.
