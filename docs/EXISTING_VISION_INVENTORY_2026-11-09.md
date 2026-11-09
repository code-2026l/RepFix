# Existing vision work: inventory before further computation

The adjacent local tree `D:/csy/csy/RepFix/anonymous_code/paper/research_upgrade`
contains an extensive earlier CIFAR-100/ResNet-18 research sequence. It was missed
by the initial search restricted to the current worktree. No old experiment should
be relaunched merely because it is absent from the current paper.

## Located local material

- `vision_a_v2_20261003/vision.py`: from-scratch CIFAR-style ResNet-18; 20 coarse
  primary classes and 100 fine auxiliary classes. This differs from the new
  ImageNet-pretrained, rotation-auxiliary proposal.
- BN/Adam controls, contact-location controls, coarse-only readers, conditional
  supervision, fresh-prefix confirmation, convex-reader and affine-capacity
  protocols and progress reports are present.
- `PROGRESS_READOUT_CONFIRM_RESULTS_20261005_ZH.md` reports three new training
  seeds and 36 readers. Mean linear off–full gap is 1.0533 percentage points;
  richer-reader gap is 0.3800 points. Its all-pair confirmation gate FAILED.
- `PROGRESS_CONDITIONAL_RESULTS_20261005_ZH.md` reports the failed conditional
  contact mechanism prediction. It must remain failed.
- `LATEST_PROGRESS.md` dated October 12 reports further affine-capacity and
  ordinary-kernel instrument audits. Those are reported historical outcomes,
  not newly replayed results in the current workspace.

The reports point primarily to an older `/home/customer/...` server tree. Raw
vision checkpoint/cache/audit chains have not yet been located locally in this
inventory. A progress report cannot substitute for independently checking them.
Historical instructions to render/analyze remotely are superseded for new work
by the user's current instruction: author, analyze, plot and compile locally.
Historical GPU-default drivers must not be launched under the current CPU rule.

## Current authorized CPU host

A bounded read-only directory inventory found a NYUv2 path (contents not yet
validated) and cached EfficientNet-B5/RAFT weights. The ImageNet-named Apex path
is an example-code directory, not evidence that ImageNet images are available.
No cached ResNet-18 weight was found in the inspected torch cache.

The CIFAR-100 archive was successfully staged from the Zenodo copy in 66.52 s:
169001437 bytes, official MD5 `eb9058c3a382ffc7106e4002c42a8d85`, SHA-256
`85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`.
This is an input-transport success, not a visual experiment result. The slow
original download was stopped; no new vision training completed.

## Stop and continuation decision

At the user's request to inspect existing work first, new campaign work was
stopped. The chart confirmation queue stopped with eight completed jobs and no
running children; its ten-seed confirmation is INCOMPLETE. Do not report this as
either a passed or failed scientific confirmation. Completed numerical records
and logs have been retrieved locally. The vision queue is also stopped, and the
input transfer has completed.

Next: establish the provenance and availability of the existing vision chains,
then identify a question they do not already answer. Reuse frozen features or
checkpoints for a small CPU feasibility check before considering additional
encoder training. Keep the existing failed gates and development/test separation.
