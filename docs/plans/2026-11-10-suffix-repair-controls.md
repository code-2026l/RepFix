# Unlabeled repair controls on the selected nonlinear suffixes

The selected ResNet-18 and two-block ViT suffixes already pass their fixed
three-seed development utility gate. Before viewing ViT test outcomes, test
whether existing label-free affine repairs can remove the need for a second
suffix. This is a baseline audit, not a new learning method or a search over
training recipes. Keep the current fine-label function unchanged at FP32.

Use the original 2048 training anchors, disjoint from fitting and validation.
Recompute healthy and continued features on the same images. Fit all nine
existing `readout_alignment.ARMS` in their existing order, with their unchanged
ridge 1e-3, variance convention and eigenvalue floor. Compose their maps into
the original coarse readout; deployment needs one encoder and two affine heads.
The frozen arm applies the original affine readout directly. No labels fit a
map. Means, scales and paired feature maps have different access requirements;
report these explicitly. In particular, paired ridge uses old/new anchor
features, not merely marginal statistics. No extra normalization, temperatures,
regularization grid or trained adapter is allowed.

Evaluate classification and the two original selective decisions on all 5000
validation images, for the already fixed U states at seeds 3800--3802. Choose
the first arm in the existing hierarchy with <=1% disagreement for every
consumer and every seed. Selection does not inspect fine or coarse accuracy.
If none passes, retain the complete failure and stop this control before test.
If one passes, freeze that choice and confirm on all ten fresh endpoints, using
all test inputs and a fresh 1024-image monitor (15700001+seed, alpha .05/2000).
The original <=1% empirical and <=5% remainder gates apply. Check actual native
outputs and matched cost before claiming a useful cheaper repair. No new test
data or model recipe is selected by this audit. Existing ResNet-18 test results
are already known; this is explicitly a follow-up implementation control there.

One CPU numerical job per backbone, up to two concurrent 12-thread workers;
existing 8 GiB/job, 24 GiB aggregate, 3600-second job and 16 GiB numerical-state
caps apply. Upload code only; all analysis, writing and compilation stay local.
