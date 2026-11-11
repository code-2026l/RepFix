# Exact one-pass visual deployment from saved continuation teachers

Test the RadioML construction on TorchVision ImageNet-pretrained
ViT-B/32 and CIFAR-100. The old coarse-class reader and its confidence
thresholds use the healthy function. Serve that function once, and
predict the current representation from its healthy final features
and final MLP activation. The student is an affine residual map on
the normalized concatenation of those features. It can be composed
with the current classifier. All student teachers get the same
fitting inputs, normalization and regression work.

The pilot uses the saved reference 6200, continuation 6300, contrastive
weight 0.1, matched two-view supervised teacher, and doubled-epoch
teacher. These teacher outcomes are already known from the earlier
study; student outcomes have not been computed. Cache both fitting
views and teacher features. The student sees the 10,000 fitting images
in their two fixed views, with no extra labels. Fit ridge residuals
from the healthy features to each teacher feature vector, standardized
with the healthy fitting-feature statistics. Set ridge coefficients
to 0.001, 0.01, 0.1 and 1, with the penalty multiplied by 20,000 fitting
examples and with an unpenalized intercept. Apply each teacher's
fixed fine-class readout to its student features. Preserve the
healthy coarse output directly in every student.

Select on pilot validation only. A candidate qualifies if its joint
student is no more than 0.25 percentage points below the joint teacher,
exceeds both matched supervised students by at least 0.2 points,
and has lower held-out contrastive loss than both. Choose the lowest
ridge among qualifying candidates. If none qualifies, retain the
pilot failure and use the coefficient maximizing the minimum joint
gain over both matched students, ties toward the smaller coefficient.
Report that fallback separately.

After freezing the coefficient, evaluate the same transformation on
the twelve saved teacher triplets from references 6201–6204 and seeds
6310–6312. Student selection never uses those outcomes. Recompute
teacher targets from each fitting pool, and evaluate students on each
reserved 5,000-image training-only validation pool. These are historical
teacher references with a new student confirmation, not fresh teacher
training or official-test confirmation. Aggregate continuations within
reference before paired 95% t intervals over four reference means.
Require positive lower gain bounds against both supervised students,
and a lower bound above -0.25 points relative to the joint teacher.
Report auxiliary loss ratios and the richer supervised frozen-feature
probe on the same inputs.

Audit native one-pass serving from raw images against the stored
healthy outputs. Check old class decisions and both fixed-confidence
rules, and check current predictions against the cached student.
Then time current-only healthy fine readout, the exact two-output
student, and the two-suffix teacher with the same FP32 healthy
reference and same prefix. Use isolated paired timing at batches
1 and 128, twelve CPU threads and alternating execution order.
Require upper 95% paired latency ratios at most 1.20 and 1.10.
All weights, protocols and publication artifacts stay local.
The remote host runs CPU calculation only and returns hashed arrays
and logs. If successful, repeat the frozen recipe on independent
pretrained weights with separate confirmation.
