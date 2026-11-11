# Cheap legacy consumers from the immutable prefix

Hypothesis: a cheap reader attached to the unchanged shared prefix retains
legacy decision information more effectively than readers attached to updated
terminal features. This is established teacher compression applied at a different
access point, not a claim to invent distillation. Its value depends on measured
decision fidelity, utility and serving cost.

Preserve release 12efd68. No failed earlier gate is reopened. This is a new
development study using fitted reference 6200 and the existing canonical input
cache. Use its 10000 fitting inputs (both views), 5000 validation inputs, original
coarse teacher and fixed thresholds. No labels enter fitting; no official test.
Compare spatial/token mean pooling and richer prefix features. ResNet richer
features flatten the spatial map; ViT richer features concatenate CLS and token
mean. Standardize from fitting inputs only. Fit ridge coefficient .001 followed
by a width-256 GELU residual reader, zero output initialization, AdamW .001,
weight decay .0001, batch256, 100epochs. Use centered teacher-logit MSE normalized
by fitting target variance. Initialization/order seed6400. Retain affine and
residual endpoints for each feature set, four candidates per backbone.

Advance only candidates with <=1% disagreement for every class and fixed
threshold consumer on validation. The current model is unmodified by construction.
No candidate means stop. Passing candidates need fresh reference confirmation,
comparison with terminal-feature readers at matched training-input budget, and
isolated image-to-output latency under the original1.20/1.10 limits. No cost,
novelty or generalization claim follows from passing this initial gate alone.

All code authored locally. Numerical execution CPU only in opaque roots,
up to2x12threads, no RSS cap, 256GiB host-available reserve. Pull numerical
records/logs promptly. No paper change until evidence is complete and replayed.
