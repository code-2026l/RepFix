# Real visual auxiliary: fixed cross-view contrastive screen

The prior reconstruction grid failed. Test a distinct real auxiliary with no
new annotations: match each image to its fixed second view while contrasting
the other images. This is standard contrastive learning, not a new loss claim.
The paper's question is whether useful auxiliary learning coexists with a retained
legacy function. Scientific novelty is not supplied by renaming SimCLR.

Use the existing ViT two-block canonical population, reference6200 and seed6300,
with its unchanged10000 fit/5000 validation/2048 anchor split,20% label mask,
two views, rates, AdamW/L2,10epochs,batch128 and final labeled-reader fitting.
The auxiliary uses terminal features standardized by the existing fixed healthy
normalizer and L2-normalized. NT-Xent temperature .2; the paired view is the
positive, self similarities excluded, all other views are negatives. No projection
head, new trainable parameters, semantic labels or official test data.

Four pilot jobs: CE-only D and joint weights .01/.1/1. All copy identical healthy
suffix/readout states, primary optimizer initialization, sample order and view RNG.
CE uses the same randomly selected view and masked labels as the earlier D.
Joint evaluates the other view for the auxiliary. Dropout stays in eval mode and
LayerNorm parameters stay frozen. The healthy path is retained separately.
No optimizer tensors are shared between jobs.

Held-out auxiliary quality is the mean NT-Xent on both fixed validation views,
fixed contiguous128-image batches (including the final short batch), raw fixed-
normalization features, no fitting on validation. Compare with dose-independent D.
Extend a weight only if final primary accuracy improves by>=.20 points, held-out
contrastive loss ratio<=.95, and the healthy consumer path is unchanged. Retain
all pilot outcomes. The D endpoint must reproduce the prior same-seed CE-only
accuracy and saved-model features before accepting any comparison.

Only passing weights extend to continuation6301/6302 at reference6200. Select
the smallest weight with all three positive primary gains, mean>=.20 points,
mean auxiliary ratio<=.95 and preserved legacy path. Only then confirm at
references6201--6204 and seeds6310--6312 with the original reference-level t
interval rule, positive primary lower bound and auxiliary-ratio upper bound<=.95.
Saved D endpoints may serve as controls only after the pilot's matched reproduction.
Actual serving fidelity/cost must be separately verified before any deployment
claim. Failure at any gate stops expansion, without changing the grid or task.

Author all code locally; CPU host executes numerical code only in opaque roots,
no CUDA, max4x12threads,256GiB host-available reserve, no arbitrary RSS cap.
Pull useful metrics and states promptly. Do not overwrite concurrent manuscript
edits. Preserve the prior certified-audit failure and coupling controls.
