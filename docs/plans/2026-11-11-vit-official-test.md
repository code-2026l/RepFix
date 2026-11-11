# Official-test confirmation of the frozen exact ViT student

This extension is fixed before reading any student outcomes. If the
cached teacher pilot advances, freeze its ridge coefficient before
opening official CIFAR-100 test images. Retain the same twelve saved
teacher triplets and student fits. The earlier validation outcomes
remain a distinct confirmation on historical training-only splits.

Compute the healthy prefix, healthy features and final MLP activations
once for all 10,000 official test images. Apply every frozen student
fine readout and each reference's healthy coarse reader to those
features. For each teacher compute its native fine logits from the
same cached prefix tokens. Audit exact old outputs on raw images,
including all 100 fine classes from a second healthy reader that was
fitted before the student. Check the two confidence rules for both
coarse and fine consumers. Save per-input logits or predictions,
thresholds and raw-to-cached differences. Student selection and
offline fitting use no test images or labels.

Require positive lower 95% t intervals for joint-student gains over
both matched supervised students across the four fitted reference
means, and noninferiority at -0.25 points to the joint teacher.
Report the teacher contrasts too. If student retention fails, retain
the official outcome without modifying the selected coefficient.
Use both direct current-only teacher and current-only student as
timing baselines. The primary cost comparison uses the original
current-only teacher to include all student transformation work.
