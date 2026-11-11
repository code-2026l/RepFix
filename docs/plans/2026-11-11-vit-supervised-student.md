# Direct supervised nonlinear control on the retained ViT features

Before reading official-test accuracy, add a direct supervised
classifier on the same retained healthy feature concatenation.
Use the frozen width 128, healthy fitting normalization, same
20,000 view examples, 30 epochs, batch 256, Adam 0.001 and weight
decay 0.0001. Initialize its affine term at the healthy fine readout,
and its nonlinear output at zero. Allow both the affine term and
nonlinear head to learn from the same 20% fitting labels, with no
teacher targets and no additional annotations. Match input order
and hidden initialization to the distilled student.

Run all twelve historical fitted references and continuations.
Report validation and official-test primary contrasts against this
control, alongside both predeclared teacher-distilled controls.
Its old output uses the same healthy function. Keep its single-pass
predictions, with no current teacher for rescue. This additional
control tests how much of the result follows from fitting a richer
supervised readout on the retained pretrained features.
