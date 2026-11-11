# One-pass legacy compatibility at the useful RadioML update rate

Use the same RadioML2016.10A 12,000 fitting signals, 3% stratified
primary labels, width-64 encoder and 12-epoch healthy checkpoint as
the trajectory study. Use 24 matched epochs at Adam rate 0.0001 for
joint and detached SNR continuation. Use development seeds 8900–8907
and confirmation seeds 9000–9015. Both rates and budget are fixed from
the prior study. Develop on 1,600 training-pool signals disjoint from
the fit. In confirmation, fit and freeze every candidate before
accessing the reserved 4,400 test signals.

Each deployable program takes a signal and returns current and old
modulation logits. Evaluate class and fixed 80% and 50% confidence
decisions against the same healthy reader. Compare the following.

1. Unrestricted joint and detached training, direct and exact old
   paths, and the prior 512-anchor ridge repair.
2. A stronger fixed-feature affine fit to all 12,000 fitting windows.
   Distill old logits using a normalized centered MSE plus unit-temperature
   KL. Extend the fit with 8, 16, or 32 second-layer ReLU units driven by the
   current first-layer activation. Freeze the entire current function.
   Concatenate those units into the second affine operation and stack the
   two readouts into the output affine operation. Include width zero
   with the same optimizer and data. Train the repair for 36 epochs,
   batch 256, Adam rate 0.003, on paired healthy outputs.
3. Same-width dual distillation initialized at the joint endpoint.
   Continue for 36 epochs at Adam rate 0.0003 or 0.001. Fit the current
   head to the joint teacher and a separate affine head to the healthy
   teacher. The current loss has weight one and the old loss has weight
   one or four. Use the same normalized centered MSE plus KL and fitting
   signals. Semantic labels and SNR targets are unused in either repair
   stage. Combine the output heads with the last affine layer.

For both repair families initialize the old affine term with
ridge 0.001 on all fitting signals. For the narrow extension, initialize
the added output weights at zero. No candidate is selected using
confirmation outcomes. Rank development candidates by extra deployed
multiply-adds, then maximum disagreement, then listed order. A candidate
qualifies if all development references stay at or below 4% for every
rule and mean primary accuracy is within 0.15 percentage points of
unrestricted joint. If no candidate qualifies, confirm the candidate
with lowest maximum risk among those within the utility margin,
breaking ties with extra multiply-adds. If that set is empty, use the
same order without the utility margin. Retain every candidate in the
development records. Confirm the chosen program and the fixed full-fit
affine, old-only ridge, and exact reference controls.

The joint gate requires at most 5% disagreement for every rule on every
confirmation reference, a positive lower paired 95% confidence limit
against matched detached training, a lower limit above -0.25 points
against unrestricted joint, and zero old encoder calls. Time the folded
FP32 program against the same folded joint current-only model on an
isolated CPU queue, with 16 paired reference programs and batches 1 and
128. Use single-thread TorchScript, alternating order in nine paired
blocks after warmup. Require upper paired 95% limits for the ratio at
most 1.20 and 1.10, respectively. Report any numerical change caused by
folding and the comparison to a matched exact two-encoder executable.
The intervals are across reference seeds on a fixed benchmark pool.

Run training and timing on CPU only. Build and analyze locally, and
retain hashed per-input outputs, the fixed choice and paired timing
records. Append the confirmed outcome and all gate failures to the paper.
