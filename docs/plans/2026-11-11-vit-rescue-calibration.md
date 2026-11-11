# Calibrate the rescue budget before official-test outcomes

The pure nonlinear student's validation retention varies more across
the four historical fitted references than in the first pilot.
Before computing any official-test predictions or accuracy, evaluate
the three fixed rescue percentiles on those twelve validation runs.
Healthy official-test features may already have been prepared, but
neither student nor teacher accuracy on those inputs has been read.

Treat these training-only validation results as final model-selection
data. Average the three continuations within each of the four reference
groups. Select the smallest of 10%, 25% or 50% with positive lower
95% gain bounds against both equally treated supervised students
and a lower bound above -0.25 points relative to the joint teacher.
If no candidate qualifies, use 50% and mark the calibration failure.
The original 10% single-reference pilot choice remains in the evidence.

Freeze this final percentile before official prediction jobs. The
official 10,000-image audit is the held-out evaluation for this policy.
Every old classifier continues to use the same healthy function.
The trace implementation evaluates one padded current-tail item per
batch to support an empty selected set. The reported fraction counts
real selected inputs and the physical timing includes that padding.
Time one continuation from each of the four fitted reference groups,
using the first fixed continuation seed 6310, nine alternating blocks,
twelve CPU threads, batches 1 and 128, and isolated owned queues.
Retain both the original prototype and the final calibration outcome.
