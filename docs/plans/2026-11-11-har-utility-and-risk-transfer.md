# Low-label HAR utility and early risk across readouts

Use UCI-HAR activity recognition and the existing width-64 MLP. Split
the original training subjects by seeded permutations. Development
holds out four evaluation subjects and three separate monitor subjects
per reference. Confirmation holds out four monitor subjects from the
training pool and uses the official held-out subjects for outcomes.
All splits within a reference are by subject. Sample 3% of fitting
windows within each activity as the common labeled set.

Warm up a healthy model for eight epochs of cross-entropy on clean and
masked views. For each fit minibatch mask 25% of coordinates independently
using a fixed generator and replace them by the fitting mean. Use 24
continuation epochs, batch 256 and Adam at 1e-3 for the nominal update.
Share the healthy initialization, labels, views, minibatch order and
encoder forward budget across all arms. The control averages supervised
loss on both views. Train the reconstruction decoder on detached masked
features in every arm. In the reconstruction arms, let its reconstruction
loss also reach the encoder.

Screen reconstruction weights .03, .3 and 3, NT-Xent weights .03, .1
and .3, and neighborhood-consistency weights .1 and 1 on seeds
8100–8107. Use temperature .2 for NT-Xent. For neighborhood consistency,
the row target is the softmax of clean-input cosine similarities at
temperature .05, including the paired window. Match it with clean-to-
masked representation similarities at temperature .2. Reconstruction
targets are clean fitting windows with loss scaled by fitting variance.
Choose the largest mean accuracy gain among candidates with mean gain
at least 0.5 percentage points and mean held-out auxiliary loss ratio
at most .98. Resolve ties by the listed order. Retain every candidate.

After selection, run fresh development references 8120–8131 and
confirmation references 8200–8215. Each uses four matched continuations
from one healthy checkpoint, crossing the selected auxiliary and its
two-view control with rates 1e-4 and 1e-3. Use the current trained head
for the primary endpoint. Evaluate the nominal-rate gain and selected
auxiliary loss on held-out subjects, with paired 95% t intervals across
the 16 reference differences. The utility gate is a positive accuracy
lower bound and an auxiliary loss-ratio upper bound below one. Replay
the exact healthy model and its old readers on the same inputs. The
retained output path shares the current prediction across arms and
executes the original MLP for the old output.

## Forecast transferred before the terminal update

Fit a six-class activity reader and a binary movement reader from the
healthy features of the same labeled fitting windows. Walking, upstairs
and downstairs form movement. Sitting, standing and lying form the
other class. Fit fixed confidence thresholds at 80% and 50% coverage
for each reader. Use class and both threshold decisions as old rules.

Record reference logits and the first and third epoch logits on
monitor windows. For each rule calculate the signed margin of preserving
the healthy decision. The class margin uses the healthy winner against
the largest other logit. For a confidence rule, the margin uses the
old winner's log odds against the threshold when it was accepted,
combined with its class margin. For an old abstention, use threshold
log odds minus current maximum log odds. Use margin at epochs 0, 1
and 3, the changes 0–3 and 1–3, and early disagreement indicators
as predictors. All predictors use the old output, with no semantic
labels on monitoring or evaluation inputs.

Fit a pooled ridge logistic model of terminal disagreement from the
12 development references, with a cell intercept for update recipe,
learning rate, reader and decision rule. Use L2 coefficient .01 after
standardizing continuous features on development. Calibrate per-cell
mean forecast bias from development. This preserves the same cell
prior used by the schedule-only comparator and adds the change
estimated from the new monitor. Clip predictions to [0,1].
Compare against epoch-three persistence and a schedule-only estimator
that knows the rate, objective, reader type and rule. The latter uses
the mean terminal risk in each development cell. Freeze all fitted
parameters and comparators before confirmation. Save the resulting
monitor forecasts before epoch four, and open the official test data
only after the terminal update. Keep the forecast and utility gates
separate.

For each reader, forecast the maximum of the three rule risks. Average
absolute errors across both readers and all four updates within each
reference. The forecasting gate requires negative upper 95% paired
interval endpoints for MAE differences from both comparators. Report
false alarms and missed exceedances at 5%, per-rule rates, and
terminal primary utility. All jobs run on CPU. Save compact outputs,
optimizer-stage logs and sealed forecasts for local replay. Writing and
compilation stay local.
