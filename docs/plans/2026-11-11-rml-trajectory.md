# Early forecasts across RadioML update regimes

Test whether an unlabeled trajectory improves on a schedule-only
prediction when both stable and risky updates are possible. Use the
RadioML2016.10A cache and the width-64 modulation/SNR MLP. Sample
12,000 fitting signals and 1,600 disjoint monitor signals from the
existing training pool. In calibration, reserve another 1,600 disjoint
training signals for the terminal endpoint. Use fixed reference seeds
8500–8511 for calibration and 8600–8615 for confirmation. In
confirmation draw 4,400 signals from the existing test pool after
sealing all forecasts and completing continuation. Seed the split and
minibatch stream by reference.

Use 3% labeled signals per modulation as the common primary label
set. Warm for 12 epochs of primary plus detached SNR regression.
Continue for 24 epochs, crossing detached or joint SNR regression
(unit weight) with Adam rates 0.00001, 0.0001 and 0.001. Branch from
the same weights and Adam state. Share minibatches and label budget.
Fit a frozen modulation reader on the same labeled fitting signals.
Check class and confidence decisions at 80% and 50% healthy coverage.
Measure all three rules on every monitored and evaluation input.

Capture reference and epoch-2 and epoch-6 logits. Fit the seven
signed-margin features, cell intercepts and reliability scale as in
the HAR protocol. A cell is the joint/detached program, exact rate and
rule. Use calibration references 8500–8507 for fitting and
8508–8511 for choosing the reliability scale, then refit on all 12.
Calibrate on labeled disagreement events from the old function,
with no task labels from monitor or evaluation signals. The
schedule baseline uses the same calibration cells and their empirical
terminal risk. Persistence uses the exact epoch-6 monitor risk.

Seal forecasts before training epoch seven. Compute absolute errors
of the maximum rule risk, averaging the six continuation errors
within a reference before paired 95% t intervals. Require negative
upper interval endpoints against both comparators. Report 5% warnings
and false alarms, with coverage separated by rate. Test an early
release rule that serves the unchanged reader below its 5% predicted
risk and keeps the healthy reference above it. Replay both paths on
all test inputs, record every violation and reference invocation,
and compare the schedule-based and persistence-based release rules.
Report utility separately and retain every completed arm.
