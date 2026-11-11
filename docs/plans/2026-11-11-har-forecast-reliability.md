# Calibrate the contribution of the early monitor

This addendum is fixed before reading the auxiliary screen or any
calibration outcomes. Keep the subject splits, forecast inputs, primary
utility gate and confirmation references in
`2026-11-11-har-utility-and-risk-transfer.md`.

Fit the cell-intercept logistic model on calibration references
8120–8127. Standardize the seven continuous and indicator predictors
from those references and fit with the stated L2 penalty. On the
separate calibration references 8128–8131, choose how much to add from
the new monitor. Compare scales 0, .25, .5, .75 and 1 on the difference
between mean predicted risk on the new monitor and mean prediction in
the fitting cell. Add the scaled difference to the fitting cell's
actual mean terminal risk. Choose the scale that minimizes MAE across
both readers and four updates, with ties toward the smaller scale.

Refit the logistic model and cell means on all 12 calibration
references, then freeze them with that scale. Report the unshrunk
forecast as a fixed sensitivity check. The comparison uses the same
cell means for both forecasts and the schedule-only baseline.

If the auxiliary screen has no qualifying candidate, retain the screen
failure and use the fixed .3 reconstruction arm for the forecast
experiment. Mark this fallback in the choice and report its matched
utility interval with the screen outcome.
