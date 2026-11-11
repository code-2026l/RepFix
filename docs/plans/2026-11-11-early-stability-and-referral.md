# Stability control and limited reference queries

The six development trajectories select multiplier 1. All six direct
terminal risks exceed 5%. The original confirmation and its gates remain
fixed. The new analysis is frozen before inspection of any confirmation
outcomes. It reuses the sealed epoch-two logs and held-out output arrays
from the 18 planned updates.

Add six stability controls at encoder learning rate 3e-7, with the same
auxiliary, references 7420–7425, partitions, RNG streams and eight-epoch
schedule. Use the same early and terminal measurement code. These
controls complete a separate 24-update assessment of the 5% warning.
Use the epoch-two maximum direct decision disagreement as the forecast
and use the same 5% threshold. Require at least three safe and three
unsafe episodes, sensitivity and specificity each at least 80%. Report
the counts by schedule and the continuous MAE. No new extrapolation
parameters are fitted.

Test a policy with limited exact-reference calls. At the epoch-two
monitor, try direct, moments, and ridge without referral, in that order.
If none has maximum disagreement at most 2.5%, try ridge with exact
reference calls on the lowest decision margins at fractions 25%, 50%,
75%, and 90%, in increasing order. Use the minimum joint margin of the
class winner and both confidence decisions. Use a stable sort and round
the requested count up. Choose full reference if every candidate misses
the 2.5% target. Use exactly the same action and fraction at the terminal
evaluation. The terminal rank uses only the candidate's current logits.
Only referred examples need an old output in the policy. The replay uses
all old outputs to measure disagreement for every action.

Replay the rule on the hashed epoch-two logits, without refitting from
confirmation outcomes. For comparison, use a fixed seeded uniform
ordering at the same query fractions and with its own 2.5% monitor rule.
Report actions, actual disagreements, and realized reference-call counts
for both rules. The prespecified intervention gate is zero budget
violations at 5% and median reference-call fraction at most 50% across
unsafe episodes. Compare the original 18 episodes with the original
full-suffix policy on the same inputs. The point prediction and empirical
evaluation do not provide a future finite-corpus certificate. Query
counts describe this batched policy, with no latency conversion.

Keep the separate accuracy and auxiliary-utility comparisons, and retain
all stable and harmful cases. All training is CPU only and all analysis
and paper preparation are local.
