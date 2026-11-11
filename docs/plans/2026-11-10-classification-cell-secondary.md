# Secondary classification-contract study and consumer-family ablation

This is a separately declared secondary study, before reviewing full joint
development outcomes or any new confirmation test outcomes. It does not replace
the primary three-consumer gate in the main plan. The original paper's positive
HAR target was classification compatibility; cross-task replication of that
same target should be evaluated explicitly rather than silently equated with
the stronger threshold-preservation target introduced in this round.

Use the same six saved development checkpoints, warm Adam states, future
minibatches, epochs, doses, four-thread numerical recipe and gradient cap as
the full joint study. Add only a consumer-family ablation: cell-KL and
cell-log-odds with the classification cell alone (empty threshold list), each
on affine/residual-32 readers and coefficients 0.1,1,10,100. The buffered rank
ratio is unchanged. There are sixteen additional branches per development
record. Reuse the matched D/J and point/full-feature controls from Stage B;
require their initial model/optimizer and minibatch hashes to match exactly.
Do not rerun, retune or discard the original three-consumer branches.

For this secondary route only, development feasibility requires classification
disagreement <=0.025 on every seed and the same task-utility gates as the main
plan. Choose the smallest feasible reader architecture, then smallest feasible
coefficient, breaking ties by mean classification disagreement; accuracy does
not rank feasible candidates. Select the strongest matched point control by
the same criterion. Score both confidence-threshold consumers regardless of
eligibility, and keep all their adverse outcomes.

If the primary route passes, it remains the primary route and is confirmed as
specified. If it fails and this secondary route passes, freeze its independent
choice and publish it before the same fresh seeds 2300--2309. Use the original
primary monitoring count, 0.05/600 tail, utility intervals, native cost limits
and independent replay. A secondary complete classification result requires
all ten classification bounds <=0.05 and all stated task/cost checks. It cannot
be called a complete three-consumer result, regardless of its positive utility.
Keep its selection and verdict separate in the article, summary and artifacts.

The stronger-backbone follow-through may test this frozen secondary recipe as
an explicitly classification-only pilot. It uses the same planned visual
hyperparameters and F/U controls, reports all three consumers, and cannot open
official-test confirmation without three fixed development seeds meeting the
matching classification-only utility/compatibility gate. The stronger primary
three-consumer result remains a separate possible outcome. No target is
weakened after observing its confirmation results.

Resource scheduling follows the CPU scheduling addendum (at most six small
workers / 24 threads, 24 GiB aggregate, 8 GiB per job and 4 GiB new disk); all
writing, analysis, plots and manuscript builds remain local.
