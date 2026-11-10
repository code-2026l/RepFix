# Pre-computation metric addendum

This addendum is written before the first scientific training outcome of the
consumer-cell study. The main plan and all prior outcomes remain immutable.
Record both plan/addendum hashes in new results. The independent toy optimizer
and gradient checks have run; no development consumer-cell fit has run.

## Why compare two cell metrics

In a binary example let the fixed consumer threshold be 1-epsilon, healthy
confidence 1-epsilon/2 and new confidence 1-2*epsilon. Both classifiers retain
the same class, but the selective decision changes. KL to the old threshold
cell and its logit gradient vanish as epsilon -> 0, even though the violation
in log-odds tends to a positive constant. A buffered KL projection alone can
therefore underweight exactly the high-confidence threshold violations exposed
by the last HAR audit. This is a mathematical scale issue, not an observed
training outcome. It does not imply a universal bound from log-odds loss.

Keep the analytic KL projection and add a second metric on the exact same
protected cell. For old class k define s=z[k]-logsumexp(z[-k]). Cell inequalities
are z[k]-z[j]>=log(r), logit(l')<=s<=logit(u'). Define the log-odds cell loss as
the batch mean of the squared largest positive inequality violation. Bounds
0/1 are omitted as appropriate. It is invariant to adding a common logit
offset and gives zero inside the cell. Global gradient clipping, identical
across all matched arms, handles its unbounded squared-violation gradient.
Actual agreement is always evaluated directly, including precision and ties.

To separate the effect of target sets from the metric, add a centered-logit
MSE baseline: mean squared error after subtracting each logit vector's mean.
This removes the irrelevant common-logit direction while retaining a point
teacher target. This baseline is an established operation, not a new method.

Stage A's exact objective list is full KL at T=1,2,4, centered-logit MSE,
cell-KL, and cell-log-odds, on all three listed reader architectures. These
18 fits plus raw OLS use the same declared initialization, batches, 60 epochs,
optimizer and clipping. Select cell-KL versus cell-log-odds by the same
development feasibility/mean-worst-disagreement rule, without ranking accuracy.
Select the strongest point-teacher baseline by the same rule on the matched
architecture; disclose which baseline wins rather than forcing T=1.

Where Stage B is required, its four objective families are full KL at T=1,
centered-logit MSE, cell-KL and cell-log-odds, each with affine/residual-32
readers and coefficients 0.1,1,10,100. Include the originally listed controls.
Keep all gates, confirmation seeds, monitoring allocations, costs and resource
limits. The same criterion chooses between the two cell objectives. Report
objective and capacity effects separately. No new objective is added after
the development outcomes are observed.
