# What the risk-propagation experiment can establish

For a fixed cheap repaired logit vector Z and old-logit residual E, let
I_j=1{C_j(Z+E) differs from C_j(Z)}. If the conditional residual law is known,
m_j(Z)=P(I_j=1 | Z) is a conditional mean. Thus E[m_j(Z)]=E[I_j] and
Var(m_j(Z))<=Var(I_j). Integrating over many cheap current outputs can reduce
sampling variance relative to a small sample of binary error events.

This is the established conditional-expectation variance reduction identity.
The experiment tests whether estimating a shared residual mean/covariance from
few reference calls gives useful consumer-risk estimates in the saved update
setting. It does not introduce a new variance reduction theorem. It also does
not follow from knowing the task accuracy or total feature drift.

The plug-in Gaussian law is an assumption. Marginal residual moments alone do
not identify compatibility. In a binary example let Z=+a/-a equally likely and
E have marginal values +2a/-2a equally likely. With E=2Z no signs change; with
E=-2Z every sign changes. Both cases share the same distributions of Z and E,
including every marginal moment. Their dependence differs. Therefore a method
that propagates an input-independent residual law cannot provide a universal
decision-preservation certificate. The augmented-view stress test is relevant
to this dependence and distribution assumption.

The fair inference budget includes1024 paired map-fitting anchors plus the
additional32--256 reference monitor queries. Both covariance and direct-count
estimators use exactly the same repaired reader and monitor samples. Covariance
propagation additionally uses512 cheap current outputs; its reported savings
concern reference queries, not total end-to-end latency. No latency result or
safe deployment claim is implied. Accurate risk estimation may recommend
rejecting a failed repair rather than making it successful.
