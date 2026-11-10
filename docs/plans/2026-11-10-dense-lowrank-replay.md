# Dense low-rank shared replay (pre-registered before confirmation)

**Question.** The confirmed sparse-row construction retains the healthy function
exactly but trails unrestricted joint training by 0.79 points on SNR. A
first-map update supported on k rows makes the activation difference sparse and
therefore cheap, but it may be the wrong restriction: a rank-s update that moves
every row still admits an exact replay, at the cost of one m*m correction. Does
the dense low-rank family close the utility gap, and what does the rank that
closes it cost?

**Arm.** The first affine map becomes `W1 + P Q` with `P` in R^{m x s}, `Q` in
R^{s x d}, and bias `b1 + db1`; the second becomes `W2 + U V` with rank r, bias
`b2 + db2`; the last affine map and both task heads train and are folded at
serving. The exact identity is

    t2 = t2_0 + W2_0 (a1 - a1_0) + U V a1 + db2,
    a1_0 = relu(W1 x + b1), a1 = relu(W1 x + b1 + P Q x + db1),

so the healthy dense product is evaluated once and both outputs follow. Extra
MACs are `s d + m s + m m + 2 r m + 2 n_c m`, against `k d + k m + 2 r m + n_c m`
for the sparse arm; a rank-s update reaches every row, while a k-row update
reaches k of m.

**Fixed protocol.** Identical to the confirmed sparse study: HAR dose 100 with
60 warm and 30 continuation epochs, SNR dose 1 with 25 and 25; width 64, Adam
1e-3, batch 256, primary-label fractions 0.1 and 0.2, the same splits, warm
orders, healthy checkpoints and Adam moments, the same future-order seeds, and
the same three consumers (class, coverage-80, coverage-50). Q and V are
initialised from the leading right singular vectors of the healthy first- and
second-map task gradients on the fixed 1024-input calibration set; P and U start
at zero, as in the sparse arm. No consumer loss is added.

**Grid and selection, fixed before outcomes.** `(s, r)` in
`{(16,12), (24,16), (32,24)}`, chosen by the same rule as the sparse study:
smallest analytic extra MACs among configurations that pass the development
utility gate on seeds 800--802, else the largest. The rank sweep on saved
confirmation endpoints is a separate diagnostic and does not select anything.

**Development gate.** HAR: mean primary difference to the matched detached arm
at least -0.5 points and mean reconstruction-MSE ratio to it at most 0.95; SNR:
paired 95% lower bound of the primary gain against the detached arm above zero
and mean difference to the ordinary full-update joint control at least
-0.5 points. Every consumer's observed disagreement at most 1% in both domains.

**Confirmation, fixed before outcomes.** Seeds 2520--2529, ten per domain:
paired detached/joint dense arms plus ordinary full-update detached/joint
controls from the same warm checkpoints. Criteria: the development gate holds
with paired intervals; every seed's three consumers meet the observed 1% level
and the exact finite-population bound at alpha 0.05/600 is at most 0.05; the
packed serving path is the deployed one. The original 1.20/1.10 native cost
limits remain the cost criterion and are not relaxed. Analytic extra MACs and
measured packed/current ratios are both reported.

**Honesty.** A single-seed implementation smoke (SNR seed 2500, ranks 32/24, and
HAR seed 2500, ranks 16/16) ran before this plan; it was used only to validate
the replay identity and scoring, not to choose ranks. It is recorded as
development and is not part of confirmation. The rank sweep over saved endpoints
is retrospective and is reported as a diagnostic. The released sparse study and
its verdicts are unchanged.
