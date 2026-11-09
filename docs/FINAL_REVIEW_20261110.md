# Final evidence and manuscript review

This release closes the material comparison, consumer-contract and artifact
consistency work in the fixed completion plan. It does not establish a broadly
effective new adaptation algorithm or a foundation-model result.

## Scientific changes

The matched completion reuses all ten fresh HAR confirmation seeds, 1300--1309.
It is retrospective: earlier outcomes were already known. A utility-aware rule
on the previous development grid chooses full-feature alignment at lambda 1.
The original compatibility-only rule and its failed feature-map gate remain
unchanged. No new coefficient is selected on these test outcomes.

| Matched HAR arm | Primary difference to detachment (pp, 95% CI) | Reconstruction MSE / D | Mean old class changes |
|---|---:|---:|---:|
| Joint + training-only OLS | +0.02 [-0.38, 0.43] | 0.767 | 3.22% |
| Joint legacy-head distillation | +0.30 [-0.01, 0.61] | 0.780 | 1.67% |
| Full-feature alignment, lambda 1 | +0.56 [0.24, 0.88] | 0.771 | 1.92% |

Full-feature alignment is competitive. The paper no longer presents the
distilled head as the sole construction retaining useful learning. Its original
ten-seed confirmation still meets its fixed reconstruction, primary
noninferiority, decision and cost criterion. That result concerns classification
decisions under the prescribed monitor; it does not preserve the full old
function or establish primary accuracy superiority.

The new audit freezes two selective consumers at healthy-training confidence
quantiles, targeting 80% and 50% training coverage. Their actual healthy test
coverage is 71.32% and 42.65%. No auxiliary-learning reader certifies either
selective contract on any seed under the new 1024-input monitor and 0.05/1000
tail allocation. Detached + OLS certifies 4/10 and 3/10 without reconstruction progress.
Distillation changes 4.71% and 6.51% of these decisions, and OLS reduces this to
3.51% and 4.82%. These new audits do not replace earlier counts under a different
monitor and family allocation. Class, confidence-threshold and exact-score
contracts are now distinguished throughout the paper.

Total disagreement includes beneficial corrections. On joint HAR endpoints,
27.66% of classes change and 22.97% of examples undergo negative flips;
training-only OLS reduces negative flips to 1.27%, and distillation to 0.58%.
The paper cites positive-congruent training for this established distinction
and reports the complete correct/wrong flows.

An isolated native float32 benchmark returns both outputs with one encoder
call. At batch 1024, joint + OLS, feature alignment and feature + OLS cost
1.064x, 1.057x and 1.060x current-only inference; batch-one ratios are
1.109x, 1.102x and 1.110x. These measurements concern these HAR implementations,
not a cross-task reduction of the earlier SNR double-encoder cost.

## Stronger-backbone screen and stopping decision

A distinct-consumer ResNet-18 screen uses CIFAR-100 fine classes for the current
task and coarse classes for the old reader. Both healthy heads use the same
20% labeled fitting subset. The hierarchy makes coarse labels derivable from
fine labels; the paper does not claim they are inaccessible. A supervised old
label refit is included with its stronger access stated.

The one-epoch seed-1800 screen fails the preregistered utility gate: fine
accuracy is 0.3972 with the frozen encoder and 0.3594 after unconstrained
updating, a -3.78 pp difference. The completed lambda-0.1 distilled arm reaches
0.4288 fine accuracy but changes 44.56% of legacy coarse decisions. OLS raises
coarse accuracy to 0.6048 while still changing 41.20% of decisions.

Lambda 1 and 10 produce nonfinite losses. The bounded queue stops after those
failures, interrupting B(0.1) and leaving B(1,10) unlaunched. Those three B arms
are unmeasured, not adverse measured endpoints. No coefficients or learning
rates are retuned, no later development seeds are run, and the official test
split remains unopened. This is a failed recipe screen; it cannot establish
impossibility of backward-compatible learning on vision.

## Remaining scientific gaps

1. **Scientific increment.** Distillation, linear alignment, nullspace projection
   and OLS are established operations. The contribution is the specific
   scale-sensitive mechanism, signed diagnostic distinctions and matched
   consumer-contract evidence. The linear identities alone are elementary;
   a broad algorithmic novelty claim would be unsupported.
2. **Transfer of positive utility.** The complete single-encoder construction
   remains confirmed on HAR under one split and recipe. Neither the SNR arm nor
   this stronger-backbone screen supplies a second positive confirmation.
   Additional seeds on HAR do not establish task or population generality.
3. **Actual deployment contracts.** Classification agreement does not retain
   thresholds, calibrated probabilities or downstream policies. The current
   selective-consumer results expose that gap directly. No deployed production
   workload or foundation-model result is claimed.

These gaps limit the scope of the contribution. They cannot be removed by stronger wording
or by renaming a known method. The manuscript now states the supported result
without presenting local constructions as a universal law.

## Writing, figures and artifact checks

The abstract and introduction lead with the mechanism and the readout contract.
The main comparison includes the competitive full-feature baseline and the
selective-consumer boundary. Bold run-in paragraph headings remain absent.
The added figure separates harmful/corrective changes from contract-specific
acceptance, with editable vector outputs and readable final-size labels.

All authoring, statistical aggregation, plotting and LaTeX builds were local.
Remote work used at most two CPU workers, with empty CUDA visibility, bounded
RAM/time and opaque root folders. All queues are stopped or finished, including
deliberately canceled pending work. Peak aggregate RSS was 3,223,314,432 bytes;
the round occupied 665,167,397 remote bytes. All 55 saved state files were
retrieved and checked by SHA-256, with 57 new local consumer tensor hashes
verified. Original healthy and donor states were verified in the prior release.

Independent replay reproduces 100 HAR endpoint records (90 distinct readers)
and seven vision reader functions across three completed vision endpoints.
Metric differences are zero; OLS stationarity residuals satisfy the recorded
limits. Two replay implementation errors were corrected: an inherited-reader
path and a float32-versus-float64 metric mismatch. The stopped ledgers and final
successful replays are both retained. They do not alter training results.

The compact source/supplement packages exclude raw datasets and trained tensor
checkpoints. They include numerical records, protocols, source snapshots,
failure logs, figure generators and dataset-free provenance checks. The final
verification JSON records tests, all figure reconstructions and page-for-page
text/pixel comparison after an extracted-archive build.
