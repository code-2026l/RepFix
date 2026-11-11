# Fresh HAR confirmation and matched alignment baselines

The new construction is independently trained on ten new seeds, 1300--1309,
using HAR's existing official train/test benchmark. The public fixed protocol
and frozen development choice precede confirmation. This advances the earlier
retrospective result without replacing its original 9/10 verdict.

## Results

| Reader on protected features | reconstruction MSE / detach | primary difference, pp | legacy decisions changed | primary certificates | native CPU ratio, batch 1024 |
|---|---:|---|---:|---:|---:|
| Direct online logit head | 0.780 [0.756, 0.804] | +0.30 [-0.01, 0.61] | 1.67% | 10/10 | 1.064 |
| Training-only OLS, stored fp32 | same encoder and heads | same | 1.46% | 10/10 | 1.071 |
| Logit map | 0.777 [0.754, 0.801] | +0.42 [0.03, 0.80] | 1.64% | 9/10 | 1.067 |

Intervals are descriptive paired 95% t intervals across training seeds. The
direct and OLS arms meet the full fixed HAR criterion: reconstruction progress,
primary noninferiority with a 0.5-point margin, all ten empirical disagreements
and primary-monitor bounds within 5%, updated nonlinear layers, one encoder
call and both serving-cost limits. Primary superiority is not established for
the direct head; the demonstrated auxiliary benefit is reconstruction progress.

The primary monitor contains 1024 inputs with tail 0.05/160. All three readers
certify on the additional 2048-input monitor, but this does not rescue the logit
map's failed primary criterion. The existing HAR study and SNR failure keep
their original decisions.

## Strong baselines and selection

The comparison implements the single-update linear feature-alignment objective
of BC-Aligner (Hu et al., KDD 2022), with an explicit division by feature width.
It uses a learned bias-free 64x64 map initialized to identity. A second map has
the identical parameterization and aligns old logits instead. Distillation,
OLS and map composition are established operations, acknowledged in the paper.

Each family selects its smallest coefficient with mean validation disagreement
at most 5%, before utility is examined. On HAR, selected feature-map 0.01 loses
0.63 primary points and misses the half-point gate. Its full curve includes
viable stronger penalties; lambda 1 has -0.09 points, MSE ratio 0.715 and 1.29%
disagreement. Thus the failed gate concerns the fixed policy and does not prove
feature alignment cannot work. Only HAR logit-map 0.1 and direct-head OLS pass
development and open the new confirmation, alongside fixed direct-head controls.

All SNR candidates miss the development gate. Selected feature-map 100 changes
16.5% of predictions; logit-map 10 and direct-head OLS lose about 0.62 and 0.65
primary points. There is no new SNR test confirmation and no expanded grid.

## Serving precision and independent checks

The inherited diagnostic computes current logits in float64. A separately fixed
native audit evaluates both heads in float32, without changing models or
readers. Direct/OLS cost ratios are 1.121/1.132 at batch one and 1.064/1.071
at batch 1024. Both also satisfy the diagnostic timing criteria. All 50 native
current-head predictions match their recorded diagnostic predictions.

OLS fp32 differs from its float64 fit on 11 seed--example decisions among
29,470 comparisons, with maximum logit error 0.182. Every reported certificate
uses the actual fp32 reader. The claim is bounded decision compatibility,
not exact preservation of scores or a universal numerical identity.

All 50 fresh consumer endpoints independently replay with zero metric error,
and the fitted readers' stationarity residuals are below 1e-6. All 114 scientific
state files have been pulled locally and checked against remote SHA-256 hashes;
104 consumer tensor hashes and 10 healthy model/optimizer pairs also match.
Every owned queue is finished with no failed jobs. Peak aggregate RSS was
2,026,872,832 bytes; two workers and four threads each were the maximum.
Remote work was CPU numerical computation only. Analysis, figures, writing
and compilation remained local.

## Remaining scientific work

This confirmation strengthens the positive construction, while leaving external
generality and the scientific increment beyond established alignment/distillation
as the main research risks. It does not establish a new optimization algorithm,
foundation-model generality, information erasure, or a universal cost frontier.
The pretrained vision screens still bound the original accuracy failure. The
next valuable experiment should test a new task and stronger backbone against
the complete feature-alignment and distillation baselines, with utility included
in development acceptance and a fresh fixed confirmation.
