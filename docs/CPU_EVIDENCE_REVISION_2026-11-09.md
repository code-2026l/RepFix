# CPU evidence and local manuscript revision

This revision starts from `61e67793d5dac843601bd47ff60debebbdaa7448` and preserves the earlier checkout. All authoring, analysis, figures, and LaTeX compilation were local. Remote work was numerical CPU calculation only; no GPU was used and no article files were uploaded.

## Completed calculation

Twenty completed runs produce 200 continuation branch endpoints:

| Study | Seeds | Healthy / continuation epochs | Doses |
|---|---:|---:|---|
| HAR isotropic variance | 10 | 60 / 30 | 0.1, 3, 100 |
| HAR projected variance | 5 | 60 / 30 | 0.1, 3, 100 |
| RadioML real SNR head | 5 | 25 / 15 | 3, 10, 30 |

Each run has one primary-only branch and joint, detached, and STF-cal branches at each dose. Branches clone the full healthy model and Adam state and use identical subsequent minibatches. The calibration split is excluded before encoder or readout fitting. The last five isotropic HAR seeds replicate the fixed recipe after the initial batch; all outcomes are included.

The run controller allowed at most four single-thread workers, with CPU-time, wall-time, per-process memory, aggregate memory, free-memory, and free-disk guards. Monitored aggregate RSS peaked at 5,722,083,328 bytes (5.72 GB). The 20 numerical jobs accumulated 1,476 seconds of job wall time, executed in parallel. These are observed resource records, not a cost estimate.

Five opaque remote directories, including the smoke test, contain 122,051,174 bytes in total. Final inspection found no active process belonging to these runs. All 298 remote output/checkpoint/log files match their local SHA-256 copies; the uploaded bundles contain Python calculation files only. Connection details and directory mappings remain local under the ignored `tmp/` directory.

## Main evidence

At HAR isotropic dose 3, the ten-seed frozen readout falls from 0.946 to **0.453 ± 0.135**, while input-statistic recalibration reaches **0.947 ± 0.005**, a new linear head **0.952 ± 0.004**, and the jointly trained head **0.948 ± 0.006**. Detachment retains 0.943. Relative covariance trace is 0.0037 of the initial healthy reference. The paired frozen difference from primary-only continuation is -0.491, with a descriptive seed-level 95% t interval [-0.593, -0.388]. These intervals condition on this dataset and recipe.

The limits are also retained:

- At HAR isotropic dose 100, input recalibration reaches only 0.829 versus 0.935 after refitting. The trained path is not pure scalar contraction.
- The projected-variance control at dose 3 gives frozen / recalibrated / refitted accuracy 0.652 / 0.782 / 0.946. Normalization does not remove all compatibility loss.
- In healthy-checkpoint SNR continuation, dose 30 gives frozen accuracy 0.243 versus 0.304 detached, with relative scatter 0.870. The severe historical from-initialization onset does not transfer to this different protocol. Split, warm-up, head state, and horizon all differ; the comparison does not identify which factor explains the difference.
- For 75 sampled SNR joint updates, the gradient-flow variance sign and actual Adam-step sign disagree 33 times. Actual increments and same-state primary-only counterfactuals are recorded separately.

## Mathematical and measurement corrections

1. Proposition 2 now distinguishes the bias-class exit threshold while increasing scale from zero from the first prediction flip while shrinking from the healthy scale. The multiclass example `(1, 10c, -1+12c)` gives 0.1 and 0.5, respectively.
2. The historical 0.054 is explicitly the median of seed medians of finite bias-exit thresholds; seed medians range from 0.020 to 0.207. Infinite thresholds are not silently treated as a full-batch median.
3. The corrected signed balance `psi=d-alpha*a` remains defined at zero restoration. The ratio `q*=a/d` is used only when defined.
4. Matched certificate ablations separate predictor, envelope, and scale: canonical frozen triangle 2/405; corrected canonical triangle 3/405; corrected canonical combined 10/405; optimized counterparts 15, 121, and 154. The directly measured frozen exact-residual control certifies 128/405 at canonical scale. Every certificate uses exact flux; the ROC is retrospective.
5. The phase figure now compares discrepancy and envelope at the same optimized scale (median ratio 0.885). The old canonical-versus-optimized ratio of 1.19 mixed normalizations.
6. Historical calibration participation in encoder training and the missing conformal finite-sample correction are stated explicitly. New coverage is still empirical; calibration/test exchangeability is not assumed for the official HAR subject split.

## Manuscript changes

- Rewrote the abstract and introduction around the paired continuation evidence.
- Promoted the paired HAR trajectory and historical real-head dose scan to main-text Figure 2.
- Added all paired study endpoints and actual-step counts as Appendix Tables 30 and 31.
- Added backward-compatible representation learning to the related-work positioning using verified primary sources.
- Fixed the raw-versus-temperature Figure 1 legend, replaced the misleading three-arm certificate comparison, removed the overwide table layout, and hid distracting hyperlink borders.
- Removed a repeated final paragraph that occupied an otherwise almost empty appendix page.

The local PDF has **9 main-text pages and 48 pages total**, references beginning on page 10, zero unresolved references, zero overfull boxes, and blank author metadata. The five numerical scale-boundary tests pass. All 220 stored model/optimizer checkpoints pass parameter and state hash checks, and detached controls match primary-only endpoints bitwise.

## Files and regeneration

- Manuscript: `paper/RepFix_ICLR2027.tex` and `paper/RepFix_ICLR2027.pdf`.
- Full numerical summary and seed intervals: `repro/results/CPUCONT/summary.json`.
- Checkpoint/control verification: `repro/results/CPUCONT/verification.json`.
- Matched certificates: `repro/results/CPUCONT/certificate_matched.json`.
- Per-seed records and logs: `repro/results/CPUCONT/{har,replication,projected,real}/`.
- Driver: `repro/scripts/paired_readout_continuation.py`.
- Local table/figure/number generation: `repro/scripts/summarize_cpu_evidence.py`.
- Local build: `repro/scripts/build_cpu_paper.py`.

The executed driver SHA-256 is `b395215409a276cafcc074c452be0842dbc6d2197c22c49a449261b8afbb1df5`. The public source artifact includes complete numerical records and logs; regenerable trained checkpoints and private machine mappings are retained locally and excluded from Git and the source ZIP. No GitHub push or OpenReview action was performed.

This revision strengthens causal evidence and measurement validity. The results support the stated controlled and protocol-specific claims; they do not establish a universal dose law, a general forecast from the cheap probe, or guaranteed protection for arbitrary Adam trajectories.
