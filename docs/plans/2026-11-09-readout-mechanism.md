# Readout mechanism and real-auxiliary strengthening plan

**Goal:** Resolve the scale-versus-drift ambiguity and test real reconstruction auxiliaries under the paired checkpoint protocol, then revise the local article from the complete results.

**Architecture:** Preserve the completed 20-run evidence and the previous local artifact. Author all numerical drivers and manuscript changes locally. Execute only bounded CPU calculations in fresh opaque remote directories. Pull numerical records/logs promptly. No GPU, remote article files, GitHub push, or OpenReview action.

**Baseline:** `af46e81bd104c7b7a973d3b8345b88496e89f430`; new local branch `research/readout-mechanism-20261109`.

## Scientific alternatives and selected approach

1. Increasing the number of controlled toxifier runs would narrow some seed uncertainty, but would not address whether the reported repair is scalar normalization or more general compatibility repair.
2. Adding an architecture alone would broaden coverage, but leave the mechanism and nearest alignment baselines unresolved.
3. The selected approach is a fixed hierarchy of unlabeled repairs on the already saved checkpoints, followed by matched real-reconstruction continuation studies. A small head-state control may be used to investigate the SNR protocol boundary, with its design fixed before calculation and all cells retained.

## Task 1: unlabeled repair hierarchy

- Restore each healthy reference model and reproduce the original readout exactly by restoring the deterministic initialization sequence. Require the archived healthy and frozen endpoint metrics to match before interpreting new controls.
- Fit every repair on fitting-split features only; never fit a transformation, threshold, or hyperparameter on test labels or test statistics.
- Fixed arms: frozen, mean only, scalar RMS correction, mean plus scalar RMS, per-coordinate statistics, paired diagonal regression, scaled orthogonal Procrustes, covariance alignment, and paired ridge linear alignment.
- The linear alignment ridge is fixed in standardized coordinates at `1e-3`; covariance eigenvalue flooring is fixed at `1e-6` of the largest eigenvalue. These are diagnostic baseline choices, not test-selected optima or claims to reproduce every published alignment algorithm.
- Record accuracy/NLL/ECE, reference prediction agreement, held-out alignment residual, anchor residual, and map hashes. Save the reconstructed reference head for future audit.
- Use all 10 isotropic HAR, 5 projected HAR, and 5 SNR checkpoint records. Report existing endpoint retrained heads as the decodability reference rather than silently substituting a different fit.
- Meaningful numerical tests must recover known positive scale/translation, diagonal changes, and orthogonal transformations, and demonstrate that matching covariance alone cannot identify a rotation.

## Task 2: real reconstruction auxiliaries

- Five HAR seeds: healthy 60 epochs, continuation 30 epochs, doses `1,10,100`.
- Five RadioML I/Q reconstruction seeds: healthy 25 epochs, continuation 15 epochs, doses `30,300,3000`.
- Use the unchanged paired driver and hold out calibration examples before all encoder/head fitting. Warm the auxiliary decoder on detached features; clone all model and Adam states and subsequent batches.
- Keep all joint/detached/STF-cal cells, including failures or cases where an auxiliary is not useful. Do not choose a dose on test accuracy.
- Evaluate the saved healthy and final decoders on fitting, calibration, and test inputs with no further fitting. Report reconstruction MSE beside consumer and primary-task results, so auxiliary utility is measured rather than presumed.
- Run one small smoke calculation before each new numerical code path. Global concurrency remains at most four single-thread workers; retain prior memory/disk/time guards.

## Task 3: assess the SNR boundary if needed

- At a fixed healthy checkpoint and dose 30, separate warm versus initial auxiliary weights from retained versus reset auxiliary Adam state, while keeping the encoder, primary head, their optimizer state, calibration readout, and future minibatches fixed.
- Use five seeds, retain all factorial cells, and sample initial actual updates densely enough to avoid inferring early effects from endpoint flux magnitudes.
- This is a conditional head-state experiment. It must not be presented as isolating every difference between historical from-initialization and continuation protocols.

## Task 4: local revision and validation

- Decide from the data which transformation family explains readout failure and repair. Avoid equating per-coordinate recalibration with pure scalar scaling.
- Add the closest appropriate normalization/alignment baselines and explicitly state the paired-anchor requirement.
- If a useful real auxiliary preserves or improves the current task while breaking the frozen consumer, promote that paired result; otherwise keep the limitation and do not manufacture a positive story.
- Improve the main-text hierarchy without adding redundant theorems or reporting retrospective exact-flux scores as prospective prediction.
- Compile and render locally, retain the main-text page constraint, verify record/checkpoint provenance, ensure all owned remote workers finish, and rebuild the local source/numerical artifact. Preserve the previous artifact separately.

## Scope and stopping conditions

Do not start an unbounded experiment search. Maximum four concurrent CPU workers, 8 GiB RSS per worker, 24 GiB aggregate RSS, 30 CPU minutes and 40 wall minutes per worker, with host-memory and disk guards. Start with repair diagnostics and the predeclared reconstruction grids; inspect all results before deciding whether the fixed SNR control adds necessary information. No result selection based on which cells make the article look strongest.
