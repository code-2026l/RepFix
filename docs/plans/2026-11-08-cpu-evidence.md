# CPU Evidence and Manuscript Revision Implementation Plan

> Execution: implement and review each task locally in this session; remote machines execute bounded CPU calculations only.

**Goal:** Strengthen the manuscript with a paired checkpoint-continuation experiment, correct the scale-boundary interpretation, audit the certificate comparison, and rebuild an evidence-consistent local PDF.

**Architecture:** All source authoring, statistical analysis, figures, manuscript edits, and LaTeX compilation remain on the local workstation. An isolated remote directory with an opaque random name receives only calculation code and required numerical inputs. Small CPU-only jobs run with explicit thread, memory, concurrency, and time limits; completed results and logs return locally before claims are revised.

**Tech Stack:** Python, NumPy, PyTorch CPU, SSH/SFTP or SCP, existing experiment modules, local Matplotlib, local MiKTeX/Poppler.

## Constraints and provenance

- User authorization: CPU computation on the existing remote host; no GPU use; no remote writing/compiling of the article; no real project/article name in newly created remote directory names.
- Baseline: commit `61e67793d5dac843601bd47ff60debebbdaa7448` in an isolated local worktree. Preserve the previous worktrees and do not push to GitHub.
- Never upload LaTeX, PDFs, bibliography, narrative plans, SSH keys, or credentials.
- Inspect CPU affinity/cgroup quotas, available memory, load, disk, existing processes, Python packages, and scheduler availability before choosing a resource budget.
- Set `CUDA_VISIBLE_DEVICES` empty, PyTorch tensors explicitly on CPU, and BLAS/PyTorch threads explicitly bounded. No GPU discovery or allocation is needed.
- Begin with one smoke job. Use an explicit process pool or scheduler rather than shell job-count assumptions. Stop launching on low memory/disk, failed numerical checks, resource excess, or repeated failures.
- Preserve per-seed outcomes, failures, deterministic splits, checkpoint hashes, configuration, timing, resource observations, and code hashes. Do not select runs on test accuracy.

## Design choices

1. **Primary experiment (recommended):** train one healthy encoder, fit the reference readout, clone the full checkpoint and optimizer state, and continue matched branches under auxiliary contact, detachment, and selective filtering. This directly tests the deployment claim and avoids comparing independently trained coordinate systems.
2. **Complementary controls:** pure scaling of healthy features, input-statistic recalibration, readout refitting, and an unfiltered primary-only continuation. Distinguish scale mismatch from general feature drift and ordinary primary-training drift.
3. **Archive-only analysis:** recompute certificate arms under the same fixed scale as well as their reported optimized scales. This is inexpensive but cannot replace the paired causal experiment.

## Task 1: resource inventory and bounded transport

- Create local inspection/transport scripts and save a sanitized resource report locally.
- Resolve an available CPU Python environment and a minimal data source without copying the paper.
- Choose a conservative budget from measured resources; record it before launching jobs.
- Create one opaque remote directory per bounded campaign and record each mapping locally.

## Task 2: correct and verify the scale boundaries

- Add pure numerical helpers for the zero-scale bias-class exit boundary and the first flip while shrinking from the healthy scale.
- Cover multiclass intermediate flips, empty crossing sets, unique winners, positive scaling invariance, and median treatment of infinite thresholds.
- Keep the existing `0.054` result explicitly conditional on finite bias-exit thresholds and report seed variability rather than relabeling it as a universal transition.

## Task 3: paired checkpoint continuation

- Author a CPU-only experiment driver locally with strict data splits: calibration labels are excluded from encoder and head fitting.
- Train one healthy checkpoint per seed; clone model parameters, optimizer state, and subsequent minibatch order for all branches.
- Start with HAR controlled contraction, several predeclared doses, primary-only and detached references, and the existing filtering semantics.
- Record frozen/recalibrated readout metrics and representation geometry along each continuation. Fit replacement heads at predeclared endpoints only.
- Report actual optimizer-step variance changes on a fixed diagnostic batch; keep gradient-flow flux distinct from Adam increments.
- Run a single small smoke calculation first, inspect identity/control/resource checks locally, then run the fixed multiseed batch.
- Extend to a real auxiliary head only if the CPU/data budget permits and the primary protocol passes its checks.

## Task 4: certificate ablation

- Recompute from stored primitives, using the same checkpoints and scale definitions across arms.
- Separate probe correction, envelope form, and scale optimization. Report total and contraction-specific coverage, sign agreement, and exact-flux dependence.
- Treat the margin ROC as a retrospective audit; do not claim prospective cheap prediction from a score that uses the exact derivative.

## Task 5: local analysis and manuscript revision

- Pull all complete per-seed records and logs to local storage, verify hashes, and generate figures/statistics locally.
- Revise Proposition 2, the abstract and introduction, deployment protocol, certificate claims, calibration description, and related-work positioning from verified evidence only.
- Move the most informative real-head dose evidence into the main text if the page budget permits; reduce repetitive claims before shrinking typography.
- Correct the Figure 1 raw-versus-temperature legend and the overwide dissociation table.

## Task 6: local build and delivery

- Compile only locally with the existing ICLR style and bibliography.
- Check unresolved references, page count, source/data consistency, and render relevant PDF pages with Poppler for visual inspection.
- Save a concise local run/revision report including negative results and remaining limits.
- Confirm this run's remote children have exited and that no GPU was used. Keep bounded numerical artifacts until their local copies are verified; delete no unrelated remote files or processes.

## Success criteria

- At least one complete, auditable paired continuation study with independent calibration data and matched controls, or an explicit evidence-based account of why the deployment claim must be narrowed.
- Correct mathematical boundary definitions and accurate certificate comparisons.
- A locally compiled, visually checked manuscript whose claims match completed numerical records.
