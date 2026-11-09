# Consumer risk and convolutional continuation implementation plan

**Goal:** Test whether the failure/repair findings extend to a waveform CNN and whether a fixed, label-free consumer audit can make conservative, useful checkpoint decisions.

**Architecture:** Preserve all previous numerical drivers and artifacts. Reuse the exact paired training engine with a locally authored CNN encoder wrapper, then audit all 30 archived runs and five new CNN runs with fixed candidate repairs. Numerical execution only is remote; all design, code, statistics, plots, writing, compilation, and packaging remain local.

**Tech stack:** CPU PyTorch, NumPy/SciPy exact hypergeometric tails, the existing bounded four-worker queue, local Matplotlib and LaTeX.

## Alternatives and decision

- More MLP datasets or seeds would strengthen precision/coverage but leave the small-encoder and operational-value gaps.
- A new optimizer could target a useful-auxiliary tradeoff, but would require a separate method study and comparison, and is not justified by the current diagnostic evidence.
- Selected: one architecture extension on the identical RadioML inputs/split, plus a fixed consumer audit with explicit abstention. This targets two specific gaps without inventing a new optimizer or claiming standard normalization/alignment or disagreement bounds as novel.

The remaining primary-task-benefit gap is retained: improvement on a real reconstruction objective is not proof of a classification gain. No winner or award outcome can be promised by completing these experiments.

## Task 1: locally author and smoke-test the CNN wrapper

- Create `repro/scripts/cnn_reconstruction_continuation.py`.
- Input: unchanged flattened I/Q vectors reshaped to `2 x 128`.
- Encoder: Conv1d `2->8`, kernel 7, padding 3, ReLU, max pool 2; Conv1d `8->16`, kernel 5, padding 2, ReLU, max pool 2; flatten `16 x 32`, linear to width 64. No stochastic layer or normalization is introduced.
- Primary and reconstruction heads, losses, independent calibration split, optimizer, exact cloned states, future orders, actual-step audits, and evaluation are the unchanged paired engine.
- Save the wrapper source hash and full architecture in addition to the unchanged engine hash. Deterministic reference-head replay must reconstruct the same initialization sequence.
- Smoke on a small training/test subset, inspect CPU/resource controls and runtime before the full grid. All source authoring stays local.

## Task 2: predeclared CNN replication

- Five seeds `0..4`, fully labeled fitting subset; healthy 25 epochs, continuation 15, batch 256, Adam `1e-3`.
- Real reconstruction doses `30,300,3000`, modes `joint,detach`, plus primary-only: seven branches per seed. STF-cal is not tested here; no claim about CNN filter generalization will be made.
- Same prepared RadioML tensor cache as the MLP study. No test-selected dose, horizon, or model width. Retain weak, null, and adverse outcomes.
- At most four global single-thread workers, 8 GiB per job/24 GiB aggregate, 30 CPU minutes/40 wall minutes per job, host-memory/disk guards. No GPU. Pull partial logs/records promptly.

## Task 3: fixed consumer-risk audit

- Create `repro/scripts/consumer_risk.py`, `test_consumer_risk.py`, and `checkpoint_consumer_audit.py`.
- A fixed finite evaluation corpus is randomly partitioned after the model/repair family is fixed: 1,024 unlabeled monitor inputs and the remaining scored inputs. The monitor labels are not used to choose any repair or threshold. The fitting anchors remain disjoint training inputs.
- Monitor auditing requires healthy reference predictions on those inputs (cached for the fixed corpus or obtained by running the reference encoder). This access is distinct from the paired fitting anchors needed by some repairs and must be reported.
- Compare each candidate's predictions to the frozen healthy consumer on the monitor. Invert the exact hypergeometric CDF to upper-bound the number of changed predictions on the unobserved remainder. Loss increase relative to the healthy consumer is at most disagreement, for any labels. This is a standard control baseline, not absolute accuracy identification or a new generalization theorem.
- Reserve bank-wide error probability 0.05 over `30*10*9 + 5*7*9 = 3015` fixed endpoint/repair candidates. The simultaneous bound permits selection from the predetermined hierarchy. Report both fixed risk budgets 0.05 and 0.10, including abstentions. No test-label tuning.
- Repair order is the existing nine-arm hierarchy. Select the first candidate whose upper bound meets the risk budget. Evaluation labels enter only the subsequent score, after the decision is fixed. Check repair reconstruction and checkpoint provenance before interpretation.
- Compare frozen-consumer certifications with a declared scatter-safe heuristic (`relative scatter >= 0.3` on monitor features). Report false-safe declarations and certification coverage on joint endpoints; report primary/detached controls separately. Do not interpret an uncertified candidate as demonstrated harm.
- This audits a present checkpoint before obtaining labels. It does not forecast future optimizer steps, certify a new population, or validate monitor-adaptive encoder training. The finite-corpus random-sampling scope is particularly important for correlated HAR windows.
- Meaningful tests: exact finite-population coverage by exhaustive small-corpus enumeration, boundary cases, monotonicity, loss/disagreement sharpness, and repair selection with abstention.

## Task 4: analysis and local article revision

- Evaluate all 30 archived runs and all five new CNN runs. Keep all endpoint/candidate results and exact source/state links.
- Report CNN healthy, frozen, repaired/refitted, own-head, reconstruction, and geometry outcomes. Do not claim a stronger model without measuring its healthy accuracy.
- State clearly that unlabeled accuracy prediction, backward compatibility, and model-update leakage certificates are existing topics. Primary sources checked include Garg et al. (ICLR 2022, arXiv:2201.04234) and Usama & Qadir (November 2026 preprint, arXiv:2611.04192). Our contribution remains the controlled scale/variance/readout diagnosis and its measured operating limits.
- Promote only supported findings. Preserve a nine-page main text, compile/render locally, verify CPU-only artifacts and no active owned jobs, and build a fresh source/evidence package without replacing the previous artifact.

## Task 5: conditional head-state factorial, final bounded follow-up

The full CNN grid is near-null and leaves the historical/continuation contrast unexplained. Invoke the conditional design already specified in the previous mechanism plan, before any factorial outcome is computed. This is the final numerical extension of this round; no outcome-dependent dose search follows.

- Five archived healthy SNR references, dose 30, 15 continuation epochs, matched future batches.
- Cross auxiliary weights (warm/initial pre-warm values) with auxiliary Adam state (retained/reset). Shared encoder, primary head, and their Adam states remain identical at branching. Cold/retained is a deliberate control, not a recommended reset procedure.
- Warm/retained must reproduce the original endpoint bitwise. Run seed 0 first as a complete implementation/replay check; retain it in the full five-seed results, then execute seeds 1--4 unchanged.
- Record dense actual/counterfactual step audits in the first two epochs and selected later epochs, without attributing an entire trajectory from sparse endpoint fluxes.
- Keep all four cells, all null/adverse outcomes, readout/geometry metrics and auxiliary error. Report seed-paired contrasts, not 20 independent trials.
- These 20 endpoint states contain five exact replays and 15 new conditional trajectories. They are outside the fixed 3,015-candidate consumer bank and receive no certificate from that bank.
- Preserve global four-worker CPU, memory/time/host guards. All authoring/analysis/writing remains local. Stop numerical extensions after this factorial.
