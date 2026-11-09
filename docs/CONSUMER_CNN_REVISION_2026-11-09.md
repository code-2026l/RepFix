# Consumer compatibility, CNN boundaries, and publication figures

This report describes the preserved earlier artifact. The subsequent fixed-plan
positive control and new layered figures are documented in
`PROTECTED_BENEFIT_REVISION_2026-11-09.md`; earlier findings below are retained.

The revision starts from `fa73e89c56e3e011feac0bd41368e1cc65c46d3a` on local branch `research/consumer-risk-cnn-20261109`. Earlier reports and source artifacts remain available separately. All code, analysis, figures, article writing, compilation, and packaging are local. Remote execution is numerical CPU calculation only, with opaque directories, global concurrency at most four, and the previous memory/time/host guards. GitHub and OpenReview were not changed.

## The research gaps are broader than generalization

The earlier results establish a scale-sensitive consumer mechanism and a paired real-HAR reconstruction example. They do not establish broad real-auxiliary failure across strong models, reliable warning of future failures, or primary-task gains preserved by an intervention. Novelty also requires precise positioning against backward-compatible representations, normalization/alignment, unlabeled performance estimation, and model-update auditing.

This round targets two of these issues: an encoder extension on exactly the same inputs and training recipe, and a conservative consumer decision before using new evaluation labels. The second is a standard disagreement-control baseline with explicit scope, not a new generalization theorem or an absolute accuracy estimator.

## Waveform CNN: a stronger reference, near-null auxiliary effect

The fixed CNN reshapes the unchanged I/Q vector to `2 x 128`, applies convolution/pooling blocks with 8 and 16 channels, then maps the flattened features to width 64. Heads, losses, prepared split, calibration reservation, Adam, cloned state, future batches, and actual-step audits reuse the unchanged paired engine. Five seeds use 25 healthy and 15 continuation epochs, reconstruction doses `30,300,3000`, joint/detached arms, and primary-only continuation. There are seven branches per seed. No STF-cal CNN result is claimed.

Across five seeds:

- Healthy frozen readout: **0.424**, versus **0.297** for the MLP reference.
- Continued primary-only/detached frozen readout: **0.418**.
- Joint dose 3000: frozen **0.415**, per-feature recalibration **0.425**, refitted head **0.442**, current primary head **0.437**, relative trace **1.032**.
- Test reconstruction MSE ratio to matched detachment: **0.918**, **0.851**, and **0.917** at doses 30, 300, and 3000.

Thus the larger HAR compatibility loss is not reproduced on this CNN grid. This extension strengthens the protocol's coverage and defines a boundary; it does not close the broad real-task generalization gap. There is no established primary-classification gain or evidence here about foundation models.

## Fixed-corpus unlabeled consumer control

For classification error, an update's loss increase relative to a healthy reference is at most its prediction disagreement with that reference. We combine this standard inequality with exact hypergeometric sampling. Each fixed corpus supplies 1,024 uniformly sampled unlabeled monitor inputs; the remainder is scored after decisions are fixed. A simultaneous tail allocation of `0.05 / 3015` covers the predeclared bank of 335 endpoints times nine candidate repairs. Selection is the first certified member of the existing fixed hierarchy, including identity/frozen, or abstention. Maps use fitting features only, and monitor labels do not choose candidates or thresholds. Reference metrics are also replayed independently as a provenance check.

At a 10-percentage-point relative classification-error budget, among **105 joint endpoints**:

- **49** have a certified choice: **12** keep the frozen consumer and **37** use a repair.
- **56** abstain.
- **0** selected choices have an observed loss increase above budget.
- The healthy-reference monitor-scatter heuristic declares 72 endpoints safe, including **11** whose frozen-consumer loss increase exceeds budget.

At the stricter 5-point budget, **18** choices certify (one identity and 17 repairs), **87** abstain, and no selected choice exceeds budget. The scatter heuristic has 16 false-safe declarations. These are descriptive counts of dependent endpoints; no independent-trial confidence interval is attached to them.

All certified joint choices are HAR cases. **All 45 RadioML joint endpoints abstain**, including nearly harmless continuation cases. A requirement to preserve reference decisions is conservative when many decisions change while accuracy remains similar or improves. Failure to certify is not demonstrated harm. The unsigned-probe heuristic declares 30 endpoints safe with zero observed false-safe declarations at either budget. At 10 points this is fewer declarations than the 49 consumer choices; at 5 points it exceeds the 18 choices. Their access, decisions and scopes differ, so the control is not uniformly higher-coverage or uniformly better.

The actual finite-corpus disagreement bounds have zero observed violations across all 3,015 candidates, and every choice/abstention is reproduced locally from the saved bounds. This observed result is distinct from a guarantee of zero mistakes on arbitrary future inputs.

Scope and access are material:

- The reference predictions on monitor inputs must be available, from a cache or retained healthy encoder.
- Repairs in this study use the full fitting feature set; monitor sampling and anchor extraction have separate costs.
- The random-partition guarantee covers a fixed corpus and candidate family. It needs no independence assumption for adjacent HAR windows, but does not cover new populations, future optimizer steps, or monitor-adaptive encoder training.
- The bound concerns relative 0-1 error. NLL, ECE, and conformal coverage require separate auditing. A temperature-only counterexample keeps every prediction unchanged while increasing NLL substantially.
- This is a current-checkpoint, label-free decision study. Prospective lead time and intervention benefit remain untested.

## Literature positioning and article changes

Primary-source checks confirmed the relevance of Garg et al., *Leveraging Unlabeled Data to Predict Out-of-Distribution Performance* (ICLR 2022, arXiv:2201.04234), and Usama & Qadir, *Where Did the Update Go?* (November 2026 preprint, arXiv:2611.04192). The related work and appendix acknowledge these areas. The contribution remains the controlled scale/variance/readout diagnosis and its measured limits. The standard consumer control is not presented as a novel estimator.

The main text now includes the CNN boundary result and consumer decision/abstention results. Algorithm 1 clearly separates a gradient-flow sign certificate from a consumer classification-error control. Its ambiguous instruction to certify a batch and “update model” is replaced by an explicit gradient-flow scope. New appendix sections give the architecture, full grid, sampling proof, access assumptions, all audit outcomes, and limits. Redundant overview citations are replaced by a more focused primary-source task-grouping citation; full venue names are standardized without changing paper titles, authors, or years.

The local PDF has **nine actual main-text pages and 55 total pages**, references starting on page 10, zero unresolved references, zero overfull boxes, and blank author metadata. The build checks where the main text actually ends before the statement pages, including continuation text and floats. Local all-page rendering and selected page/figure inspection verify layout. The **26 tests** cover scale boundaries, repair identifiability, exact small-corpus coverage, monotonicity, selection/abstention, loss sharpness, and the NLL blind spot.

## The main story and all eleven figures

The abstract and introduction now follow one evidence chain: a shared update can preserve current-task decodability and improve a real auxiliary while breaking a retained consumer; the repair hierarchy identifies the scale component and richer drift; signed update audits delimit what the diagnostic can establish. The opening figure defines the three interfaces before comparing controlled and real-objective results. The isolated pure-scale sweep moves to the mechanism appendix. The main text distinguishes post-hoc input repair from training-time auxiliary filtering.

The five archived HAR reconstruction checkpoints are additionally evaluated with the complete existing repair hierarchy on the full original test set. All 450 map hashes (five seeds, ten endpoints, nine maps) and checkpoint hashes match their previously fixed consumer-bank counterparts. At joint dose 100, full-test paired linear alignment reaches **0.947**, versus **0.826** frozen and **0.941** for the current task head. No monitor/remainder accuracy is mixed into this plot. The input map leaves encoder and auxiliary head unchanged, so it preserves the learned reconstruction error. This is a known alignment control requiring paired training anchors, not a new algorithm or an established primary-task gain. Records and all seed dispersions are shipped in `recon_repairs/` and `reconstruction_repair_summary.json`.

All **eleven** included figures are regenerated at the final 5.5-inch width as vector PDF and editable SVG, with PNG previews. The common font, readable final-size labels, restrained colors, line weights, and uncertainty conventions replace mixed plot styles. The opening panels show every seed and SD, with identical zero-based accuracy axes. Conservative consumer choices show abstentions explicitly with hatched regions. Legacy restyling asserts unchanged plotted coordinates, uncertainty paths, areas, and heat-map values. The numerical content of the older evidence audit remains equal. Text-bound and vector-font checks run before export; final pages are also inspected visually. `publication_figure_manifest.json` and `story_figure_manifest.json` retain source/output hashes, physical sizes and plotted-data hashes. `render_paper_figures.py` is the local entry point for all eleven assets.

The figure design follows the evidence-first organization of public author versions of ICLR/NeurIPS award papers, without copying assets or data; references and design decisions are recorded in `docs/plans/2026-11-09-story-and-figures.md`. Design polish does not change an experimental result or imply an award prediction.

## Conditional head-state factorial

The final predeclared numerical extension crosses warm/initial auxiliary weights with retained/reset auxiliary Adam state at five existing healthy SNR references, dose 30 and a fixed 15-epoch continuation. Shared encoder/primary parameters and moments, frozen consumer, and future minibatches match. The initial/retained cell is an artificial factor-isolating control, not a reset recommendation.

Warm/retained reproduces the old full model and Adam endpoint bitwise for all five seeds. Frozen accuracies are **0.243**, **0.241**, **0.244**, and **0.226** for warm/retained, warm/reset, initial/retained, and initial/reset; relative traces are **0.639**, **0.643**, **0.726**, and **0.888**. All four cells have zero scatter-collapse endpoints relative to the matched primary-only continuation. Auxiliary state alone does not reproduce the historical severe-collapse gap. The paired accuracy contrasts have wide descriptive intervals; no significant ordering is claimed.

The 620 sampled update observations are dependent. First-step primary counterfactuals match across cells, and moment-only changes leave the first encoder increment unchanged. The 20 saved endpoint states comprise five exact replays and 15 new conditional trajectories, explicitly outside the 3,015-candidate consumer bank. Full CPU/state/order verification, seed-paired contrasts, early audit counts and all endpoint records are retained. This ends numerical extensions for this round.

## Numerical integrity and resource limits

All 40 new CNN model/optimizer checkpoints pass reference, cloned-state, order, CPU-device, and exact detached-control checks. All 35 consumer audits reproduce healthy/frozen metrics and link to the original record/checkpoint hashes. All monitor partitions, 3,015 bounds, and choices are verified locally. The fixed consumer bank contains **35 paired training runs, 335 continuation endpoints, and 370 model/optimizer checkpoints**. The separate factorial adds 20 endpoint states, including five exact replays. Large trained states remain local and are excluded from the source artifact.

The 50 principal numerical jobs in this extension accumulated **3,707.7 seconds** of job wall time across parallel workers. Peak aggregate monitored RSS was **6,292,881,408 bytes** (5.86 GiB), with at most four single-thread workers. Final inspection across all **16** opaque directories, including earlier rounds and smoke checks, found **no active owned run process**. All **781** remote output/checkpoint/log files match local SHA-256 copies; total remote footprint is **439,556,273 bytes**. Uploaded bundles contain numerical Python source and no article documents. GPU visibility was disabled and saved parameters are CPU tensors. Connection details and operational mappings remain private under ignored `tmp/`.

Key public evidence is under `repro/results/CPUCONT/{frontier,cnn_consumer,head_state_first,head_state,recon_repairs}/`, with consumer, factorial and reconstruction-repair summaries/verification. Local regeneration uses `summarize_consumer_frontier.py`, `render_paper_figures.py` and `build_cpu_paper.py`. Full checkpoint validation with `verify_consumer_frontier.py` / `summarize_head_state.py` additionally requires the large local states; their verification JSON is shipped. The source artifact preserves earlier evidence, all new scientific records/logs, manuscript, figures, and verification; it excludes raw datasets and private machine information.

The next substantive research target is a real auxiliary with independently established primary-task benefit, followed by a predeclared prospective audit/intervention that preserves both that benefit and the old consumer. Wider architectures alone do not establish that result. This round improves evidence quality and operating scope without claiming that all research gaps, acceptance, or an award have been secured.
