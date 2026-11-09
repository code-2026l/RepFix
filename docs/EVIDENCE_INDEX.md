# Manuscript-to-evidence index

This index refers to *Silent Scale Shift: When Shared Updates Break Existing
Readouts*, the supplied 62-page manuscript. Figure numbers follow that PDF.
Paths are relative to the artifact root. `C` below abbreviates
`repro/results/CPUCONT/`.

## Principal evidence

| Question | Recorded evidence | Analysis / protocol |
|---|---|---|
| Can decodability survive a large frozen-readout loss? | `C/har/`, `C/replication/`, `C/projected/`, `C/summary.json` | `paired_readout_continuation.py`; Tables `tab:cpu_paired`, `tab:scale_audit` |
| How much can unlabeled scale repair recover? | `C/mechanism/`, `C/mechanism_summary.json` | `checkpoint_repair_audit.py`, `readout_alignment.py`; Table `tab:repair_hierarchy` |
| Does the failure occur while a real auxiliary improves? | `C/frontier/`, `C/task_validation/`, `C/recon_repairs/` | `reconstruction_task_audit.py`; Table `tab:reconstruction_full` |
| What does the CNN change? | `C/frontier/`, `C/cnn_consumer/` | `cnn_reconstruction_continuation.py`; Table `tab:cnn_continuation` |
| Does auxiliary-head state explain the historical SNR gap? | `C/head_state/`, `C/head_state_first/`, `C/head_state_summary.json` | `auxiliary_state_factorial.py`; Table `tab:head_state` |
| Can a useful auxiliary coexist with unchanged old logits? | `C/benefit_development/`, `C/benefit_development_choice.json`, `C/benefit_confirmation/`, `C/benefit_replay/` | `benefit_compatibility_cpu.py`; fixed protocol `docs/plans/2026-11-09-prospective-benefit-compatibility.md` |
| Does native-shape affine folding retain the benefit? | `C/compiled_confirmation/`, `C/compiled_replay/`, `C/compiled_interface_summary.json` | `compiled_interface_cpu.py`; fixed protocol `docs/plans/2026-11-09-compiled-interface.md` |
| Does a trainable nonlinear side pass its development gate? | `C/nonlinear_development/`, `C/nonlinear_development_choice.json`, `C/nonlinear_validation_replay/` | `nonlinear_interface_cpu.py`; fixed protocol `docs/plans/2026-11-09-nonlinear-interface.md` |
| What is the arithmetic correction? | `C/nonlinear_development_choice_float_bug.json`, corrected `C/nonlinear_development_choice.json` | `select_nonlinear_configuration.py`, `test_nonlinear_selection.py` |
| Can reference disagreement support a present-checkpoint decision? | `C/consumer_frontier_summary.json`, `C/mechanism/`, `C/cnn_consumer/` | `consumer_risk.py`, `checkpoint_consumer_audit.py`; Table `tab:consumer_audit` |
| When is the unsigned probe locally informative? | `repro/results/PBA/`, `C/certificate_matched.json` | `probe_boundary_audit.py`, `certificate_matched_audit.py`; Table `tab:probe_arms` |

Analysis and driver filenames in the last column are under `repro/scripts/`
unless a full path is given. The main text's 0.453 to 0.938 repair comparison
uses ten HAR continuation seeds. The reconstruction comparison uses five seeds;
the protected-benefit confirmation uses ten new seeds. These studies are separate.

## All 17 figures

Every file below is under `paper/figures/`. Use the coordinated entry point
`repro/scripts/render_paper_figures.py` to reconstruct them.

| Figure | PDF file | Generator | Main records |
|---:|---|---|---|
| 1 | `fig_readout_overview.pdf` | `plot_readout_story.py` | `C/har/`, `C/replication/`, `C/mechanism/`, `C/recon_repairs/` |
| 2 | `fig_nullspace_control.pdf` | `plot_paper_diagrams.py` | Analytic schematic, Equation `eq:consumer_nullspace` |
| 3 | `fig_probe_arms.pdf` | `plot_probe_arms.py` | `repro/results/PBA/`, `C/certificate_matched.json` |
| 4 | `fig_probe_phase.pdf` | `plot_probe_phase.py` | `repro/results/PBA/` |
| 5 | `fig_survival_gap.pdf` | `paper/revision/build_figures.py` | Real-head and `repro/results/P02/` records |
| 6 | `unified_all.pdf` | `plot_final_figures.py` | `results/BSCALE/`, `results/R7/`, `results/R5fin/`, `repro/results/` |
| 7 | `fig_probe_boundary.pdf` | `paper/revision/build_figures.py` | `results/WARM/`, `results/BSCALE/`, probe traces |
| 8 | `fig_diagnostic_intervention.pdf` | `paper/revision/build_figures.py` | `repro/results/_screena/`, real-head comparisons |
| 9 | `battery_repair.pdf` | `plot_final_figures.py` | Controlled battery records under `repro/results/` |
| 10 | `fig_readout_scale_law.pdf` | `plot_readout_story.py` | `results/READOUT/readout_scale_law.json` |
| 11 | `fig_paired_and_repair.pdf` | `plot_readout_story.py` | `C/mechanism/`, paired trajectories |
| 12 | `fig_benefit_confirmation.pdf` | `plot_benefit_confirmation.py` | `C/benefit_confirmation_summary.json`, confirmation records |
| 13 | `fig_compiled_control.pdf` | `plot_compiled_interface.py` | Analytic schematic, Equation `eq:compiled_affine` |
| 14 | `fig_compiled_frontier.pdf` | `plot_compiled_frontier.py` | `C/compiled_interface_summary.json` |
| 15 | `fig_nonlinear_interface.pdf` | `plot_nonlinear_study.py` | Analytic schematic, Equation `eq:nonlinear_fusion` |
| 16 | `fig_nonlinear_validation.pdf` | `plot_nonlinear_study.py` | `C/nonlinear_development_choice.json` |
| 17 | `fig_consumer_audit.pdf` | `plot_readout_story.py` | `C/consumer_frontier_summary.json` |

Generator basenames are under `repro/scripts/`. Figure manifests retain source
and numerical-artist hashes; diagrams are marked as schematics in their captions.

## Reading the records

Completed per-run JSON files are the primary numerical records. Redundant
`.partial.json` progress snapshots are omitted only when their completed run is
included. Complete null and adverse results remain in the package.

The nonlinear side study stops at its fixed development gate: no scientific test
confirmation or full-data latency measurement follows. The earlier floating-point
choice and exact-count correction are both retained. Confirmation seeds are kept
separate from development seeds and from other precision or label-budget regimes.

`*_verification.json` and replay JSON files record earlier checks of saved tensors.
The compact ZIP excludes those tensors; consult the reproduction guide before
attempting state-dependent replay. Historical result and manifest conventions,
including aggregate-only exceptions, are detailed in `results/README.md`.
