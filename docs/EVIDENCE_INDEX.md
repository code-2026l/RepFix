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

## All 20 figures

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
| 17 | `fig_chart_confirmation.pdf` | `plot_chart_confirmation.py` | `C/chart_confirmation/`, `C/chart_confirmation_summary.json` |
| 18 | `fig_protected_audit.pdf` | `plot_protected_audit.py` | `C/protected_head_audit/`, `C/protected_head_audit_summary.json` |
| 19 | `fig_alignment_confirmation.pdf` | `plot_alignment_confirmation.py` | `C/alignment_confirmation/`, `C/alignment_confirmation_summary.json` |
| 20 | `fig_consumer_contracts.pdf` | `plot_consumer_contracts.py` | `C/completion_har/`, `C/completion_summary.json` |
| 21 | `fig_consumer_audit.pdf` | `plot_readout_story.py` | `C/consumer_frontier_summary.json` |

Generator basenames are under `repro/scripts/`. Figure manifests retain source
and numerical-artist hashes; diagrams are marked as schematics in their captions.

## Reading the records

Completed per-run JSON files are the primary numerical records. Redundant
`.partial.json` progress snapshots are omitted only when their completed run is
included. Complete null and adverse results remain in the package.

The nonlinear side study stops at its fixed development gate: no scientific test
confirmation or full-data latency measurement follows. The earlier floating-point
choice and exact-count correction are both retained. Confirmation is held out from each study's development seeds. Some confirmation
seeds recur across studies, including protected-head seeds 300--309. Precision
and label-budget regimes are reported separately.

`*_verification.json` and replay JSON files record earlier checks of saved tensors.
The compact ZIP excludes those tensors; consult the reproduction guide before
attempting state-dependent replay. Historical result and manifest conventions,
including aggregate-only exceptions, are detailed in `results/README.md`.

## Integrated deployment studies

| Question | Records under `repro/results/CPUCONT/` | Driver / protocol |
|---|---|---|
| Does post-hoc folding preserve decisions as well as aggregate metrics? | `chart_development/`, `chart_confirmation/`, `chart_replay/` | `posthoc_chart_cpu.py`, `2026-11-09-deployment-chart.md` |
| Does an update invalidate vision decisions despite accurate readouts? | `vision_boundary/`, `vision_flows_900.json` through `902` | `vision_boundary_cpu.py`, `vision_decision_flows.py` |
| What does the original learned legacy head retain? | `protected_head/`, `protected_head_summary*.json` | `protected_head_cpu.py`, original SNR plan and HAR addendum |
| Does frozen-feature fitting explain joint distillation? | `protected_head_audit/`, `protected_head_audit_summary.json` | `audit_protected_head_cpu.py`, `legacy_head_fit.py`; retrospective fixed plan |
| Do saved control readers replay and satisfy their fitting conditions? | `protected_head_reader_replay/` | `verify_frozen_legacy_readers_cpu.py` |
| What does returning both current and legacy outputs cost? | `protected_head_timing_audit.json` | `benchmark_protected_outputs_cpu.py` |

The original HAR 9/10 and SNR 0/10 gates are preserved. The new HAR OLS 10/10
count is a fresh fixed-corpus audit of the same saved encoders, not a new
training confirmation. No large-model or information-erasure conclusion is
inferred from these controls. Source snapshots preserve exact recorded digests;
see `repro/source_versions/README.md` for the historical vision hash convention.

## New alignment comparison and fresh HAR seeds

| Question | Records under `repro/results/CPUCONT/` | Driver |
|---|---|---|
| How do complete-feature and logit alignment compare under the same map parameterization? | `alignment_development/`, `alignment_development_choice.json` | `joint_alignment_cpu.py`, `select_joint_alignment.py` |
| Does the construction meet the complete criterion on newly trained seeds? | `alignment_confirmation/`, `alignment_confirmation_summary.json` | `confirm_joint_alignment_cpu.py`, `summarize_joint_alignment.py` |
| Do all 50 saved consumers replay? | `alignment_replay/` | `verify_joint_alignment_cpu.py` |
| Does serving meet both diagnostic and native precision limits? | `alignment_timing.json`, `alignment_native_timing.json` | `benchmark_joint_alignment_cpu.py`, `benchmark_native_alignment_cpu.py` |
| Were all numerical states and source bytes retrieved/verified? | `alignment_execution_audit.json` | Recorded CPU ledgers and SHA-256 inventories |

Fresh HAR direct and OLS readers each certify 10/10 with the primary 1024-input
monitor and meet the complete fixed criterion. Logit-map 9/10 remains a failed
primary decision despite 10/10 with additional monitoring inputs. No SNR
candidate passes development. The original studies' decisions are unchanged.


## Final matched controls and deployment contracts

The fixed completion plan is `docs/plans/2026-11-10-final-evidence-completion.md`,
published before the new computations. The HAR comparison is retrospective.

| Question | Records under `repro/results/CPUCONT/` | Driver |
|---|---|---|
| Do strong feature and frozen-feature controls retain useful learning? | `completion_har/`, `completion_summary.json` | `complete_har_controls_cpu.py`, `summarize_completion.py` |
| Do class agreement, negative flips and fixed selective consumers coincide? | Same complete records | `consumer_contracts.py`, `plot_consumer_contracts.py` |
| Do all 100 recorded HAR readers replay? | `completion_replay/` | `verify_completion_cpu.py` |
| What does actual native serving cost? | `completion_timing.json` | `benchmark_completion_cpu.py` |
| Does a distinct coarse vision consumer retain compatibility? | `vision_consumer_screen/`, `vision_consumer_choice.json` | `vision_consumer_cpu.py`, `select_vision_consumer.py` |
| Do the three completed vision endpoints replay? | `vision_consumer_replay/` | `verify_vision_consumer_cpu.py` |
| Were sources, states, failure stops and resources verified? | `completion_execution_audit.json`, `logs/final_completion/` | Recorded SHA-256 inventories and bounded queue ledgers |

The vision screen stops with three completed endpoints, two nonfinite branches,
one interrupted branch and two unlaunched branches. No test confirmation follows.
Full-feature HAR alignment is competitive; auxiliary-learning readers' selective
counts remain 0/10 under the new monitor and bank. Detached + OLS has 4/10 and
3/10 without reconstruction progress. Earlier confirmation counts are unchanged.
See `docs/FINAL_REVIEW_20261110.md` and `docs/REVIEWER_GUIDE.md`.
