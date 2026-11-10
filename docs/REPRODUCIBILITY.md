# Reproducing the supplementary artifact

Run commands from the extracted `repfix_supplementary` directory. The supplied
PDF is the polished manuscript: nine main pages and 81 pages in total.

## File and record checks

```text
python repro/scripts/verify_supplement.py
```

Python 3.10 or newer is sufficient for this check. It verifies the manifest,
required manuscript inputs, source and protocol hashes stored in the CPU records,
the frozen nonlinear/chart decisions, and the protected-head original and
retrospective verdicts. Exact historical driver bytes are retained under
`repro/source_versions/`. Some early vision replay records hash `repr(bytes)`
instead of the raw bytes; the verifier checks and counts that convention
explicitly. It reads records rather than loading model
states or recomputing predictions.

## Review environment

```text
python -m pip install -r requirements-review.txt
```

`requirements-review.txt` lists dependencies for dataset-free review, including
pytest and PyMuPDF. `requirements.txt` describes the broader training environment.
The release was
checked with Python 3.12.12, PyTorch 2.5.1, NumPy 1.26.4, SciPy 1.11.4,
Matplotlib 3.10.9, and pytest 9.1.1. The optional checks use CPU and one numerical
thread. The hardware timings reported in the paper retain their original setup.

## Structural tests

```text
python repro/scripts/verify_supplement.py --tests
```

The 91 tests cover finite-corpus disagreement bounds, repair availability, affine
scale boundaries, normalized-head nullspaces, constrained affine folding,
nonlinear fusion, exact-count selection at a zero-benefit tie, invertible chart folding,
vision access/split controls, and teacher-head fitting on affine and nonlinear
examples. They use
synthetic numerical inputs. Their success validates the stated algebra and
implementation controls; the experimental results are in the recorded studies.

## All manuscript figures

```text
python repro/scripts/verify_supplement.py --figures
```

The helper runs `repro/scripts/render_paper_figures.py` in a temporary copy. The
coordinated generators reconstruct all 23 included figures from shipped
records. It compares plotted numerical artists and dimensions with the release
manifest. Files are exported as PDF, PNG, and SVG at a 5.5-inch width.

Figures use Times New Roman when available, with STIX/DejaVu fallback fonts.
Font availability can change rendering on another system while the numerical
artist checks remain meaningful. The included vector PDFs reproduce the delivered
manuscript's appearance.

## Manuscript build

```text
python repro/scripts/verify_supplement.py --paper
```

Install a LaTeX distribution supplying `pdflatex` and `bibtex`, and Poppler's
`pdftotext`. These executables must be on `PATH`. The ICLR style, bibliography,
numerical inputs, and figure files are included. The helper runs
`repro/scripts/build_cpu_paper.py` in a temporary copy, checks warnings and page
counts, and compares extracted text and 72-dpi page pixels with the delivered PDF.
PDF metadata timestamps can change during compilation.

All optional checks can run together:

```text
python repro/scripts/verify_supplement.py --tests --figures --paper
```

The original archive contents remain untouched. Each step prints its result;
`REVIEW_CHECKS.json` stores the checks performed before this package was delivered.

## Data and training

The following inputs are outside this archive:

| Study | Additional input | Entry point |
|---|---|---|
| HAR | UCI HAR training/test files | `code/cross_domain_mtl.py`; `repro/scripts/paired_readout_continuation.py` |
| RadioML | Prepared RadioML2016.10A I/Q, modulation labels, and SNR cache | `repro/scripts/paired_readout_continuation.py`; `repro/scripts/radioml_split_audit.py` |
| Battery | NASA PCoE/MATR data and the stated cell split | `repro/scripts/battery_validation_protocol.py` |
| NYUv2 | Prepared RGB-D/depth cache | `repro/scripts/nyu_prep.py`; `repro/scripts/nyu_bmtl.py` |
| CelebA | Prepared image and attribute inputs | `code/cross_domain_mtl.py` |
| Finance | Licensed factor cache | `data/DATA.md`; `code/wf_repro.py` |

The scientific loaders use `REPFIX_DATA_X` as the dataset root. Set it to your
prepared cache directory. HAR continuation expects
`uci_har/extracted/UCI HAR Dataset` beneath it. Consult each loader for its other
cache names; the RadioML split audit specifies the source-to-cache check. The
prepared RadioML partition is seeded, not an asserted unique official split.

The scripts and original configurations describe retraining. Use a separate
output location for new runs, preserving the released result files and frozen
development choices. Dataset acquisition and retraining are separate from the
dataset-free checks above. NYUv2 parquet preparation additionally requires
`pyarrow`, which is unnecessary for reviewing the shipped numerical records.

## Model-state replay

The source contains replay and state-audit drivers, and their completed numerical
records. Full independent model-state replay additionally needs the saved `.pt`
files from the corresponding training run. Those tensors are excluded from this
compact release. In particular, `summarize_benefit_compatibility.py`,
`summarize_compiled_interface.py`, and `summarize_nonlinear_interface.py` inspect
saved model/Adam states and should not be treated as dataset-free entry points.

The default verification checks available file and record hashes. It does not
substitute that check for a fresh replay of omitted tensor states. The evidence
index distinguishes paired seeds, separate protocols, development estimates,
confirmation results, and historical aggregates.

## Protected-head audit inputs

`protected_head_cpu.py` is the original frozen experiment. Its old timing
fields are retained but do not measure the cost of returning both outputs.
`audit_protected_head_cpu.py` replays all 20 confirmation records and computes
the fixed retrospective OLS/ridge readers on saved encoders.
`verify_frozen_legacy_readers_cpu.py` reloads the saved heads, checks metrics,
certificates and convex stationarity without refitting.
`benchmark_protected_outputs_cpu.py` measures current plus legacy outputs from
one encoder call, including the float64 OLS implementation.

`summarize_protected_audit.py` derives the release table and verdicts from the
shipped records. The archive records all 60 original and 140 control consumer
replays; the default verifier checks those records and hashes rather than
loading the model tensors. Original checkpoint files and fitted-head tensors
are retained locally outside the compact release. See
`docs/PROTECTED_HEAD_AUDIT_20261109.md` for access assumptions, the distinction
between original and retrospective evidence, and the timing correction.

## Joint alignment and fresh confirmation

The fixed development grid is in `alignment_development/`; its immutable
choice is `alignment_development_choice.json`. `joint_alignment_cpu.py` reuses
the original healthy checkpoints and paired future minibatches.
`confirm_joint_alignment_cpu.py` trains the new HAR seeds 1300--1309 only after
that gate. `verify_joint_alignment_cpu.py` independently reloads their saved
states; `benchmark_joint_alignment_cpu.py` measures the inherited diagnostic
path, and `benchmark_native_alignment_cpu.py` adds native float32 serving.

`summarize_joint_alignment.py`, `alignment_development_table.py` and
`plot_alignment_confirmation.py` derive the new statistics, tables and figure
from released records. The default archive verifier checks record/source/plan
links and preserved decisions; it does not replay tensors excluded from the
compact archive. `alignment_execution_audit.json` records the completed CPU
queues, verified transferred state hashes and uploaded numerical source hashes.
See `docs/FRESH_CONFIRMATION_20261110.md` for all criteria and adverse outcomes.


## Final completion and stopped vision screen

`complete_har_controls_cpu.py` consumes the saved fresh-confirmation record and
its original donor states. The lambda-1 feature control is chosen from prior
development outcomes under the utility-aware rule in the final completion plan.
`consumer_contracts.py` records disjoint correct/wrong flows and two fixed
confidence-threshold consumers. `summarize_completion.py` rebuilds the matched
table and paired intervals; `plot_consumer_contracts.py` rebuilds the vector figure.
`verify_completion_cpu.py` replays all 100 recorded endpoints; there are 90
distinct reader/model pairs because the inherited OLS fit is reproduced exactly.
`benchmark_completion_cpu.py` measures native current-plus-legacy serving.

`vision_consumer_cpu.py` decodes only the official CIFAR-100 training pickle
for the fixed development screen. Its prepare stage caches healthy features and
two heads; branch stages use the same seed-1800 checkpoint and minibatches.
The released selection script reads the stopped queue: three endpoints complete,
two coefficients become nonfinite, one branch is interrupted and two remain
unlaunched. No further development or official test evaluation follows.
`verify_vision_consumer_cpu.py` independently checks saved encoder/features,
readers, metrics, selective decisions and OLS stationarity.

`completion_execution_audit.json` records completed and deliberately stopped
queues, all transferred file hashes and historical source versions. Original
raw logs and 55 model-state files remain local outside the compact archive.
Scrubbed review logs are in `logs/final_completion/`. The default verifier checks
these recorded checks and preserved negative decisions; it does not load the
excluded tensors or rerun training. See `docs/FINAL_REVIEW_20261110.md` for scope.

## Shared-reference replay and consumer-cell studies

The immutable 2026-11-10 plans specify the fixed cell grids, class-only ablation,
shared-replay architecture, development choices, failed cost verdict and later
capacity check. `cell_development_stop.json` preserves the formal class-only
HAR eligibility and the separate futility stop; it does not report test outcomes.

`shared_replay_cpu.py` produces the six development records. The frozen choice
is `shared_replay_development_choice.json`. `shared_replay_confirm_cpu.py` trains
the 20 new confirmation seeds and full-update controls. `summarize_shared_replay.py`
computes paired intervals. `audit_shared_replay_cpu.py` reconstructs the two
functions from raw states; `benchmark_shared_replay_cpu.py` and the separate
Inductor pilot retain absolute times, ratios, setup and strong reuse controls.
The numerical runtime is CPU PyTorch 2.7.1; dataset-free review uses the local
environment listed above. Timing results are not hardware-independent.

`vision_shared_replay_cpu.py` and `vision_replay_capacity_cpu.py` produce the two
fixed ResNet grids; `audit_vision_replay_cpu.py` independently reloads all ten
completed endpoint states. Both grids stop before official test scoring. The
study is last-block primary-task adaptation, not full-backbone or auxiliary
transfer. `shared_reference_replay.py` and `shared_reference_vision.py` implement
the real-arithmetic identities and separately measured serving arithmetic.

`recovery_replay_execution_audit.json` records every new queue and file digest,
all 630 transferred `.pt` files, approximately 1.20 GB remote bytes, and the
retained technical failures. Source bytes that differ from current helpers
are preserved by digest under `repro/source_versions/`. Scrubbed technical
tracebacks are in `logs/shared_replay/`. These omitted tensor files are needed
for a fresh prediction replay; the compact verifier checks the shipped records
and hashes rather than claiming to load them.
