# Reproducing the supplementary artifact

Run commands from the extracted `repfix_supplementary` directory. The supplied
PDF is the polished manuscript: nine main pages and 62 pages in total.

## File and record checks

```text
python repro/scripts/verify_supplement.py
```

Python 3.10 or newer is sufficient for this check. It verifies the manifest,
required manuscript inputs, source and protocol hashes stored in the CPU records,
and the frozen nonlinear decision. It reads records rather than loading model
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

The 47 tests cover finite-corpus disagreement bounds, repair availability, affine
scale boundaries, normalized-head nullspaces, constrained affine folding,
nonlinear fusion, and exact-count selection at a zero-benefit tie. They use
synthetic numerical inputs. Their success validates the stated algebra and
implementation controls; the experimental results are in the recorded studies.

## All manuscript figures

```text
python repro/scripts/verify_supplement.py --figures
```

The helper runs `repro/scripts/render_paper_figures.py` in a temporary copy. The
ten coordinated generators reconstruct all 17 included figures from shipped
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
