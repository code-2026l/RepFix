# RepFix

**Silent Scale Shift: When Shared Updates Break Existing Readouts**

This artifact accompanies the 62-page manuscript in
[`paper/RepFix_ICLR2027.pdf`](paper/RepFix_ICLR2027.pdf), with nine main-text pages.
It contains the source, complete recorded outcomes, analysis and figure scripts,
fixed study protocols, and structural checks.

The paper separates task decodability from compatibility with an existing readout.
Matched continuation and input repair isolate a scale-sensitive failure. Signed
update audits examine the diagnostic, while the SNR confirmation and single-encoder
controls measure the benefit and cost of preserving compatibility. The artifact
retains the CNN, frozen-trunk, nonlinear-side, and calibration boundary results.

## Paper and downloads

- [Read the final paper](paper/RepFix_ICLR2027.pdf) — 9 main pages, 62 pages total.
- [Download supplementary material](artifacts/2026-11-09/RepFix_Supplementary_20261109.zip)
- [Download manuscript source and evidence](artifacts/2026-11-09/RepFix_ICLR_Polished_20261109_source.zip)
- [Reproduction guide](docs/REPRODUCIBILITY.md) · [Evidence index](docs/EVIDENCE_INDEX.md)

![Readout compatibility overview](paper/figures/fig_readout_overview.png)

## Start here

From this directory, verify the extracted files and archived source/protocol links:

```text
python repro/scripts/verify_supplement.py
```

This first check uses the Python standard library and reads the shipped manifest.
It does not download data or train a model.

For the structural tests, figure reconstruction, and manuscript build:

```text
python -m pip install -r requirements-review.txt
python repro/scripts/verify_supplement.py --tests --figures --paper
```

The optional checks run in a temporary copy, leaving the extracted release intact.
The paper build also requires `pdflatex`, `bibtex`, and `pdftotext` on `PATH`.
See [the reproduction guide](docs/REPRODUCIBILITY.md) for each check and its inputs.

## Find the evidence

- [Evidence index](docs/EVIDENCE_INDEX.md): manuscript figures and claims mapped to
  recorded results, analysis scripts, and fixed protocols.
- [`repro/results/CPUCONT/`](repro/results/CPUCONT/): paired continuation, repair,
  real reconstruction, CNN, head-state, protected-benefit, compiled, and nonlinear
  studies. Frozen development choices and the exact-count correction are included.
- [`results/README.md`](results/README.md): detailed directory map for historical
  benchmarks; [`repro/README.md`](repro/README.md): current script entry points.
- [`docs/plans/`](docs/plans/): study protocols. Plans referenced by recorded hashes
  are retained byte for byte.
- [`paper/figures/`](paper/figures/): vector figures and editable SVG counterparts.

## What can be reproduced

The archive supports dataset-free file/provenance checks, 47 focused structural
tests, reconstruction of all 17 included figures from recorded results, and a local
LaTeX build. The validation choice and failed nonlinear development gate can be
inspected without rerunning training.

Raw datasets and trained tensor checkpoints are not included. Training and an
independent replay of saved model states require those additional inputs; shipped
replay records document the earlier completed checks. Dataset setup is described
in [the reproduction guide](docs/REPRODUCIBILITY.md#data-and-training).
Historical aggregate-only results are identified in `results/README.md`.

`ARTIFACT_MANIFEST.json` records the manuscript hash and a SHA-256 digest for every
payload file. `REVIEW_CHECKS.json` records the checks performed on this release.
The MIT license and third-party terms are in [`LICENSE`](LICENSE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Citation metadata is in
[`CITATION.cff`](CITATION.cff).
