# RepFix

Silent Scale Shift: When Shared Updates Break Existing Readouts

This repository accompanies the manuscript and its full supplementary material. It contains editable LaTeX, vector figures, numerical records, analysis,
fixed study plans and review checks. Previous positive and negative studies remain.

An accurate current head does not show whether an existing readout still works.
Matched continuation and unlabeled repair isolate the scale component. Signed
update audits, specified class/threshold consumers, and actual serving programs
then separate task learning, interface fidelity and computation cost.

The fresh evidence includes useful HAR reconstruction, dense SNR auxiliary
learning at unrestricted-joint accuracy, and complete nonlinear-suffix transfer
on ResNet-18 and ViT-B/32. For vision, ten fresh continuation seeds per backbone
share one pretrained reference, split and label mask. Gains over the stronger
frozen-feature probe are 0.49 [0.37, 0.60] and
1.59 [1.49, 1.68] percentage points, respectively.
These are supervised-transfer results, separately from the auxiliary studies.
Full-precision and approximate reference implementations have distinct actual
consumer/cost records. The [scientific assessment](docs/FINAL_REVIEW_20261110.md)
reports every failed budget, negative control and remaining scope condition.

A separate confirmation varies four fitted ViT references, label masks and
train/validation partitions, with three continuation seeds per reference.
The gain over richer frozen features is 1.39 [0.80, 1.97] points, using four
reference means for the interval. Both real visual reconstruction pilots fail
their fixed auxiliary-benefit gate; conditional-reference routing also stops
at development. See the [follow-up report](docs/FOLLOWUP_REPORT_20261110.md).

![Shared computation and fresh transfer](paper/figures/fig_suffix_frontier.png)

## Paper and files

- [Final paper](paper/RepFix_ICLR2027.pdf): 9 main pages, 92 total.
- [Supplementary archive](artifacts/2026-11-10/RepFix_ReferenceAudit_20261110_supplement.zip)
- [Editable source and evidence](artifacts/2026-11-10/RepFix_ReferenceAudit_20261110_source.zip)
- [Reading guide](docs/REVIEWER_GUIDE.md) · [Evidence index](docs/EVIDENCE_INDEX.md)
- [Editorial and figure refinement](docs/EDITORIAL_FINISH_20261110.md) · [Integrity checks](docs/EDITORIAL_VISUAL_VERIFICATION_20261110.json)

## Verify the artifact

```text
python repro/scripts/verify_supplement.py
```

The default check uses the standard library to verify files, numerical records,
source/plan hashes, frozen choices and recorded outcomes. For the optional CPU
checks, install the review environment and run:

```text
python -m pip install -r requirements-review.txt
python repro/scripts/verify_supplement.py --tests --figures --paper
```

This runs in a temporary copy and preserves the extracted source. It checks 132
focused structural tests, reconstructs 27 manuscript figures from recorded data,
and compares the rebuilt paper's page text and pixels. LaTeX requires `pdflatex`,
`bibtex` and `pdftotext`. See the [reproduction guide](docs/REPRODUCIBILITY.md).

## Evidence and training

- [`repro/results/CPUCONT/`](repro/results/CPUCONT/): complete continuation,
  repair, distillation, nonlinear-scope and actual-serving records.
- [`docs/plans/`](docs/plans/): fixed protocols and byte-preserved historical versions.
- [`repro/scripts/`](repro/scripts/): numerical drivers, local analysis, plots and verification.
- [`paper/figures/`](paper/figures/): publication-width vector PDF/SVG and PNG previews.
- [`repro/source_versions/`](repro/source_versions/): exact executed source snapshots.
- [Historical result map](results/README.md) · [Current script entry points](repro/README.md)

Raw datasets, official weight files, trained tensors and large representation
caches are outside the compact archives. Recovered tensor and output checks are
recorded explicitly; the default verifier does not silently rerun a pretrained
backbone or claim to load those excluded inputs. The numerical CPU environment
and hardware are separate from the lightweight local review environment.
