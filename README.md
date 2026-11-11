# RepFix

RepFix studies what happens to a fixed classifier when its encoder is updated.
The repository contains the training and analysis code, experiment configurations,
recorded results, and the material needed to rebuild the figures and manuscript.

## Reproduce the results

Start with the file and provenance check. It uses the Python standard library
and the included records.

```sh
python repro/scripts/verify_supplement.py
```

To run the structural tests, rebuild figures, and compile the manuscript:

```sh
python -m pip install -r requirements-review.txt
python repro/scripts/verify_supplement.py --tests --figures --paper
```

The manuscript build requires pdflatex, bibtex, and pdftotext on PATH.
Training requires the original datasets or pretrained weights where specified
in the reproduction guide. The repository includes the reported per-run
results and does not redistribute those inputs or trained checkpoints.

## Repository layout

- [Training and evaluation code](code/) and [experiment configurations](configs/)
- [Reproduction guide](docs/REPRODUCIBILITY.md) and [script entry points](repro/README.md)
- [Evidence index](docs/EVIDENCE_INDEX.md), [historical results](results/README.md),
  and [experiment protocols](docs/plans/)
- [Manuscript source and figures](paper/)
- [License](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md)
