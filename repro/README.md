# Reproduction scripts and archived results

The current manuscript has **17 included figures**, rebuilt by ten coordinated
generators through one entry point. Start with the artifact-wide
[reproduction guide](../docs/REPRODUCIBILITY.md) and
[evidence index](../docs/EVIDENCE_INDEX.md).

From the artifact root:

```text
python repro/scripts/verify_supplement.py
python repro/scripts/verify_supplement.py --tests --figures --paper
```

The first command checks shipped hashes and recorded provenance with the Python
standard library. The optional checks run in a temporary copy, require the review
dependencies, and keep the extracted release intact. They run on CPU.

## Entry points

| Purpose | Script |
|---|---|
| Verify files, recorded protocol/source hashes, and optional review checks | `scripts/verify_supplement.py` |
| Rebuild all 17 manuscript figures from shipped records | `scripts/render_paper_figures.py` |
| Compile the supplied manuscript and check its page boundary | `scripts/build_cpu_paper.py` |
| Paired checkpoint training | `scripts/paired_readout_continuation.py` |
| Fixed readout repairs | `scripts/checkpoint_repair_audit.py`, `scripts/readout_alignment.py` |
| Reference-anchored compatibility training | `scripts/benefit_compatibility_cpu.py` |
| Frozen-trunk compiled control | `scripts/compiled_interface_cpu.py` |
| Compatible nonlinear-side development | `scripts/nonlinear_interface_cpu.py` |
| Fixed-corpus disagreement bound | `scripts/consumer_risk.py` |

Training and saved-state replay additionally need the specified datasets or tensor
checkpoints. The `summarize_*` state auditors inspect model/Adam files excluded from
this compact package. The default verification reads their archived records and
hashes; it does not perform a fresh tensor replay.

`results/CPUCONT/` holds the continuation and compatibility studies. The separate
root `results/` directory holds historical domain sweeps and readout audits; its
[README](../results/README.md) documents directory conventions and aggregate-only
exceptions. Complete negative outcomes and fixed development decisions are retained.

Older standalone plotting scripts and research notes remain available, but their
figure numbering and absolute-threshold scaling descriptions belong to earlier
versions. Use the coordinated renderer and current evidence index for this paper.
