# Editorial and visual finish

The user asks for direct, idiomatic English throughout and refined scientific
figures, using the two attached architecture diagrams as visual references.
The experiments, selection rules, uncertainty, negative results and access
assumptions are fixed. This pass runs entirely locally.

## Design

Use restrained depth for computational objects: translucent stacked feature
planes, solid module faces and clearly separated data paths. Use flat statistical
graphics for measurements. A completely flat redesign would lose the reference
diagrams' spatial hierarchy; elaborate rendered 3D would compete with the evidence.
The selected combination keeps depth in mechanisms and precision in results.

All figures share Arial typography, a muted blue/teal/rose palette, thin neutral
axes, compact panel headings and editable PDF/SVG text. Color retains its semantic
meaning across figures. Each quantitative panel keeps its actual seed values and
uncertainty; typography changes must not change plotted data. Inspect every panel
at its final manuscript size, and rebuild the PDF after layout changes.

## Prose

Lead with the finding, explain the mechanism, and give the decisive comparison.
Remove repeated caveats, procedural narration and unnecessary negations. Put an
assumption where it affects interpretation; retain the experiment's actual scope
once rather than restating it in every paragraph. Keep theorem hypotheses,
failure outcomes, precision distinctions and the fixed-reference/split condition.
Use ordinary connected paragraphs, with no bold run-in sentence headings.

Read the full included TeX tree. Record equations, theorem statements, citations,
tables and numerical macros before editing, then check their integrity afterwards.
No new empirical claim or result is introduced by this pass.

## References for expression

- ICLR 2025 *Learning Dynamics of LLM Finetuning*:
  https://openreview.net/pdf/44b14ab32f5be2b5d680877df1e8821958c8d539.pdf
- ICLR 2025 *AlphaEdit*:
  https://proceedings.iclr.cc/paper_files/paper/2025/file/29c8c615b3187ee995029284702d3f43-Paper-Conference.pdf
- Nature artwork guidance: https://www.nature.com/documents/nature-final-artwork.pdf

Borrow the progression from observation to mechanism and evidence, and the
artwork principles of readable final-size labels, consistent panels and editable
vectors. Retain the conference manuscript format rather than copying journal
dimensions. The attached diagrams inform layering and connection design only.

## Delivery

Preserve the preceding release. Rebuild all figures, compile with nine main pages
and no warnings, inspect the main pages and appendix contact sheets, then run the
complete extracted-archive verification. Deliver PDF, source, supplement and
submission ZIP in one new local folder, with an updated GitHub draft PR.
