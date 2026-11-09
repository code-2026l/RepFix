# ICLR manuscript: argument and language edit

The user has chosen the conference paper as the immediate target. This round
edits the complete manuscript and captions; it does not expand the experiments.

## Editorial decisions

- Center the paper on the distinction between task decodability and compatibility
  with an existing readout. Lead with the matched continuation and repair evidence.
- Explain the scale mechanism before the gradient diagnostic. Give each result a
  clear question, observation, and interpretation.
- Treat the reference-anchored confirmation as a constructive compatibility
  control with measured benefit and cost. Preserve attribution to prior work.
- Keep the CNN, frozen-trunk, nonlinear-side, calibration and protocol-dependent
  results. State each material limitation where it changes the interpretation;
  remove repeated disclaimers and experiment-history narration.
- Preserve all numerical evidence, mathematical assumptions, study distinctions,
  selection rules, fixed splits, uncertainty definitions and source citations.
- Keep the ICLR style and at most nine actual main-text pages. Edit captions for
  readable standalone interpretation without changing plotted data.

## Style references read

- Darcet et al., *Vision Transformers Need Registers*, ICLR 2024:
  https://arxiv.org/html/2309.16588v2
  Concrete incompatibility first, feature analysis next, intervention afterward.
  Award verified against the ICLR announcement:
  https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/
- Schaeffer et al., *Are Emergent Abilities of Large Language Models a Mirage?*,
  NeurIPS 2023: https://arxiv.org/html/2304.15004v2
  A specific alternative explanation followed by complementary tests; headings
  carry the argument. These are editorial references, not new evidence for RepFix.

No wording is copied from these papers and no performance or award comparison
is claimed. The current scientific evidence determines the strength of the prose.

## Delivery and validation

1. Save the incoming TeX/PDF under `tmp/editorial_20261109`.
2. Edit the title, abstract, introduction, main argument, appendix prose and captions.
3. Compare mathematical environments, numeric tokens, numerical inputs, figure
   hashes, labels and citations with the incoming manuscript; inspect all changes.
4. Compile locally, inspect warnings and render the complete PDF. Review main
   pages at reading size and every appendix page for layout defects.
5. Deliver the revised PDF, source package and concise editorial report locally.
   Do not push GitHub or submit the manuscript.
