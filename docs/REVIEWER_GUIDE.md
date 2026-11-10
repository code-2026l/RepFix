# Reading the current evidence

Start with the two main figures and the deployment-frontier table. Figure 1
separates useful task information from an old readout's scale-sensitive failure.
Figure 2 shows shared-prefix preservation and fresh ResNet-18/ViT-B/32 gains over
head-only and richer frozen-feature controls. The table adds actual old decisions
and matched CPU overhead rather than inferring deployment cost from MACs.

| Question | Read first | Recorded evidence |
|---|---|---|
| What fails under scale drift, and what does input repair restore? | Sections 1--2, Figure 1 | Matched HAR variance/reconstruction and fixed repair hierarchy |
| What does unsigned gradient conflict omit? | Section 3 | Signed variance/Adam audits, exact affine crossings and assumptions |
| Which nonlinear learning is useful? | Section 5, Figure 2 | Complete suffix and L/M/F/N/H controls, ten fresh seeds per selected backbone |
| Why retain a reference rather than simply repair or distill? | Scope/repair/control appendix | Nine affine repairs and fixed one-forward distilled-head controls, all outcomes |
| Which actual programs meet the cost and consumer gates? | Deployment table and serving appendix | Full FP32 and selected reference programs, actual outputs, isolated latency and export sizes |
| Does saved matrix rank identify a scientific law? | Rank/factorial appendix | Same-rank support comparison, SVD scope and gauge counterexample |
| What remains open? | FINAL_REVIEW_20261110.md | Independent references/splits, full-backbone auxiliary learning, online policy |
| Can the compact artifact be checked? | REPRODUCIBILITY.md | Manifest, structural tests, reconstructed figures and all-page comparison |

Class decisions, confidence-threshold decisions and complete logits are different
interfaces. Repairing new terminal features and retaining a healthy output path
also require different access. Visual gains here concern supervised transfer on
one fixed reference/split; the HAR/SNR studies provide separate real auxiliary-
learning evidence. The raw-state transport and local output checks say precisely
which computations are repeated and which inputs remain outside the ZIP.
