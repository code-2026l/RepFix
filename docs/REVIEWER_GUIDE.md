# Reading the current evidence

The main text follows failure mechanism, repair target and preservation
cost. Start with the two main figures and the deployment-frontier table. Figure 1
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
| Does transfer survive new reference fits and partitions? | Reference-replication appendix | Four fitted ViT references, twelve nested continuations; reference-level intervals |
| Can few reference queries estimate repair risk? | Small-budget query appendix | Four fitted references; direct-count and isotropic controls; point estimates separate from certification |
| Does a visual auxiliary help at matched update steps? | Cross-view appendix and deployment table | Four fitted ViT references and twelve matched continuations |
| Does the auxiliary help against stronger training-work controls? | Cross-view appendix and COMPUTE_CONTROL_RESULT_20261111.md | Both supervised controls across the full four-reference, three-continuation grid |
| Does the fixed recipe transfer to another pretrained checkpoint? | CHECKPOINT_REPLICATION_20261111.md | Four AugReg pairs with the same images, mask, views and continuation seed |
| Can existing anchors estimate disagreement? | Reusing the Fitting Anchors | Fixed-design ridge leave-one-out and transfer to contrastive updates |
| Can early predictions guide a fresh encoder update? | Early Decisions for a Planned Encoder Update, Figure 30 | Sealed epoch-two decisions on CIFAR-10, held-out outcomes, stability controls and limited-reference replay |
| Can strong updates use a cheaper old branch? | Legacy Precision Under Strong Updates, Figure 31 | Six official-test partitions, zero exact calls and paired physical CPU timing |
| What remains open? | Current result reports | Accuracy gains from the auxiliary on further tasks and live evaluation of limited referral |
| Can the compact artifact be checked? | REPRODUCIBILITY.md | Manifest, structural tests, reconstructed figures and all-page comparison |

Class decisions, confidence-threshold decisions and complete logits are different
interfaces. Repairing new terminal features and retaining a healthy output path
also require different access. The original visual test gains condition on one
reference and split per backbone. The supervised and cross-view ViT validation
studies vary four fitted references and partitions. The raw-state transport and
local output checks specify
which computations are repeated and which inputs remain outside the ZIP.
