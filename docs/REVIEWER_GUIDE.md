# Reading the paper

The paper asks whether an encoder update can retain its learning benefit while keeping an existing readout usable. The main text follows the failure mechanism, the consumer decision rule, and the cost of retaining it.

| Question | Main text | Supplement |
| --- | --- | --- |
| Can task information survive a failed readout? | Sections 1-2, Figure 1 | Matched continuation and repair hierarchy |
| Which updates contract the representation? | Section 3 | Diagnostic-boundary proofs and Adam counterfactuals |
| What does a repair preserve? | Sections 4-5 | Alignment, class/selective contracts and negative controls |
| Can healthy computation retain useful learning? | Section 5, Figure 2 and deployment table | Matched RadioML distillation; official CIFAR-100 routed program |
| Does routing help beyond additional calls? | Section 5 | Equal-call retrospective visual ablation |
| Can monitoring guide a pending update? | Section 6 | Separate early-warning studies and their task-specific failures |

Read the RadioML one-pass and final visual deployment subsections first for the main constructive results. Earlier support/rank screens, alternative serving precision and failed configurations provide boundary evidence. Complete grids, source hashes and numerical records remain in the artifact.

The official-test routed vision result uses one pretrained checkpoint and four fitted-reference groups with three continuations each. The AugReg result concerns the earlier fixed teacher recipe. Forecasting and retrospective routing controls are separate studies.
